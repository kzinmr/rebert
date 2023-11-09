# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import datasets as hf_datasets
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Convert dataset into MDS format, optionally concatenating and tokenizing"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--data_subset", type=str, default=None, help='E.g. "all" or "en"'
    )
    parser.add_argument("--splits", nargs="+", default=["train", "train_small", "val"])
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

    parsed = parser.parse_args()

    if (
        os.path.isdir(parsed.out_root)
        and len(set(os.listdir(parsed.out_root)).intersection(set(parsed.splits))) > 0
    ):
        raise ValueError(
            f"--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}."
        )

    return parsed


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: int
    truncated_samples: Union[int, None]


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits = {}

    def __iter__(self):
        for _, v in self.splits.items():
            yield v


class TrainSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "train",
        folder_split: str = "train_small",
        raw_samples: int = 1000000,
        truncated_samples: int = 100000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValSmallConstants(DataSplitConstants):
    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_small",
        raw_samples: int = 10000,
        truncated_samples: int = 10000,
    ):
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


def build_constants(
    train_raw_samples, val_raw_samples, val_chars_per_sample=-1, val_chars_per_token=-1
):
    # dataset.info.splits["train"].num_examples
    # dataset.info.splits["validation"].num_examples

    constants = DatasetConstants(
        chars_per_sample=val_chars_per_sample,
        chars_per_token=val_chars_per_token,
    )
    constants.splits["train"] = DataSplitConstants(
        hf_split="train",
        folder_split="train",
        raw_samples=train_raw_samples,
        truncated_samples=None,
    )
    constants.splits["val"] = DataSplitConstants(
        hf_split="validation",
        folder_split="val",
        raw_samples=val_raw_samples,
        truncated_samples=None,
    )

    constants.splits["train_small"] = TrainSmallConstants()
    constants.splits["val_small"] = ValSmallConstants()
    return constants


def build_c4_constants():
    return build_constants(
        train_raw_samples=364868892,
        val_raw_samples=364608,
        val_chars_per_sample=2163,
        val_chars_per_token=4,
    )


def build_sp_constants():
    return build_constants(
        train_raw_samples=-1,
        val_raw_samples=-1,
        val_chars_per_sample=-1,
        val_chars_per_token=-1,
    )


CONSTS = {
    "c4": build_c4_constants(),
    "slimpajama": build_sp_constants(),
}

DATAMAP = {
    "c4": {"dataset": "c4", "data_subset": "en"},
    "slimpajama": {"dataset": "cerebras/SlimPajama-627B", "data_subset": None},
}


def rename_args_for_dataset(args, dataset_name):
    args.dataset = DATAMAP[dataset_name]["dataset"]
    args.data_subset = DATAMAP[dataset_name]["data_subset"]


class NoConcatDataset(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(
        self, dataset_name: str, split: str, data_subset: Union[str, None] = None
    ):
        self.hf_dataset = hf_datasets.load_dataset(
            path=dataset_name, name=data_subset, split=split, streaming=True
        )
        self.redpajama_valid_set_names = {
            "RedPajamaCommonCrawl",
            "RedPajamaC4",
            "RedPajamaWikipedia",
        }

    def __iter__(self) -> Iterable[Dict[str, bytes]]:
        for sample in self.hf_dataset:
            # convert to bytes to store in MDS binary format
            if self.dataset_name == "cerebras/SlimPajama-627B":
                if (
                    sample["meta"]["redpajama_set_name"]
                    not in self.redpajama_valid_set_names
                ):
                    continue
            yield {"text": sample["text"].encode("utf-8")}

    def build_dataloader(
        self,
        batch_size,
    ) -> DataLoader:
        # Multiple workers is only supported on linux machines
        num_workers = 0
        if "linux" in platform.platform().lower():
            num_workers = min(64, self.hf_dataset.n_shards)  # type: ignore

        if num_workers > 0:
            # If using multiple workers, configure each worker to prefetch as many samples as it can, up to the aggregate device batch size
            prefetch_factor = max(1, 2 * batch_size // num_workers)
        else:
            # If not using workers, the torch DataLoader expects the default value for prefetch_factor, which non-intuitively must be 2.
            prefetch_factor = 2

        return DataLoader(
            dataset=self,
            sampler=None,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )


def generate_samples_for_mds(
    loader: DataLoader, truncate_num_samples: Optional[int] = None
) -> Iterable[Dict[str, bytes]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        first_batch = list(batch.values())[0]
        current_bs = len(first_batch)
        for batch_idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[batch_idx] for k, v in batch.items()}


def main(args: Namespace) -> None:
    """Main: create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    try:
        dataset_constants = CONSTS[args.dataset]
    except KeyError:
        raise ValueError(
            f'Constants for dataset "{args.dataset}" not found. Currently only "slimpajama" and "c4" are supported.'
        )

    columns = {"text": "str"}

    for split_name in args.splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError:
            raise KeyError(f"Constants not defined for split {split_name}.")
        # Only generate the splits requested
        if split.folder_split not in args.splits:
            continue

        # Get samples
        rename_args_for_dataset(args, args.dataset)
        dataset = NoConcatDataset(
            dataset_name=args.dataset,
            split=split.hf_split,
            data_subset=args.data_subset,
        )
        loader = dataset.build_dataloader(batch_size=512)
        samples = generate_samples_for_mds(
            loader, truncate_num_samples=split.truncated_samples
        )
        # Write samples
        print(f"Converting {split.folder_split} to MDS format...")
        denominator = (
            split.truncated_samples
            if split.truncated_samples is not None
            else split.raw_samples
        )
        outpath = os.path.join(args.out_root, split.folder_split)
        with MDSWriter(
            columns=columns, out=outpath, compression=args.compression
        ) as out:
            if denominator > 0:
                for sample in tqdm(samples, desc=split.folder_split, total=denominator):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=split.folder_split):
                    out.write(sample)


if __name__ == "__main__":
    main(parse_args())

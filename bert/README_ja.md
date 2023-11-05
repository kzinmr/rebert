# ようこそ！

このベンチマークは、BERT モデルの事前学習とファインチューニングの両方をカバーしています。このスターターコードを使えば、C4 データセットで [MLM 事前学習](#mlm-pre-training) を行い、[GLUE ベンチマークタスクでファインチューニング](#glue-fine-tuning)を行うことができます。また、私たちの[Mosaic BERT](#mosaic-bert)モデルのソースコードとレシピも提供しています。

## コンテンツ

### 事前学習

- `main.py` - YAML をパースし、[Composer](https://github.com/mosaicml/composer) Trainer をビルドし、ローカルまたは MosaicML プラットフォーム上で MLM の事前学習を開始するための簡単なスクリプトです。
- `yamls/main/` - Mosaic BERT と HuggingFace BERT を事前学習するための設定です。これらは `main.py` を実行する際に使用される。
- `yamls/test/main.yaml` - `main.py` の実行を素早く確認するための設定。

### ファインチューニング

- `sequence_classification.py` - ローカルまたは MosaicML 上で、独自のデータセットを用いて、分類タスクのファインチューニングを簡単に行うためのスタータースクリプトです。
- `glue.py` - より複雑なスクリプトで、YAML を解析し、8 つの GLUE タスク（ここでは WNLI タスクは除外しています）にまたがる多数のファインチューニング学習ジョブをオーケストレーションします。
- `src/glue/data.py` - GLUE のファインチューニングで `glue.py` が使用するデータセット。
- `src/glue/finetuning_jobs.py` - GLUE タスクごとに 1 つ、`glue.py` によってインスタンス化されるカスタムクラス。個々のファインチューニングジョブとタスク固有のハイパーパラメータを扱います。
- `yamls/finetuning/` - Mosaic BERT と HuggingFace BERT をファインチューニングするための設定。これらは `sequence_classification.py` と `glue.py` を実行する際に使用します。
- `yamls/test/sequence_classification.yaml` - `sequence_classification.py` の実行を素早く確認するための設定。
- `yamls/test/glue.yaml` - `glue.py` の実行を素早く確認するための設定。

### Shared

- `src/hf_bert.py` - MLM(事前学習)または分類(GLUE のファインチューニング)のための HuggingFace BERT モデル。[Composer Trainer](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.Trainer.html#composer.Trainer)との互換性のために、[`ComposerModel`s](https://docs.mosaicml.com/en/stable/api_reference/generated/composer.models.HuggingFaceModel.html)でラップされています。
- `src/mosaic_bert.py` - MLM（事前学習）または分類（GLUE のファインチューニング）用のモザイク BERT モデル。詳細は[Mosaic BERT](#mosaic-bert)を参照してください。
- `src/bert_layers.py` - HuggingFace API との互換性を考慮した、独自の高速化メソッドを組み込んだ Mosaic BERT レイヤ/モジュール
- `src/bert_padding.py` - パディングのオーバーヘッドを回避するための Mosaic BERT ユーティリティ
- `src/flash_attn_triton.py` - Mosaic BERT で使用される [FlashAttention](https://arxiv.org/abs/2205.14135) 実装のソースコード
- `src/text_data.py`- [MosaicML ストリーミングデータセット](https://streaming.docs.mosaicml.com/en/stable/)
- `src/convert_dataset.py` - HuggingFace のテキストデータセットを[MosaicML ストリーミングデータセット](https://streaming.docs.mosaicml.com/en/stable/)に変換するスクリプト
- `requirements.txt`
- `requirements-cpu.txt`

## クイックスタート

### システム推奨環境

以下の環境を推奨します：

- NVIDIA GPU を搭載したシステム
- [MosaicML の PyTorch ベースイメージ](https://hub.docker.com/r/mosaicml/pytorch/tags)を実行する
  - Docker コンテナ： `mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04`.

この推奨 Docker イメージには、以下の依存関係が事前に設定されています：

- PyTorch バージョン：1.13.1
- CUDA バージョン：11.7
- Python バージョン：3.10
- Ubuntu バージョン：20.04

## Prepare your data

(ローカルに保存されている、またはクラウドストレージからダウンロードするのに時間がかからない小さなデータセットをお持ちの場合は、このセクションをスキップできます).\_.

このベンチマークでは、[C4: Colossal, Cleaned, Common Crawl dataset](https://huggingface.co/datasets/c4)で BERT を学習する。C4 で事前学習を実行するには、このデータセットのコピーを自分で作成する必要がある。

あるいは、スクリプト[main.py](./main.py#L98)の中で、私たちの dataloader を自由に置き換えてください。
ファインチューニングに移るときは、スクリプト[sequence_classification.py](./sequence_classification.py#L63)に追加することで、独自のデータセットで学習することができます。
とりあえず、事前学習のために C4 データを準備することに集中しましょう。

まず、データセットをネイティブフォーマット（zip 圧縮された JSON の集合）から
を MosaicML のストリーミングデータセットフォーマット（バイナリの`.mds`ファイルの集まり）に変換します。
.mds`フォーマットになったら、データセットを中央の場所（ファイルシステム、S3、GCS など）に保存し、任意の計算機にデータをストリーミングできる。
データセットを中央の場所（ファイルシステム、S3、GCS など）に保存し、任意の数のデバイスと任意の数の CPU ワーカーを持つ任意のコンピュートクラスタにデータをストリーミングすることができる。
mosaicml-streaming を使用する利点については、[こちら](https://streaming.docs.mosaicml.com/en/stable/)を参照してください。

### C4 をストリーミングデータセット `.mds` フォーマットに変換する

C4 のコピーを作成するには、`convert_dataset.py`を使います：

```bash
# train_small'と'val'の分割データをダウンロードし、StreamingDataset形式に変換する。
# インターネットの帯域幅にもよりますが、20-60秒かかります。
# ./my-copy-c4/train_small`と `./my-copy-c4/val`の2つのフォルダが表示されるはずで、それぞれ ~0.5GB です。
# 注意: BERTではサンプルの連結を行わないため、ここでは `--concat_tokens` オプションを使用しません。
# その代わりに、サンプルは単純にパディングされるか、最大シーケンス長に切り詰められます。

python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val

# プロファイルだけでなく）本当にモデルを学習したい場合は、'train'スプリットをダウンロードします。
# 帯域幅やCPUなどに依存するが, 1時間から数時間かかるだろう.
# 最終的なフォルダ `./my-copy-c4/train` は~800GBになるので、容量を確保してください！
# python src/convert_dataset.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train

# 上記のどのコマンドでも、.mdsファイルを圧縮することもできます。
# これは、変換後にオブジェクトストアに保存する予定がある場合に便利です。
# python src/convert_dataset.py ... --compression zstd
```

複数の学習実行を計画している場合、作成した C4 の**ローカル**コピーを中央の場所にアップロードすることができます。こうすることで、将来的にデータセットの準備ステップを省略することができます。そうしたら、`yamls/main/` の YAML を修正して、`data_remote` フィールドが新しい場所を指すようにする。そうすれば、ローカルコピーを作成する代わりに、単純にデータセットをストリーミングできるようになる！

### データローダーのテスト

dataloader が動作することを確認するために、`val`スプリットで次のような簡単なテストを実行します：

```bash
#これは `val` スプリットから `StreamingTextDataset` データセットを作成し、 PyTorch Dataloader に渡して、それを繰り返し処理してサンプルを表示します。
# ローカルパスを指定するだけなので、ストリーミングやコピーは行われません。
python src/text_data.py --local_path ./my-copy-c4 --tokenizer bert-base-uncased

# これは同じことをしますが、{remote}から{local}にデータをストリームします。
# リモートパスはファイルシステムまたはオブジェクトストアのURIです。
# 例えば遅いNFSボリュームから速いローカルディスクへのコピー
python src/text_data.py --local_path /tmp/cache-c4 --remote_path ./my-copy-c4 --tokenizer bert-base-uncased
# オブジェクトストアからのストリーム
# python src/text_data.py --local_path /tmp/cache-c4 --remote_path s3://my-bucket/my-copy-c4 --tokenizer bert-base-uncased
```

データが準備できたので、学習を開始しよう。

### 事前学習のテスト

事前学習が正しく実行されることを確認するために、まず C4 validation split のローカルコピーを用意し（上のセクションを参照）、テスト用の設定を使用して `main.py` 事前学習スクリプトを 2 回実行します。
まず、ベースラインの HuggingFace BERT を使用します。次に、Mosaic BERT を使用します。

```bash
# テスト設定と HuggingFace BERT を使って事前学習スクリプトを実行します。
composer main.py yamls/test/main.yaml

# テスト設定と Mosaic BERT を使用して事前学習スクリプトを実行する。
composer main.py yamls/test/main.yaml model.name=mosaic_bert
```

### ファインチューニングのテスト

ファインチューニングが正しく実行されることを確認するために、テスト用コンフィグと HuggingFace と Mosaic BERT モデルの両方を使用してスクリプトを実行します。
最初に、`sequence_classification.py` をベースラインの HuggingFace BERT で検証し、再度 Mosaic BERT で検証します。

```bash
# テスト構成と HuggingFace BERT でファインチューニングスクリプトを実行
composer sequence_classification.py yamls/test/sequence_classification.yaml

# テスト設定と Mosaic BERT を使用してファインチューニングスクリプトを実行
composer sequence_classification.py yamls/test/sequence_classification.yaml model.name=mosaic_bert
```

次に、両方のモデルの `glue.py` を検証します。

```bash
# テスト設定と HuggingFace BERT で GLUE スクリプトを実行
python glue.py yamls/test/glue.yaml && rm -rf local-finetune-checkpoints

# テスト設定と Mosaic BERT で GLUE スクリプトを実行
python glue.py yamls/test/glue.yaml model.name=mosaic_bert && rm -rf local-finetune-checkpoints
```

## 実際の学習

依存関係のインストールと C4 データセットのローカルコピーのビルドが完了したので、学習を開始しましょう！まずは C4 上で MLM の事前学習を行います。

**注意:** YAML のデフォルトでは、上記の例に従って `./my-copy-c4` を C4 データセットの場所として使用します。データセットの保存場所に違いがある場合（別のフォルダ名を使用した場合や、コピーをすでに S3 に移動した場合など）は、事前学習 YAML の **`data_remote`と`data_local`のパスを編集する** ことを忘れないでください。

`train_small`スプリットだけをダウンロードした場合は、train_dataloader がそのスプリットを指していることを確認する必要があります。
YAML の `split: train` を `split: train_small` に変更してください。
これはテスト用の YAML `yamls/test/main.py` で既に行われており、設定をテストするために使用することもできます（[Test pre-training](#test-pre-training) を参照してください）。

### MLM pre-training

pre-training の予算を最大限に活用するために、**Mosaic BERT** を使用することをお勧めします！詳しくは[下記](#mosaic-bert)を参照してください。

`main.py`事前学習スクリプトは、N 個のプロセス(GPU デバイスあたり 1 プロセス)を生成する `composer` ランチャーを使って実行します。
単一ノードで学習する場合、`composer` ランチャーはデバイスの数を自動検出します。

```bash
# これは、ダウンストリームの GLUE 精度が約 83.3%に達する HuggingFace BERT を事前に学習します。
# 8つのA100_80g GPUを搭載したシングルノードで約11.5時間かかります。
composer main.py yamls/main/hf-bert-base-uncased.yaml

# これは、およそ1/3の時間で同じダウンストリーム精度に到達するMosaic BERTを事前に学習します。
composer main.py yamls/main/mosaic-bert-base-uncased.yaml
```

\*\*保存とロードの場所をカスタマイズするために、参照する YAML (例えば、`yamls/main/mosaic-bert-base-uncased.yaml`) を修正することを忘れないでください。 `yamls/test/` にある YAML だけがすぐに使える。詳細は[configs](#configs)セクションを参照。

NOTE: docker 内での実行時には `shm-size` 制限に引っかかることがあるためホスト側の設定を確認すること。

### シングルタスクのファインチューニング

事前学習の後はファインチューニングです。私たちは、独自のカスタムデータセット上で事前学習された BERT モデルのファインチューニングを簡素化するための便利なスタータースクリプトとして`sequence_classification.py`を提供しています。**データセットをプラグインしてこのスクリプトを修正するだけで、気になるタスクで BERT モデルをファインチューニングすることができます。**

スタータースクリプトを修正した後、参照 YAML（例えば `yamls/finetuning/mosaic-bert-base-uncased.yaml`）を更新して変更を反映させる。準備ができたら `composer` ランチャーを使用する。

```bash
# カスタム分類タスクで BERT モデルをファインチューニングします！
composer sequence_classification.py yamls/finetuning/mosaic-bert-base-uncased.yaml
```

### GLUE fine-tuning

GLUE ベンチマークは、8 つの NLP 分類タスク（ここでも WNLI タスクは除外します）の平均性能を測定します。MLM タスクから重みのセットが得られたら、各タスクに対して個別に重みをファインチューニングし、タスク全体の平均パフォーマンスを計算します。

この複雑なファインチューニングパイプラインを処理するために、`glue.py` スクリプトを提供します。

このスクリプトは、マシン上の全ての GPU で、それぞれのファインチューニングジョブの並列化を処理します。
つまり、`glue.py`スクリプトは、データ並列ではなく、_タスク_ 並列によって、GLUE タスクの小さなデータセット/バッチサイズを利用します。これは、GPU 間で並列化されたバッチで一度に 1 つのタスクを学習させるのではなく、1 つの GPU を使用して異なるタスクを並列に学習させることを意味します。

**Note:** `glue.py` スクリプトを使い始めるには、まず、開始チェックポイントフィールドが `main.py` によって保存された最後のチェックポイントを指すように、各参照 YAML を更新する必要があります。詳しくは [configs](#configs) セクションを参照してください。

`yamls/glue/`の YAML を修正して、GLUE の開始点として事前に学習したチェックポイントを参照するようにしたり、デフォルト以外のハイパーパラメータを使用するようにしたりしたら、標準の `python` ランチャーを使用して `glue.py` スクリプトを実行します
(ここでは、`glue.py` が独自のマルチプロセスオーケストレーションを行うため、`composer` ランチャーは使用しません)：

```bash
# これは HuggingFace BERT 上で GLUE のファインチューニング評価を実行します。
python glue.py yamls/finetuning/glue/hf-bert-base-uncased.yaml

# これは Mosaic BERT で GLUE のファインチューニング評価を実行します。
python glue.py yamls/finetuning/glue/mosaic-bert-base-uncased.yaml
```

GLUE のスコアはスクリプトの最後に出力され、YAML で有効になっていれば Weights and Biases を使って追跡することもできます。
その他の [composer がサポートしているロガー](https://docs.mosaicml.com/en/stable/trainer/logging.html#available-loggers) も簡単に追加できます！

公平な警告: `glue.py` の内部で起動されたすべてのプロセスは、学習中に独自のプリントアウトを生成します。そのため、コンソールが少しカオスに見えても驚かないでください。それは動いているということです :)

## Configs

このセクションは、この README を通して参照され、`yamls/`にあるコンフィグ YAML を説明するためのものです。

YAML の使い方について簡単に説明しておきます：

- YAML を使うのは、すべての設定を明示的にするためです。
- Python ファイルで直接好きなように設定することもできます。
- YAML ファイルは特別なスキーマやキーワード、その他のマジックを使いません。YAML ファイルはスクリプトが読み取るための dict を記述するクリーンな方法です。

言い換えると、このスターターコードをあなたのプロジェクトに合わせて自由に変更でき、ワークフローで YAML を使うことに縛られません。

### main.py

`yamls/main/` にある設定を使用する前に、`main.py` を実行します：

- `save_folder` - モデルのチェックポイントの保存場所を指定します。これは `run_name` に依存することに注意してください。例えば、 `save_folder` を `s3://mybucket/mydir/{run_name}/ckpt` に設定すると、 `{run_name}` は `run_name` の値に置き換わります。そのため、複数の学習実行で同じラン名を再利用することは避けるべきである。
- `data_remote` - ストリーミング C4 ディレクトリのファイルパスを設定する。[ Prepare your data](#prepare-your-data)の指示に従い、ローカルに C4 のコピーを作成した場合、デフォルト値の `./my-copy-c4` が機能します。データセットを中央の場所に移動した場合は、`data_remote` に新しい場所を指定するだけでよい。
- `data_local` - データセットがストリームされるローカルディレクトリへのパスである。`data_remote` がローカルの場合、 `data_local` にも同じパスを使用することができる。`./my-copy-c4` のデフォルト値は、このようなローカルコピーで動作するように設定されている。データセットを中央の場所に移動した場合は、`data_local` を `/tmp/cache-c4` に設定するとうまくいくはずである。
- `loggers.wandb` (オプション) - W&B にログを記録したい場合は、`project` と `entity` フィールドを埋めるか、このロガーを使用したくない場合は `wandb` ブロックをコメントアウトする。
- `load_path` (オプション) - チェックポイントから開始したい場合は、このように設定します。

### sequence_classification.py

`sequence_classification.py`を実行する際に、`yamls/finetuning/`にある設定を使用する前に、以下の項目を入力する必要があります：

- `load_path` (オプション) - チェックポイントから開始したい場合は、このように設定します。Mosaic BERT をファインチューニングする場合、これは空のままにしてはならない。
- `save_folder` - モデルのチェックポイントの保存場所を指定します。これは `run_name` に依存することがあることに注意してください。例えば、`save_folder` を `s3://mybucket/mydir/{run_name}/ckpt` に設定すると、`{run_name}` が `run_name` の値に置き換わります。そのため、複数の学習実行で同じラン名を再利用することは避けるべきである。
- `loggers.wandb` (オプション) - W&B にログを記録したい場合、`project` と `entity` フィールドを埋めるか、このロガーを使用したくない場合は `wandb` ブロックをコメントアウトする。
- `algorithms` (オプション) - pre-training の前にチェックポイントモデルに適用された、アーキテクチャを変更するアルゴリズムを含める。例えば、事前学習で `gated_linear_units` をオンにした場合、ファインチューニングでもオンにする必要があります！

### glue.py

glue.py を実行する際に、`yamls/finetuning/glue/`にある設定を使用する前に、以下の項目を埋める必要があります：

- `starting_checkpoint_load_path` - ファインチューニングを行うときにどのチェックポイントから開始するかを指定します。これは `<save_folder>/<checkpoint>` のようになるはずです。ここで `<save_folder>` は事前学習設定で設定した場所です（上のセクションを参照）。
- `loggers.wandb` (オプション) - W&B にログを記録したい場合は、`project` と `entity` フィールドを入力します。このロガーを使用したくない場合は、`wandb` ブロックをコメントアウトします。
- `base_run_name` (オプション) - 複数の実行で同じ名前を再使用しないように注意する。
- `algorithms` (オプション) - 事前学習の前にチェックポイントモデルに適用された、アーキテクチャを変更するアルゴリズムを含める。例えば、事前学習で `gated_linear_units` をオンにした場合は、ファインチューニングの際にもオンにします！

## MosaicML プラットフォームでの実行

MosaicML プラットフォームで動作するように計算クラスタを設定した場合、`yaml/*/mcloud_run*.yaml` リファレンスの YAML を使って、リモートで pre-training と fine-tuning を実行することができます！

不足している YAML フィールドを入力したら（そして必要な修正を加えたら）、次のように実行するだけで pre-training を開始することができます：

```bash
mcli run -f yamls/main/mcloud_run_a100_80gb.yaml
```

あるいは、クラスタに A100 GPU と 40GB のメモリがある場合は、次のように実行します：

```bash
mcli run -f yamls/main/mcloud_run_a100_40gb.yaml
```

同様に、シーケンス分類のファインチューニングを行う場合は、不足している YAML フィールドを埋めて（例えば、事前学習チェックポイントを開始点として使用する）、実行するだけです：

```bash
mcli run -f yamls/finetuning/mcloud_run.yaml
```

GLUE のファインチューニングも同様です。不足している YAML フィールドを埋めて実行します：

```bash
mcli run -f yamls/finetuning/glue/mcloud_run.yaml
```

### マルチノードの学習

マルチノードクラスターで高性能な学習を行うには、MosaicML プラットフォームを使うのが一番簡単です ;)

しかし、もし自分のクラスタで手動で試したいのであれば、いくつかの変数を `composer` に与えるだけでよい。それから各ノードで適切なコマンドを実行する。

`glue.py`スクリプトはデバイス間のオーケストレーションを処理するもので、`composer`ランチャーで使用するようには作られていません。

#### CLI 引数を使ったマルチノード

```bash
# それぞれ8つのデバイスを持つ2つのノードを使用する。
# 合計ワールドサイズは16
# ノード 0 の IP アドレス = [0.0.0.0］

# ノード 0
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/main/mosaic-bert-base-uncased.yaml

# ノード1
composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 7501 main.py yamls/main/mosaic-bert-base-uncased.yaml

```

#### 環境変数によるマルチノード

```bash
# それぞれ8デバイスの2ノードを使用
# ワールドサイズの合計は 16
# ノード 0 の IP アドレス = [0.0.0.0］

# ノード 0
# export WORLD_SIZE=16
# export NODE_RANK=0
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
# ノード 1

# ノード 1
# export WORLD_SIZE=16
# export NODE_RANK=1
# export MASTER_ADDR=0.0.0.0
# export MASTER_PORT=7501
composer main.py yamls/main/mosaic-bert-base-uncased.yaml
```

ターミナルにログが出力されるはずです。
[Composer's logging integrations](https://docs.mosaicml.com/en/stable/trainer/logging.html)を使えば、Weights and Biases や CometML のような他の実験トラッカーも簡単に有効にできます。

## Mosaic BERT

私たちのスターターコードは、標準的な HuggingFace BERT モデルと私たち独自の **Mosaic BERT** の両方をサポートしています。後者には、スループットと学習を改善するための多くの方法が組み込まれています。
Mosaic BERT の開発における私たちの目標は、学習時間を大幅に短縮すると同時に、自分の問題で簡単に使用できるようにすることでした！

そのために、文献にある多くの手法を採用しています：

- [ALiBi (Press et al., 2021)](https://arxiv.org/abs/2108.12409v1)
- [GLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [アンパディングトリック(The Unpadding Trick)](https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/fmha.py)
- [FusedLayerNorm (NVIDIA)](https://nvidia.github.io/apex/layernorm.html)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

... そして、それらを連携させる！私たちの知る限り、これらの手法の多くはこれまで組み合わされたことがない。

あなたがこれを読んでいるなら、私たちはまだ、同等の HuggingFace BERT モデルと比較して、Mosaic BERT が提供する正確なスピードアップと性能向上をプロファイリングしているところです。今後の結果にご期待ください！

## お問い合わせ

コードで何か問題が発生した場合は、このレポに直接 Github issue を提出してください。

MosaicML プラットフォーム上で BERT スタイルのモデルを学習したい場合は、[demo@mosaicml.com](mailto:demo@mosaicml.com)までご連絡ください！

# fMRI オープンデータ解析講習

fMRI の公開データを、Google Colab と Python のライブラリ Nilearn で解析する実習の教材です。解析の経験がない人を
想定しています。ブラウザと Google アカウントがあれば、手元に何も入れずに動かせます。

## ノートブック

番号の順に進みます。「Colab で開く」を押すと、Google Colab でノートブックが開きます。

| | 内容 | 開く |
|---|---|---|
| 01 | 1 人分のデータに GLM を当てはめ、顔条件で強く反応する場所を探す。データが撮られてから解析されるまでの流れ（BIDS、OpenNeuro、fMRIPrep）も説明します | [Colab で開く](https://colab.research.google.com/github/kaosu-research/fmri-opendata-workshop/blob/main/notebooks/01_first_level_glm.ipynb) |
| 02 | 同じデータで、脳の活動のパターンから、顔条件と図形条件のどちらを行っていたかを当てる（デコーディング） | [Colab で開く](https://colab.research.google.com/github/kaosu-research/fmri-opendata-workshop/blob/main/notebooks/02_decoding.ipynb) |
| 03 | 心の理論の課題のデータで、自閉スペクトラム症（ASD）の人と定型発達の人の脳を、デコーディングで比べる | [Colab で開く](https://colab.research.google.com/github/kaosu-research/fmri-opendata-workshop/blob/main/notebooks/03_false_belief_asd.ipynb) |
| 04 | 安静時の脳のつながりから、ASD の人と定型発達の人を見分ける | [Colab で開く](https://colab.research.google.com/github/kaosu-research/fmri-opendata-workshop/blob/main/notebooks/04_abide_asd.ipynb) |
| 05 | まとめ（コードはありません） | [Colab で開く](https://colab.research.google.com/github/kaosu-research/fmri-opendata-workshop/blob/main/notebooks/05_summary.ipynb) |

## 用意するもの

- ノートパソコン（OS は問いません）と、新しいバージョンのブラウザ（Chrome、Firefox、Safari など）。
- Google Colab が使える Google アカウント。大学などの組織のアカウントでは、管理者の設定で Colab が使えないことが
  あります。そのときは、個人の Google アカウントを使ってください。
- インストールするものはありません。

## 使い方

1. 上の表の「Colab で開く」から、ノートブックを開きます。
2. メニューの「ランタイム」→「すべてのセルを実行」を選ぶか、セルを選んで Shift + Enter で上から順に実行します。
3. 最初に実行するとき、「このノートブックは Google が作成したものではありません」という警告が出ます。「このまま実行」を
   押してください。
4. 書き換えた内容を残したいときは、「ファイル」→「ドライブにコピーを保存」を選びます。保存しないと、閉じたときに変更は消えます。

- ランタイムの種類は CPU のままで動きます（GPU は使いません）。
- データは、ノートブックの中で、公開されている置き場所から取ってきます。取ってくるのは Colab のサーバなので、手元の
  回線はほとんど使いません。
- しばらく操作しないと接続が切れ、取ってきたデータも消えます。そのときは「ランタイム」→「すべてのセルを実行」で、
  最初からやり直せます。

## データ

データはこのリポジトリには入っていません。出典は、各ノートブックの最後にあります。

| ノートブック | データ | 利用の条件 |
|---|---|---|
| 01・02 | AOMIC PIOP1（OpenNeuro ds002785） | CC0 |
| 03 | OpenNeuro ds000212（前処理済みのデータは OpenNeuro Derivatives） | CC0 |
| 04 | ABIDE I（Preprocessed Connectomes Project が前処理したもの） | 非営利の目的に限る（CC BY-NC-SA 3.0） |

## このリポジトリにあるもの

```
notebooks/   ノートブック 01〜05
tools/       ノートブックの外で行った確認の計算（04 の 6 章と補足で触れた並べ替え検定）
```

## ライセンス

- コード：MIT License（[LICENSE](LICENSE)）
- 文章と図：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja)
- ただし、出典を示した他者の図（01・02 の課題の図）は、それぞれの条件に従います。

## 動作を確かめた環境

Google Colab（無料枠、CPU）の新しいランタイム：Python 3.13.16、nilearn 0.13.1、scikit-learn 1.6.1、numpy 2.1.3、
pandas 2.2.3、scipy 1.16.3

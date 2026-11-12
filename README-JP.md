# Dataset Tag Editor Standalone

[English Readme](README.md)

> **独立してメンテナンスされているフォークです。** 上流 ([toshiaki1729/dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)) は **2024年7月**から更新が止まっているため、このフォークを独立して維持しています。uv による自己完結インストール（システム Python を汚染しない）、**Smart Tag Dedup**（オフラインルールまたはローカル LLM）、**booru タグ自動補完**（a1111-sd-webui-tagcomplete 互換）、大規模データセット向けの**パフォーマンス**改善を追加しています。起動は `launch_uv.bat` (Windows) または `webui_uv.sh` (Linux/macOS) を使用してください。

web UI 上で学習用データセットのキャプションを編集できるようにします。  
こちらは、[Dataset Tag Editor](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor)のスタンドアロン版です（元のプログラムは[Stable Diffusion web UI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)用の拡張機能です）。


![](pic/ss01.png)

DeepBooru interrogator で生成したような、カンマ区切り形式のキャプションを編集するのに適しています。

キャプションとして画像ファイル名を利用している場合も読み込むことができますが、保存はテキストファイルのみ対応しています。


## 拡張機能版との違い
(Pros)  
- gradioの特定のバージョンに起因するバグの回避
- 起動と動作がより早く

(Cons)
- SD webUI用に改造されたCLIPが使えません (TIや強調などの機能)
- 古い Python < 3.9 をサポートしません


## Requirements
依存関係は全て `pyproject.toml` / `uv.lock` に固定されており、uv が自動的にインストールします。

**自分でインストールするのは [uv](https://docs.astral.sh/uv/) だけです。**
uv が専用の Python インタプリタ (3.11) と全ての依存関係（PyTorch や wd-tagger が使用する [ONNX runtime](https://onnxruntime.ai/) を含む）をこのフォルダ内にダウンロードします。
システム Python は不要で、PyTorch の手動インストールや実行時のパッケージインストールも一切ありません。

## インストール方法
1. uv を一度だけインストールします: <https://docs.astral.sh/uv/getting-started/installation/>
2. アプリを起動します（初回起動時に uv が環境を作成し、固定された依存関係をインストールします）:
   - **Windows:** `launch_uv.bat` をダブルクリック
   - **Linux / macOS:** `./webui_uv.sh`

起動せずにインストールだけ行う場合は `uv sync` を実行してください。`pyproject.toml` / `uv.lock` のピンは Gradio 4.28.3 で動作確認済みの組み合わせです。

デフォルトでは **CPU 版 PyTorch** がインストールされます。NVIDIA GPU を使用する場合は `uv sync` の後に以下を実行してください:
```sh
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run --no-sync scripts/launch.py
```
(`--no-sync` はロックされた CPU 版に戻さず CUDA 版を維持します。)


## 起動方法
`-h` または `--help` をコマンドライン引数に指定することで、利用可能なコマンドライン引数の一覧を取得できます。

### Windows
`launch_uv.bat` を起動してください。

### Linux / macOS
```sh
./webui_uv.sh [arguments]
```

### Google Colab

Google Colabユーザーは、以下のコマンドを実行し、生成されたGradioのPublic URLにアクセスすることで利用できます。  
(おそらく、これは現在Colab Proでのみ利用可能だと思います)。

```sh
%cd /content
!git clone https://github.com/okingjo/dataset-tag-editor-standalone.git
%cd /content/dataset-tag-editor-standalone
!pip install uv
!uv sync
!uv run scripts/launch.py --share
```

## 特徴
以下、「タグ」はカンマ区切りされたキャプションの各部分を意味します。
- テキスト形式（webUI方式）またはjson形式 ([kohya-ss sd-scripts metadata](https://github.com/kohya-ss/sd-scripts))のキャプションを編集できます
- 画像を見ながらキャプションの編集ができます
- タグの検索ができます
- 複数タグで絞り込んでキャプションの編集ができます
  - 絞り込み方法として、AND/OR/NOT検索ができます
- タグを一括で置換・削除・追加できます
- タグを一括で並べ替えできます
- タグまたはキャプション全体について一括置換ができます
  - [正規表現](https://docs.python.org/ja/3/library/re.html#regular-expression-syntax) が利用可能です
- Interrogatorを使用してタグの追加や編集ができます
  - BLIP、DeepDanbooru、[Z3D-E621-Convnext](https://huggingface.co/toynya/Z3D-E621-Convnext)、 [WDv1.4 Tagger](https://huggingface.co/SmilingWolf)の各ネットワークによる学習結果（v1, v2, v3）が使用可能です
- お好みのTaggerを `userscripts/taggers` に追加できます (`scripts.tagger.Tagger`を継承したクラスでラップしてください)
  - 当該フォルダにAesthetic Scoreに基づいたTaggerをいくつか実装しています
- 画像やキャプションファイルの一括移動・削除ができます


## 使い方
1. データセットを作成する
    - 既にリサイズ・トリミングされた画像を使用することをお勧めします
1. データセットを読み込む
    - 必要に応じてDeepDanbooru等でタグ付けができます
1. キャプションを編集する
    - "Filter by Tags" タブでキャプションの編集をしたい画像を絞り込む
    - 画像を手動で選んで絞り込む場合は "Filter by Selection" タブを使用する
    - 一括でタグを置換・削除・追加する場合は "Batch Edit Caption" タブを使用する
    - キャプションを個別に編集したい場合は "Edit Caption of Selected Image" タブを使用する
      - DeepDanbooru等も利用可能
    - 選択したものをデータセットから一括移動・削除したい場合は "Remove or Delete Files" タブを使用する
1. "Save all changes" ボタンをクリックして保存する


## タグ編集の手引き

基本的な手順は以下の通りです

1. 編集対象の画像をフィルターで絞り込む
1. まとめて編集する

一括で行われる編集はすべて**表示されている画像（＝絞り込まれた画像）にのみ**適用されます。

### 1. フィルターの選び方
#### 全ての画像を一括で処理したい場合  
  フィルターは不要です。
#### 何枚かを処理したい場合  
  1. **共通のタグや、共通して持たないタグがある**  
    "Filter by Tags" タブで画像を絞り込み、編集対象だけが表示されるようにする。
  1. **何も共通点が無い**  
    "Filter by Selection" タブで画像を絞り込む。 
    フィルターへの画像の追加は[Enter]キーがショートカットです。
    フィルターを適用する際は、 "Apply selection filter" を押してください。

### 2. 編集の仕方
  ![](pic/ss07.png)  
#### 新しいタグを追加したい場合  
  1. "Batch Edit Captions" タブを開く
  1. "Edit tags" 内に追加したいタグをカンマ区切りで追記する
  1. "Apply changes to filtered images" ボタンを押す  
  ![](pic/ss08.png)  
  例："foo" と "bar" が表示されている画像に追加されます

#### 絞り込まれた画像に共通なタグを編集（置換）したい場合  
  1. "Batch Edit Captions" タブを開く
  1. "Edit tags" 内に表示されたタグを書き換える
  1. "Apply changes to filtered images" ボタンを押す  
  ![](pic/ss09.png)  
  例："male focus" と "solo" がそれぞれ "foo" と "bar" に置換されます

#### タグを取り除きたい場合  
  置換と同様の手順で、対象のタグを空欄に書き換えることで取り除けます。  
  共通のタグでない（一部の画像にのみ含まれる等）場合は、"Batch Edit Captions" タブにある "Remove" を利用することもできます。  
  ![](pic/ss10.png)  

#### 柔軟にタグを追加・削除・置換したい場合  
  1. "Batch Edit Captions" タブを開く
  2. "Use regex" にチェックを入れて "Search and Replace" する  
  ![](pic/ss11.png)  
  例："1boy", "2boys", … がそれぞれ、 "1girl", "2girls", … に置換されます。  
  カンマはタグの区切りとみなされるため、カンマを追加・削除することで新しいタグを追加・削除できます。  
  正規表現（regex）を使うと、複雑な条件に応じてタグの編集が可能です。


## トラブルシューティング
### ギャラリーに画像が表示されず、コンソールにエラーが表示されない
(おそらくv0.0.6以降)  
このプログラムのあるフォルダ以外から画像を読み込む場合は、"Settings" タブで、画像の読み取りを許可するフォルダを指定する、もしくは、下の項目と同様の方法で、サムネイル画像を一時保存するフォルダを指定してください。  
"Path whitelist to show images …" にパスを指定します。  
子フォルダに全て適用されるため、ドライブ名を"C:\\"のように指定すると、Cドライブの全てに対して読み取りを許可します。  
![](pic/ss12.png) 

### ギャラリーに画像が表示されず、コンソールに "All files must contained within the Gradio python app working directory…" と出ている
(おそらくv0.0.5以前)  
"Settings" タブで、サムネイル画像を一時保存するフォルダを指定してください。  
"Directory to save temporary files" にパスを指定して "Force using temporary file…" をチェックしてください。  

### 大量の画像や巨大な画像を開いたときに動作が遅くなる
"Settings" タブで、"Maximum resolution of ..." に希望の解像度を入れて、サムネイルのサイズを制限してください。  
数百万もの画像を含むなど、あまりにも巨大なデータセットでは効果がないかもしれません。  
![](pic/ss13.png) 

### PyTorch が CUDA を使っていない
デフォルトでは CPU 版 PyTorch がインストールされます。専用の `.venv` を CUDA 版に切り替えるには:
```sh
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv run --no-sync scripts/launch.py
```
`--no-sync` はロックされた CPU 版に戻さず CUDA 版を維持するため、起動は `uv run --no-sync scripts/launch.py` を使用してください（`launch_uv.bat` は CPU 版を再同期します）。

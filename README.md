ローカルLLMをテストするときに参考にするコードをまとめたリポジトリ

# llama_cpp_pythonをColabで使うときのセットアップ

## 利用したいモデルを準備

```
!wget ＜利用したいGGUF＞
```

## 必要なライブラリをダウンロード

```
!CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install -U llama_cpp_python
!pip install 'git+https://github.com/huggingface/transformers.git'
!pip install -U -r requirements,txt
```

# ollmaをColabで使うときの準備

## 利用したいモデルを準備

```
!wget ＜利用したいGGUF＞
```

## 必要なソフトウェアをインストール

```
!curl https://ollama.ai/install.sh | sh

!echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
!sudo apt-get update && sudo apt-get install -y cuda-drivers

import os

# Set LD_LIBRARY_PATH so the system NVIDIA library
os.environ.update({'LD_LIBRARY_PATH': '/usr/lib64-nvidia'})

```

## プロンプトテンプレートやストップトークンを指定するModelfileを作成する

```

# ファイルパス
_filepath = './Modelfile'

# ファイルに書き込む内容
filecontents = """
FROM ./<ダウンロードしたモデル>

TEMPLATE \"\"\"　使うモデルのプロンプトテンプレート \"\"\"

PARAMETER stop  "使うモデルのstopトークン"

"""

# 書き込みモード
with open(_filepath, 'w') as f:
  # ファイル作成、書き込み
  f.write(filecontents)
```

## ollamaを実行

```
# 以下のコマンドをterminalで実行
!nohup ollama serve &
!ollama create <model名> -f Modelfile
```

## 必要なPythonライブラリをインストール

```
!pip install ollama
```


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
!pip install -U -r requirements.txt
```

# ollmaをColabで使うときの準備

## 利用したいモデルを準備

```
!wget ＜利用したいGGUF＞
```

## 必要なソフトウェアをインストール

```
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve &
!ollama pull command-r-plus
```

## 必要なPythonライブラリをインストール

```
!pip install ollama
```

# Whisperを使うときの準備

```
!pip install --upgrade pip
!pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
```

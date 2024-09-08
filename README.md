ローカルLLMをテストするときに参考にするコードをまとめたリポジトリ

### llama_cpp_pythonをColabで使うときのセットアップ

# 利用したいモデルを準備

```
!wget ＜利用したいGGUF＞
```

# 必要なライブラリをダウンロード

```
!CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install -U llama_cpp_python
!pip install 'git+https://github.com/huggingface/transformers.git'
!pip install -U -r requirements,txt
```
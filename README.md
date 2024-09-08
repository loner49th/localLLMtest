ローカルLLMをテストするときに参考にするコードをまとめたリポジトリ

### llama_cpp_pythonをColabで使うときのセットアップ

# 利用したいモデルを準備
!wget ＜利用したいGGUF＞

!CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install -U llama_cpp_python

# 必要なライブラリをダウンロード
!pip install 'git+https://github.com/huggingface/transformers.git' bitsandbytes accelerate
!pip install -U langchain-community faiss-cpu langchain_core  trafilatura  langchain-text-splitters tiktoken sentence_transformers

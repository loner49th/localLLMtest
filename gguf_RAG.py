# RAG用のデータの取得

from trafilatura import fetch_url, extract

url = <RAGの下とするURL>
filename = 'textfile.txt'

document = fetch_url(url)
text = extract(document)

with open(filename, 'w', encoding='utf-8') as f:
    f.write(text)

# 取得したデータをチャンク化する
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader(filename, encoding='utf-8')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator = "\n",
    chunk_size=800,
    chunk_overlap=100,
)
texts = text_splitter.split_documents(documents)


# 埋込みモデルの準備

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# ベクターDBにチャンク化したデータを取り込み、Retrieverを準備する
vectorstore = FAISS.from_documents(texts, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 65  # Change this value based on your model and your GPU VRAM pool.
n_batch = 4048

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=<ダウンロードしたGGUF名>,
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_ctx=n_batch,
    stop=["ストップトークン"],
    use_mmap=False,
    top_p=0.95,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# そのモデルのプロンプトテンプレートにそって、テンプレートを作る
prompt = PromptTemplate.from_template(
    """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
以下のcontextだけに基づいて質問に回答してください。

{context}

質問: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
)

model =llm
output_parser = StrOutputParser()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

def _combine_documents(
    docs,document_separator="\n\n"
):
    doc_strings = [doc.page_content for doc in docs]
    return document_separator.join(doc_strings)

from operator import itemgetter

retrieved_documents = RunnablePassthrough.assign(
    docs=itemgetter("question") | retriever,
)
final_inputs  = {"context": lambda x: _combine_documents(x["docs"]), "question": itemgetter("question")}
answer = {
    "answer": prompt | model  | output_parser,
    "docs": itemgetter("context"),
}
final_chain = retrieved_documents| final_inputs | answer

result = final_chain.invoke({"question":"＜質問内容＞"})
    
result
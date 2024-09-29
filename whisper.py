import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

result = pipe("文字起こししたいファイル名", generate_kwargs = { "language":"<|ja|>"} ) #英語の場合 { "language":"<|en|>"}

text =""
for chunk in result['chunks']:
    text += str(chunk["timestamp"])+":"+chunk['text']+"\n"

# ファイル名を指定（同じディレクトリに 'output.txt' が作成されます）
file_name = "output.txt"

# 'w' モードでファイルを開き、テキストを書き込む
with open(file_name, 'w', encoding='utf-8') as file:
    file.write(text)
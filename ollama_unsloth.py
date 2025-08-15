import re, json, hashlib, requests
from pathlib import Path
import fitz


OLLAMA = "http://localhost:11434/api/chat"  


def read_pdf(path):
    """
    PDFファイルからテキストを抽出する関数
    
    Args:
        path (str): PDFファイルのパス
    
    Returns:
        str: 抽出されたテキスト（全ページの内容を結合）
    
    Raises:
        fitz.FileNotFoundError: ファイルが見つからない場合
        fitz.FileDataError: ファイルが破損している場合
    """
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
def chunk_text(text, max_chars=1200, overlap=200):
    """
    テキストを指定された文字数でチャンクに分割する関数
    
    Args:
        text (str): 分割対象のテキスト
        max_chars (int, optional): 1チャンクの最大文字数. Defaults to 1200.
        overlap (int, optional): チャンク間のオーバーラップ文字数. Defaults to 200.
    
    Returns:
        list[str]: 分割されたテキストチャンクのリスト（200文字未満は除外）
    
    Note:
        - 句読点（。！？\?！）と改行で文を分割
        - オーバーラップを使用して文脈を保持
        - 200文字未満の短いチャンクは除外
    """
    # 文っぽい区切りで素朴に分割してから結合
    sents = re.split(r"(?<=[。！？\?！])\s*|\n", text)
    buf, cur = [], ""
    for s in sents:
        if not s: continue
        if len(cur)+len(s) <= max_chars:
            cur += (s if cur == "" else s)
        else:
            if cur: buf.append(cur)
            # オーバーラップ
            cur = cur[-overlap:] + s if overlap>0 else s
            cur = cur[-max_chars:]
    if cur: buf.append(cur)
    return [b.strip() for b in buf if len(b.strip())>200]  # 短すぎる塊は捨てる

def ask_ollama(context, model, n=5, temperature=0.2):
    """
    Ollamaを使用してコンテキストからQAペアを生成する関数
    
    Args:
        context (str): QA生成の元となるテキストコンテキスト
        model (str): 使用するOllamaモデル名
        n (int, optional): 生成するQAペア数（現在は未使用）. Defaults to 5.
        temperature (float, optional): 生成温度（0.0-1.0）. Defaults to 0.2.
    
    Returns:
        list[dict]: 生成されたQAペアのリスト
                   各辞書は {"question": str, "answer": str, "answerable": str, "quality": str} の形式
    
    Raises:
        requests.exceptions.RequestException: Ollama APIへのリクエストが失敗した場合
        json.JSONDecodeError: レスポンスのJSON解析に失敗した場合
        
    Note:
        - コンテキストは3000文字に切り詰められる
        - JSON形式で1つのQAペアを生成
        - answerable=falseの場合は回答不能と判定
        - quality=1-5で品質を自己評価
    """
    schema = {
        "type":"object",
        "properties":{
        "question":{"type":"string"},
        "answer":{"type":"string"},
        "answerable":{"type":"string"},
        "quality":{"type":"string"}
        },
        "required":["question","answer","answerable","quality"]
    }
    sys = (
      "あなたは日本語QAデータ生成器。"
      "与えられた抜粋『のみ』に基づき、思考過程は出力しない。"
      "出力はJSONのみ。前置き禁止。"
    )
    user = (
      "次の資料抜粋から、質問と回答を日本語で1つ作成せよ。\n"
      "- 各要素は question, answer, answerable(bool), quality(1-5)。"
      "- 回答は抜粋内の事実のみ。言い換え可。外部知識禁止。\n"
      "- 回答不能な内容は作らない。もし作ってしまったと思えば answerable=false。\n"
      "- 自己採点 quality=1-5。\n"
      "資料抜粋:\n<<<\n{ctx}\n>>>\n"
    ).format(ctx=context[:3000])
    r = requests.post(OLLAMA, json={
        "model": model,
        "messages":[{"role":"system","content":sys},{"role":"user","content":user}],
        "stream": False,
        "think": False,
        "options":{"temperature":temperature}
    }, timeout=180)
    r.raise_for_status()
    content = (r.json().get("message") or {}).get("content","")
    if content.strip():
      print(content)
      return [json.loads(content)]
    else:
      return []

def bigram_ratio(ans, ctx):
    """
    回答とコンテキスト間のバイグラム類似度を計算する関数
    
    Args:
        ans (str): 回答テキスト
        ctx (str): コンテキストテキスト
    
    Returns:
        float: バイグラム類似度（0.0-1.0）
               0.0: 類似度なし, 1.0: 完全一致
    
    Note:
        - 空白を除去してバイグラム（2文字組み合わせ）を作成
        - 回答のバイグラムのうち、コンテキストに含まれる割合を計算
        - 回答がコンテキストに基づいているかの判定に使用
    """
    def grams(s):
        s = re.sub(r"\s+", "", s)
        return {s[i:i+2] for i in range(max(0,len(s)-1))}
    a, c = grams(ans), grams(ctx)
    return 0 if not a else len(a & c) / max(1,len(a))

def hash_pair(q, a):
    """
    質問と回答のペアから一意のハッシュ値を生成する関数
    
    Args:
        q (str): 質問テキスト
        a (str): 回答テキスト
    
    Returns:
        str: SHA256ハッシュ値（16進数文字列）
    
    Note:
        - 重複するQAペアの検出に使用
        - 質問と回答を改行で結合してハッシュ化
    """
    return hashlib.sha256((q+"\n"+a).encode("utf-8")).hexdigest()

def build_messages_from_pdf(pdf_path, out_jsonl, ollama_model, per_chunk=5):
    """
    PDFファイルからファインチューニング用のJSONLデータを生成する関数
    
    Args:
        pdf_path (str): 入力PDFファイルのパス
        out_jsonl (str): 出力JSONLファイルのパス
        ollama_model (str): QA生成に使用するOllamaモデル名
        per_chunk (int, optional): チャンクあたりの試行回数（現在は未使用）. Defaults to 5.
    
    Returns:
        int: 生成されたQAペアの総数
    
    Raises:
        RuntimeError: PDFからテキストが取得できない場合
        
    Note:
        - PDFをチャンクに分割し、各チャンクから5回QA生成を試行
        - 品質フィルタリング: answerable=True, quality>=4, 最低文字数制約
        - バイグラム類似度>=0.25でコンテキストとの関連性を確保
        - 重複除去機能付き
        - 出力はJSONL形式（各行が{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}）
    """
    text = read_pdf(pdf_path)
    if not text:
        raise RuntimeError("PDFからテキストが取得できない。スキャンPDFはOCRしてから。")
    chunks = chunk_text(text)
    seen, kept = set(), []
    for ch in chunks:
        for _ in range(5):  # 5回繰り返す
            try:
                items = ask_ollama(ch, ollama_model, n=per_chunk)
            except Exception:
                print("Error")
                continue
            for it in items:
                q, a = it.get("question","").strip(), it.get("answer","").strip()
                print(q,a)
                ok = (
                    it.get("answerable", True) and
                    it.get("quality", 0) >= 4 and
                    len(q) >= 8 and len(a) >= 16 and
                    bigram_ratio(a, ch) >= 0.25
                )
                if not ok: continue
                h = hash_pair(q, a)
                if h in seen: continue
                seen.add(h)
                kept.append({
                    "messages":[
                        {"role":"user","content":q},
                        {"role":"assistant","content":a}
                    ]
                })
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False)+"\n")
    return len(kept)

def main():
    """
    メイン実行関数
    """
    # ========== 設定項目 ==========
    # Ollamaモデル設定
    OLLAMA_MODEL = "<データ作成に使うollamaのMODEL_NAME>"
    
    # ファイルパス設定
    PDF_PATH = "<元データとのなるPDFファイルのパス>"
    JSONL_PATH = "<学習用JSONLファイルパス>"
    
    # Unslothモデル設定
    UNSLOTH_MODEL = "unsloth/gemma-3-270m-it"
    MAX_SEQ_LENGTH = 2048
    
    # 学習設定
    BATCH_SIZE = 8
    MAX_STEPS = 100
    LEARNING_RATE = 5e-5
    OUTPUT_DIR = "outputs"
    
    # モデル保存設定
    SAVE_MODEL_PATH = "fine_tuned_model"
    
    # テストメッセージ
    TEST_MESSAGE = "テスト用のメッセージ"
    
    # 実行モード選択
    TRAIN_MODE = True  # True: ファインチューニング実行, False: 保存済みモデルで推論のみ
    # ============================
    
    if TRAIN_MODE:
        # PDFからJSONLデータを生成
        count = build_messages_from_pdf(PDF_PATH, JSONL_PATH, OLLAMA_MODEL)
        print(f"生成されたQAペア数: {count}")
        
        # Unslothによるファインチューニング処理
        run_fine_tuning(JSONL_PATH, UNSLOTH_MODEL, MAX_SEQ_LENGTH, BATCH_SIZE, MAX_STEPS, LEARNING_RATE, OUTPUT_DIR, SAVE_MODEL_PATH, TEST_MESSAGE)
    else:
        # 保存済みモデルで推論のみ実行
        run_inference_only(SAVE_MODEL_PATH, TEST_MESSAGE)

def run_fine_tuning(jsonl_path, model_name, max_seq_length, batch_size, max_steps, learning_rate, output_dir, save_model_path, test_message):
    """
    Unslothを使用したファインチューニング処理
    
    Args:
        jsonl_path: 学習用JSONLファイルのパス
        model_name: Unslothモデル名
        max_seq_length: 最大シーケンス長
        batch_size: バッチサイズ
        max_steps: 最大ステップ数
        learning_rate: 学習率
        output_dir: 出力ディレクトリ
        save_model_path: ファインチューニング後のモデル保存パス
        test_message: テスト用メッセージ
    """
    from unsloth import FastModel
    import torch
    
    fourbit_models = [
        # 4bit dynamic quants for superior accuracy and low memory use
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

        # Other popular models!
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/Llama-3.3-70B",
        "unsloth/mistral-7b-instruct-v0.3",
        "unsloth/Phi-4",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )

    model = FastModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 128,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    from datasets import load_dataset
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    def formatting_prompts_func(examples):
       convos = examples["messages"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
       return { "text" : texts, }

    dataset = ds.map(formatting_prompts_func, batched = True)

    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = 1, # Use GA to mimic batch size!
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = max_steps,
            learning_rate = learning_rate, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none", # Use this for WandB etc
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    trainer_stats = trainer.train()
    
    # モデルを保存
    print(f"モデルを {save_model_path} に保存中...")
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print("モデル保存完了")
    
    # ファインチューニング完了後のテスト
    test_generation(model, tokenizer, test_message)

def test_generation(model, tokenizer, test_message="テスト用のメッセージ"):
    """
    ファインチューニング後のテスト生成
    
    Args:
        model: ファインチューニング済みモデル
        tokenizer: トークナイザー
        test_message: テスト用のメッセージ
    """
    messages = [
        {"role" : 'user', 'content' : test_message}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    ).removeprefix('<bos>')

    from transformers import TextStreamer
    _ = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 125,
        temperature = 1, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

def run_inference_only(model_path, test_message):
    """
    保存済みファインチューニングモデルで推論のみを実行する関数
    
    Args:
        model_path (str): 保存されたモデルのパス
        test_message (str): テスト用メッセージ
    
    Note:
        - ファインチューニング済みのモデルとトークナイザーを読み込み
        - 推論のみを実行（学習処理はスキップ）
        - CUDAが利用可能な場合は自動でGPUを使用
    """
    from unsloth import FastModel
    from transformers import AutoTokenizer
    import torch
    
    print(f"保存済みモデルを {model_path} から読み込み中...")
    
    # 保存済みモデルとトークナイザーを読み込み
    # FastModel.from_pretrainedはタプル(model, tokenizer)を返す
    model, tokenizer = FastModel.from_pretrained(model_path)
    
    print("モデル読み込み完了。推論を開始します...")
    
    # テスト生成を実行
    test_generation(model, tokenizer, test_message)

if __name__ == "__main__":
    main()
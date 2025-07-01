import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse
import re
import json
from pathlib import Path
from datetime import datetime

def get_available_device():
    """利用可能なデバイスを自動検出"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_system_prompt(prompt_file=None):
    """システムプロンプトをファイルまたはデフォルトから読み込み"""
    # カスタムプロンプトが指定されている場合
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    # デフォルトプロンプトファイルを試行
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_prompt_path = project_root / "prompts" / "default.txt"
    
    if default_prompt_path.exists():
        with open(default_prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    # フォールバック用のハードコードされたプロンプト
    return "あなたは親切で知識豊富なAIアシスタントです。ユーザーの質問に丁寧に答えてください。"

def setup_eos_tokens(tokenizer):
    """EOS トークンの設定を最適化"""
    eos_token_ids = []
    
    # 基本のEOSトークン
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)
    
    # 一般的なEOSトークンを確認
    common_eos_tokens = ["<end_of_turn>", "<|im_end|>", "</s>", "<|endoftext|>"]
    for eos_str in common_eos_tokens:
        eos_id = tokenizer.convert_tokens_to_ids(eos_str)
        if eos_id is not None and eos_id != tokenizer.unk_token_id:
            if eos_id not in eos_token_ids:
                eos_token_ids.append(eos_id)
    
    return eos_token_ids

def clean_response(response, tokenizer):
    """生成されたレスポンスをクリーンアップ"""
    # 一般的なEOSトークンで分割
    eos_tokens = ["<end_of_turn>", "<|im_end|>", "</s>", "<|endoftext|>"]
    if tokenizer.eos_token:
        eos_tokens.append(tokenizer.eos_token)
    
    for eos_token in eos_tokens:
        if eos_token in response:
            response = response.split(eos_token)[0].strip()
            break
    
    # Markdown形式の太字をANSIエスケープコードに変換
    response = re.sub(r"\*\*(.*?)\*\*", r"\033[1m\1\033[0m", response)
    
    return response

def save_conversation_history(history, output_file=None):
    """会話履歴をJSONファイルに保存"""
    # スクリプトの場所を基準にhistoryディレクトリを作成
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    history_dir = project_root / "history"
    
    # historyディレクトリを作成
    history_dir.mkdir(exist_ok=True)
    
    # 出力ファイル名の決定
    if output_file:
        output_path = Path(output_file)
        # 相対パスの場合はhistoryディレクトリを基準にする
        if not output_path.is_absolute():
            output_path = history_dir / output_path
    else:
        # デフォルトファイル名（タイムスタンプ付き）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = history_dir / f"chat_{timestamp}.json"
    
    # 出力ファイルのディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "conversation": history,
        "total_messages": len([msg for msg in history if msg["role"] != "system"]),
        "model_info": {
            "saved_at": str(output_path),
            "session_duration": None  # 後で計算可能
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)

def get_config():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(
        description="Chat with LoRA fine-tuned language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python chat_with_lora.py --model google/gemma-3-1b-it --adapter models/trained_model/final_model
  
  # With custom system prompt
  python chat_with_lora.py --model google/gemma-3-1b-it --adapter ./my_adapter --system-prompt prompts/coding_assistant.txt
  
  # Save conversation history (auto-generated filename)
  python chat_with_lora.py --model google/gemma-3-1b-it --adapter ./adapter --save-history
  
  # Save conversation history with custom filename
  python chat_with_lora.py --model google/gemma-3-1b-it --adapter ./adapter --save-history my_chat.json
        """
    )
    
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        "--model", 
        required=True,
        help="Base model name from HuggingFace Hub or local path (e.g., 'google/gemma-3-1b-it')"
    )
    required_group.add_argument(
        "--adapter", 
        required=True,
        help="Path to LoRA adapter directory"
    )
    
    # Generation settings
    generation_group = parser.add_argument_group('Generation Settings')
    generation_group.add_argument(
        "--max-tokens", 
        type=int, 
        default=512,
        help="Maximum number of tokens to generate (default: %(default)s)"
    )
    generation_group.add_argument(
        "--temperature", 
        type=float, 
        default=0.2,
        help="Sampling temperature (default: %(default)s)"
    )
    generation_group.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p sampling parameter (default: %(default)s)"
    )
    generation_group.add_argument(
        "--top-k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter (default: %(default)s)"
    )
    
    # Optional settings
    optional_group = parser.add_argument_group('Optional Settings')
    optional_group.add_argument(
        "--device", 
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for inference (default: %(default)s)"
    )
    optional_group.add_argument(
        "--system-prompt",
        help="Path to custom system prompt file (default: prompts/default.txt)"
    )
    optional_group.add_argument(
        "--save-history",
        nargs="?",
        const="",  # 引数なしの場合は空文字
        help="Enable history saving. Optionally specify filename (saved in history/ directory). If no filename given, auto-generates one."
    )
    optional_group.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model data type (default: %(default)s)"
    )
    
    return parser.parse_args()

def main():
    args = get_config()
    
    # デバイス設定
    if args.device == "auto":
        device = get_available_device()
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # LoRAアダプターの検証
    if not os.path.exists(args.adapter):
        print(f"Error: Adapter path '{args.adapter}' does not exist")
        return
    
    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        print(f"Error: '{args.adapter}' does not appear to be a valid LoRA adapter")
        return
    
    # データ型設定
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device in ["cuda", "mps"] else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    
    print(f"Loading base model: {args.model}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("Base model loaded successfully")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return
    
    print(f"Loading tokenizer")
    try:
        # まずアダプターからトークナイザーを試行
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
        except:
            # 失敗したらベースモデルから取得
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # パディングトークンの設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    print(f"Loading LoRA adapter: {args.adapter}")
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model = model.to(device)
        model.eval()
        print("LoRA adapter loaded successfully")
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        return
    
    # EOS トークンの設定
    eos_token_ids = setup_eos_tokens(tokenizer)
    print(f"EOS token IDs: {eos_token_ids}")
    
    # システムプロンプトの読み込み
    system_prompt = load_system_prompt(args.system_prompt)
    conversation_history = [{"role": "system", "content": system_prompt}]
    
    print(f"\nChat Session Started")
    print(f"Commands: 'exit', 'quit', 'clear', 'history', 'save'")
    print("=" * 60)
    
    try:
        while True:
            try:
                user_input = input("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
            
            # 特別なコマンドの処理
            if user_input.lower() in ["exit", "quit", "終了"]:
                print("Chat session ended!")
                break
            elif user_input.lower() == "clear":
                conversation_history = [{"role": "system", "content": system_prompt}]
                print("Conversation history cleared!")
                continue
            elif user_input.lower() == "history":
                print("\nConversation History:")
                for i, msg in enumerate(conversation_history[1:], 1):
                    role_prefix = "User" if msg["role"] == "user" else "Assistant"
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    print(f"{i}. {role_prefix}: {content}")
                continue
            elif user_input.lower() == "save":
                saved_path = save_conversation_history(conversation_history, args.save_history)
                print(f"Conversation saved to: {saved_path}")
                continue
            elif not user_input.strip():
                continue
            
            # ユーザー入力を履歴に追加
            conversation_history.append({"role": "user", "content": user_input})
            
            # チャットテンプレートの適用
            try:
                prompt = tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                print(f"Error applying chat template: {e}")
                conversation_history.pop()
                continue
            
            # トークン化
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print("Assistant: ", end="", flush=True)
            
            # テキスト生成
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_token_ids if eos_token_ids else None,
                        repetition_penalty=1.1
                    )
                
                # レスポンスのデコードとクリーンアップ
                response_ids = outputs[0][inputs.input_ids.shape[-1]:]
                raw_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                assistant_response = clean_response(raw_response, tokenizer)
                
                print(assistant_response)
                
                # アシスタントの回答を履歴に追加
                conversation_history.append({"role": "assistant", "content": assistant_response})
                
            except Exception as e:
                print(f"Error during generation: {e}")
                conversation_history.pop()  # エラー時はユーザー入力を削除
    
    finally:
        # 終了時に履歴を保存（指定されている場合、または会話がある場合は自動保存）
        if args.save_history or len(conversation_history) > 1:  # システムプロンプト以外にメッセージがある場合
            saved_path = save_conversation_history(conversation_history, args.save_history)
            print(f"Final conversation saved to: {saved_path}")

if __name__ == "__main__":
    main()
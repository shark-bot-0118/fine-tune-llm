import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os
from dotenv import load_dotenv
import json
import argparse
from pathlib import Path

load_dotenv(override=True)

def get_config():
    """設定の取得（コマンドライン引数または環境変数）"""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with HuggingFace model
  python train_lora.py --model google/gemma-3-1b-it
  
  # Custom settings
  python train_lora.py --model google/gemma-3-1b-it --epochs 5 --batch-size 4 --learning-rate 1e-4
  
  # Custom data and output paths
  python train_lora.py --model microsoft/DialoGPT-medium --data custom_data.jsonl --output ./my_model
        """
    )
    
    # Model settings
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model", 
        type=str, 
        help="Base model name from HuggingFace Hub (e.g., 'google/gemma-3-1b-it', 'microsoft/DialoGPT-medium') or local path"
    )
    
    # Data settings
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        "--data", 
        type=str, 
        help="Path to training data file in JSONL format (default: data/fine_tune_data.jsonl)"
    )
    data_group.add_argument(
        "--max-length", 
        type=int, 
        default=512, 
        help="Maximum sequence length for tokenization (default: %(default)s)"
    )
    
    # Training settings
    training_group = parser.add_argument_group('Training Configuration')
    training_group.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs (default: %(default)s)"
    )
    training_group.add_argument(
        "--batch-size", 
        type=int, 
        default=2, 
        help="Training batch size per device (default: %(default)s)"
    )
    training_group.add_argument(
        "--gradient-accumulation-steps", 
        type=int, 
        default=8, 
        help="Number of gradient accumulation steps (default: %(default)s)"
    )
    training_group.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5, 
        help="Learning rate for training (default: %(default)s)"
    )
    
    # LoRA settings
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        "--lora-r", 
        type=int, 
        default=16, 
        help="LoRA rank parameter (default: %(default)s)"
    )
    lora_group.add_argument(
        "--lora-alpha", 
        type=int, 
        default=32, 
        help="LoRA alpha parameter (default: %(default)s)"
    )
    lora_group.add_argument(
        "--lora-dropout", 
        type=float, 
        default=0.05, 
        help="LoRA dropout rate (default: %(default)s)"
    )
    
    # Output settings
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output", 
        type=str, 
        help="Output directory for trained model (default: ../models/{PREFIX} from env var or ../models/trained_model)"
    )
    output_group.add_argument(
        "--save-steps", 
        type=int, 
        default=50, 
        help="Save checkpoint every N steps (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # 環境変数からのフォールバック
    model_name = args.model or os.getenv("BASE_MODEL_NAME")
    prefix = os.getenv("PREFIX", "trained_model")
    
    # スクリプトの場所を基準に絶対パスを構築
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_output = project_root / "models" / prefix
    
    output_dir = args.output or str(default_output)
    
    if not model_name:
        raise ValueError("Model name must be specified via --model argument or BASE_MODEL_NAME environment variable")
    
    # データファイルのパス処理
    if args.data:
        data_file = args.data
    else:
        data_file = str(project_root / "data" / "fine_tune_data.jsonl")
    
    return {
        "model_name": model_name,
        "data_file": data_file,
        "output_dir": output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "save_steps": args.save_steps
    }

def validate_dataset(dataset_file):
    """データセットファイルの事前検証"""
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file {dataset_file} not found")
    
    line_count = 0
    error_count = 0
    
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            try:
                data = json.loads(line.strip())
                if "messages" not in data:
                    print(f"Warning: Line {line_num} missing 'messages' field")
                    error_count += 1
                elif not isinstance(data["messages"], list):
                    print(f"Warning: Line {line_num} 'messages' is not a list")
                    error_count += 1
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} has invalid JSON: {e}")
                error_count += 1
    
    print(f"Dataset validation: {line_count} lines total, {error_count} errors found")
    if error_count > line_count * 0.1:  # 10%以上のエラー率
        raise ValueError(f"Too many errors in dataset: {error_count}/{line_count}")
    
    return line_count, error_count

def main():
    try:
        config = get_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    model_name = config["model_name"]
    dataset_file = config["data_file"]
    output_dir_base = config["output_dir"]
    final_model_path = os.path.join(output_dir_base, "final_model")
    
    # 出力ディレクトリの作成
    Path(output_dir_base).mkdir(parents=True, exist_ok=True)
    Path(final_model_path).mkdir(parents=True, exist_ok=True)
    
    # デバイス設定の改善
    if torch.backends.mps.is_available():
        device_map = {"": "mps"}
        print("Using Apple MPS device")
    elif torch.cuda.is_available():
        device_map = {"": "cuda"}
        print("Using CUDA device")
    else:
        device_map = {"": "cpu"}
        print("Using CPU device")

    # データセット事前検証
    try:
        line_count, error_count = validate_dataset(dataset_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Dataset validation failed: {e}")
        return

    print(f"Loading base model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation='eager'  # Gemmaモデル推奨設定
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # トークナイザーの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # トークン情報の確認
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"Model config - EOS ID: {model.config.eos_token_id}, PAD ID: {model.config.pad_token_id}")

    print("Configuring LoRA for tone/style fine-tuning...")
    # LoRA設定
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=[
            # Attention層（必須）
            "q_proj", "v_proj", "k_proj", "o_proj",
            # FFN層（口調学習に重要）
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    try:
        model = get_peft_model(model, lora_config)
        print("Trainable parameters after LoRA configuration:")
        model.print_trainable_parameters()
        
        # MPS環境での勾配計算問題対策
        model.train()  # 明示的にトレーニングモードに設定
        
        # LoRAパラメータの勾配設定を確認・修正
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 勾配が必要なパラメータのデバイス確認
                if param.device.type != 'mps' and torch.backends.mps.is_available():
                    print(f"Warning: Parameter {name} not on MPS device")
        
        print("Model prepared for training")
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        return

    print(f"Loading dataset from: {dataset_file}")
    try:
        raw_dataset = load_dataset("json", data_files=dataset_file, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    def format_and_tokenize(example):
        """データフォーマット関数の改善"""
        try:
            if "messages" not in example:
                raise KeyError("Missing 'messages' field")
            
            messages = example["messages"]
            if not isinstance(messages, list):
                raise TypeError("'messages' field must be a list")
            
            # apply_chat_templateを使用
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # EOSトークンの確認と追加
            if not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            
            # トークン化
            tokenized = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=config["max_length"],
                return_tensors="pt"
            )
            
            # ラベル設定
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            # テンソルの次元を調整
            return {k: v[0] for k, v in tokenized.items()}
            
        except Exception as e:
            print(f"Formatting error for example: {e}")
            print(f"Example content: {example}")
            # エラーの場合は None を返して後で除外
            return None

    print("Tokenizing dataset...")
    # データセットの処理を改善
    tokenized_dataset = raw_dataset.map(
        format_and_tokenize, 
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset"
    )
    # None（エラー）のエントリを除外
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: x is not None and len(x.get("input_ids", [])) > 0,
        desc="Filtering invalid examples"
    )
    
    if len(tokenized_dataset) == 0:
        print("Error: No valid examples after tokenization")
        return
    
    print(f"Successfully tokenized {len(tokenized_dataset)} examples")

    # 学習設定
    training_args = TrainingArguments(
        output_dir=output_dir_base,
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["epochs"],
        logging_steps=10,
        save_steps=config["save_steps"],
        save_total_limit=3,
        # データ型設定 - 初期化時はFalseにしてバグ回避
        fp16=False,
        bf16=False,  # 初期化後に設定
        gradient_checkpointing=False,  # MPS環境では無効化
        report_to="none",
        warmup_ratio=0.3,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        eval_strategy="no",
        dataloader_drop_last=True,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=False,
    )

    # Apple MPS/CUDA環境でのbf16設定（初期化後に設定してバグ回避）
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        training_args.bf16 = True
        print("bf16 enabled for MPS/CUDA device")
    else:
        print("Using fp32 for CPU device")

    # Trainerの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    try:
        trainer.train()
        print("Training finished successfully.")
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # モデル保存
    print(f"Saving final LoRA adapter to: {final_model_path}")
    try:
        os.makedirs(final_model_path, exist_ok=True)
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        return

    # 訓練情報の保存
    training_info = {
        "model_name": model_name,
        "dataset_file": dataset_file,
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "max_length": config["max_length"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "save_steps": config["save_steps"],
        "total_examples": len(tokenized_dataset)
    }
    
    with open(os.path.join(final_model_path, "training_info.json"), "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print("Fine-tuning completed successfully!")
    print(f"Model saved to: {final_model_path}")
    print(f"\nTo load the fine-tuned model:")
    print(f"  from peft import PeftModel")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  \n  base_model = AutoModelForCausalLM.from_pretrained('{model_name}', torch_dtype=torch.bfloat16, device_map='auto')")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_model_path}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{final_model_path}')")

if __name__ == "__main__":
    main()
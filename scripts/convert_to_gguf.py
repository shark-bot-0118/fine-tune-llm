#!/usr/bin/env python3
"""
GGUF Conversion Script
Converts models to GGUF format using llama.cpp conversion tools
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def get_llama_cpp_path():
    """llama.cppのパスを取得"""
    # スクリプトの親ディレクトリからllama.cppを探す
    script_dir = Path(__file__).parent
    llama_cpp_path = script_dir.parent / "llama.cpp"
    
    if not llama_cpp_path.exists():
        raise FileNotFoundError(f"llama.cpp not found at {llama_cpp_path}")
    
    return llama_cpp_path

def check_dependencies():
    """必要な依存関係をチェック"""
    try:
        import torch
        import transformers
        print(f"PyTorch: {torch.__version__}")
        print(f"Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"Error: Required dependency not found: {e}")
        print("Please install: pip install torch transformers")
        sys.exit(1)

def ensure_tokenizer_model(model_path):
    """tokenizer.modelファイルが存在することを確認し、なければ取得を試行"""
    tokenizer_model_path = Path(model_path) / "tokenizer.model"
    
    if tokenizer_model_path.exists():
        print(f"tokenizer.model found at: {tokenizer_model_path}")
        return True
    
    print(f"tokenizer.model not found at: {tokenizer_model_path}")
    print("Attempting to obtain tokenizer.model...")
    
    # HuggingFaceモデルの場合、元のモデルから取得を試行
    try:
        from transformers.utils import cached_file
        from transformers import AutoTokenizer
        
        # モデルがローカルパスかHuggingFace Hub IDかを判定
        if Path(model_path).is_dir():
            # ローカルディレクトリ内のconfig.jsonから元のモデル名を取得
            config_path = Path(model_path) / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # _name_or_pathから元のモデル名を取得
                original_model = config.get('_name_or_path', model_path)
            else:
                original_model = model_path
        else:
            original_model = model_path
        
        # 元のモデルからtokenizer.modelを取得
        print(f"Trying to get tokenizer.model from: {original_model}")
        original_tokenizer_model = cached_file(original_model, "tokenizer.model")
        
        if original_tokenizer_model and Path(original_tokenizer_model).exists():
            import shutil
            shutil.copy2(original_tokenizer_model, tokenizer_model_path)
            print(f"Successfully copied tokenizer.model to: {tokenizer_model_path}")
            return True
        else:
            print("tokenizer.model not available from original model")
            
    except Exception as e:
        print(f"Error obtaining tokenizer.model: {e}")
    
    # 最後の手段として、transformersのTokenizerから生成を試行
    try:
        print("Attempting to generate tokenizer.model using transformers...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tokenizer, 'save_vocabulary'):
            vocab_files = tokenizer.save_vocabulary(str(Path(model_path)))
            print(f"Generated vocabulary files: {vocab_files}")
            
            # tokenizer.modelが生成されたか確認
            if tokenizer_model_path.exists():
                print(f"Successfully generated tokenizer.model")
                return True
        
    except Exception as e:
        print(f"Error generating tokenizer.model: {e}")
    
    print("Warning: Could not obtain tokenizer.model. Conversion may fail.")
    return False

def convert_hf_model(model_path, output_dir, dtype="f16"):
    """HuggingFaceモデルをGGUFに変換"""
    llama_cpp_path = get_llama_cpp_path()
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")
    
    # tokenizer.modelの存在を確認・取得
    print("Checking for tokenizer.model file...")
    ensure_tokenizer_model(model_path)
    
    # 出力ディレクトリを作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 変換コマンドを実行
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile", str(output_dir),
        "--outtype", dtype
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error during conversion:")
        print(result.stderr)
        return False
    
    print("Conversion completed successfully")
    print(result.stdout)
    return True

def convert_lora_adapter(adapter_path, base_model, output_path):
    """LoRAアダプターをGGUFに変換"""
    llama_cpp_path = get_llama_cpp_path()
    convert_script = llama_cpp_path / "convert_lora_to_gguf.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"LoRA conversion script not found: {convert_script}")
    
    # 出力ディレクトリを作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 変換コマンドを実行（正しい引数形式）
    cmd = [
        sys.executable,
        str(convert_script),
        str(adapter_path),
        "--outfile", str(output_path),
        "--base-model-id", str(base_model)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error during LoRA conversion:")
        print(result.stderr)
        return False
    
    print("LoRA conversion completed successfully")
    print(result.stdout)
    return True

def quantize_model(input_path, output_path, quant_type="Q4_K_M"):
    """モデルを量子化"""
    llama_cpp_path = get_llama_cpp_path()
    quantize_tool = llama_cpp_path / "tools" / "quantize" / "quantize"
    
    # WindowsとLinux/macOSで実行ファイル名が異なる場合がある
    if not quantize_tool.exists():
        quantize_tool = llama_cpp_path / "build" / "bin" / "llama-quantize"
    
    if not quantize_tool.exists():
        print("Quantization tool not found. Please build llama.cpp first.")
        print("Run: cd llama.cpp && make")
        return False
    
    # 出力ディレクトリを作成
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 量子化コマンドを実行
    cmd = [str(quantize_tool), str(input_path), str(output_path), quant_type]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error during quantization:")
        print(result.stderr)
        return False
    
    print("Quantization completed successfully")
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert models to GGUF format using llama.cpp tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert HuggingFace model to GGUF
  python convert_to_gguf.py --model google/gemma-3-1b-it --output ../gguf/gemma-3-1b-it

  # Convert merged model to GGUF with quantization
  python convert_to_gguf.py --model ../output/merged_model --output ../gguf/gemma-3-1b-it --quantize Q4_K_M

  # Convert LoRA adapter to GGUF
  python convert_to_gguf.py --lora ../models/trained_model/final_model --base-model google/gemma-3-1b-it --output ../gguf/lora_adapter.gguf

  # Just quantize existing GGUF
  python convert_to_gguf.py --quantize-only ../gguf/model.gguf ../gguf/model_q4.gguf --quant-type Q4_K_M
        """
    )
    
    # Main operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--model",
        help="Path to HuggingFace model (local or hub name) to convert to GGUF"
    )
    mode_group.add_argument(
        "--lora",
        help="Path to LoRA adapter to convert to GGUF"
    )
    mode_group.add_argument(
        "--quantize-only",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Only quantize existing GGUF file (input_path output_path)"
    )
    
    # Model conversion options
    convert_group = parser.add_argument_group('Model Conversion Options')
    convert_group.add_argument(
        "--output",
        required=False,
        help="Output directory for converted model (required for --model and --lora)"
    )
    convert_group.add_argument(
        "--dtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        help="Data type for conversion (default: %(default)s)"
    )
    
    # LoRA specific options
    lora_group = parser.add_argument_group('LoRA Conversion Options')
    lora_group.add_argument(
        "--base-model",
        help="Base model for LoRA conversion (required when using --lora)"
    )
    
    # Quantization options
    quant_group = parser.add_argument_group('Quantization Options')
    quant_group.add_argument(
        "--quantize",
        choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Quantization type to apply after conversion"
    )
    quant_group.add_argument(
        "--quant-type",
        default="Q4_K_M",
        choices=["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
        help="Quantization type for --quantize-only (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model and not args.output:
        parser.error("--output is required when using --model")
    
    if args.lora:
        if not args.base_model:
            parser.error("--base-model is required when using --lora")
        if not args.output:
            parser.error("--output is required when using --lora")
    
    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    try:
        success = True
        
        if args.quantize_only:
            # 量子化のみ
            input_path, output_path = args.quantize_only
            success = quantize_model(input_path, output_path, args.quant_type)
            
        elif args.model:
            # HuggingFaceモデルの変換
            print(f"Converting HuggingFace model: {args.model}")
            success = convert_hf_model(args.model, args.output, args.dtype)
            
            # 量子化が指定されている場合
            if success and args.quantize:
                print(f"Applying quantization: {args.quantize}")
                # 変換されたGGUFファイルを探す
                gguf_files = list(Path(args.output).glob("*.gguf"))
                if gguf_files:
                    input_gguf = gguf_files[0]  # 最初のGGUFファイルを使用
                    output_gguf = input_gguf.parent / f"{input_gguf.stem}_{args.quantize.lower()}.gguf"
                    success = quantize_model(input_gguf, output_gguf, args.quantize)
                else:
                    print("Warning: No GGUF files found for quantization")
                    success = False
                    
        elif args.lora:
            # LoRAアダプターの変換
            print(f"Converting LoRA adapter: {args.lora}")
            success = convert_lora_adapter(args.lora, args.base_model, args.output)
        
        if success:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

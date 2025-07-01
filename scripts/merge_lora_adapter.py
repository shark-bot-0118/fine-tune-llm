import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import json
from datetime import datetime

def is_local_model(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter weights with base model to create a standalone model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python merge_lora_adapter.py --model google/gemma-3-1b-it --lora models/trained_model/final_model
  
  # Custom output path and dtype
  python merge_lora_adapter.py --model google/gemma-3-1b-it --lora ./my_adapter --output ./merged_gemma --dtype bfloat16
  
  # With local base model
  python merge_lora_adapter.py --model ./local_model --lora ./adapter --output ./final_model
        """
    )
    
    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        "--model", 
        required=True, 
        help="Base model name from HuggingFace Hub or local path (e.g., 'google/gemma-3-1b-it', './base_model')"
    )
    required_group.add_argument(
        "--lora", 
        required=True, 
        help="Path to LoRA adapter directory (should contain adapter_model.safetensors and adapter_config.json)"
    )
    
    # Optional arguments
    optional_group = parser.add_argument_group('Optional Arguments')
    optional_group.add_argument(
        "--output", 
        default="output/merged_model", 
        help="Output directory to save merged model (default: %(default)s)"
    )
    optional_group.add_argument(
        "--dtype", 
        default="auto", 
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model loading (default: %(default)s)"
    )
    optional_group.add_argument(
        "--device-map", 
        default="auto", 
        help="Device mapping strategy (default: %(default)s)"
    )
    optional_group.add_argument(
        "--safe-serialization", 
        action="store_true", 
        help="Use safe serialization (safetensors) for saving"
    )
    
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.lora):
        print(f"❌ Error: LoRA adapter path '{args.lora}' does not exist")
        return
        
    if not os.path.exists(os.path.join(args.lora, "adapter_config.json")):
        print(f"❌ Error: '{args.lora}' does not appear to be a valid LoRA adapter (missing adapter_config.json)")
        return
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect model source
    model_source = args.model
    model_type = "local" if is_local_model(args.model) else "hub"
    print(f"🔹 Loading base model ({model_type}): {model_source}")

    # Determine dtype
    if args.dtype == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, args.dtype)
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"✅ Base model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        print(f"✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    print(f"🔹 Loading LoRA adapter from: {args.lora}")
    try:
        model = PeftModel.from_pretrained(base_model, args.lora)
        print(f"✅ LoRA adapter loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LoRA adapter: {e}")
        return

    print("🔄 Merging LoRA weights into base model...")
    try:
        model = model.merge_and_unload()
        print(f"✅ LoRA weights merged successfully")
    except Exception as e:
        print(f"❌ Error merging LoRA weights: {e}")
        return

    print(f"💾 Saving merged model to: {args.output}")
    try:
        model.save_pretrained(
            args.output,
            safe_serialization=args.safe_serialization
        )
        tokenizer.save_pretrained(args.output)
        
        # Save merge information
        merge_info = {
            "base_model": args.model,
            "lora_adapter": args.lora,
            "merged_at": datetime.now().isoformat(),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "safe_serialization": args.safe_serialization
        }
        
        # Try to load LoRA training info if available
        training_info_path = os.path.join(args.lora, "training_info.json")
        if os.path.exists(training_info_path):
            with open(training_info_path, "r", encoding="utf-8") as f:
                merge_info["training_info"] = json.load(f)
        
        with open(os.path.join(args.output, "merge_info.json"), "w", encoding="utf-8") as f:
            json.dump(merge_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Merge complete. Merged model saved to: {args.output}")
        print(f"📋 Merge information saved to: {os.path.join(args.output, 'merge_info.json')}")
        
    except Exception as e:
        print(f"❌ Error saving merged model: {e}")
        return

if __name__ == "__main__":
    main()
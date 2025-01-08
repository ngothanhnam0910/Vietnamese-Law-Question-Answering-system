import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from trl import setup_chat_format

def main(args):
    # Tải tokenizer từ base model hoặc từ LoRA model nếu có thêm các token đặc biệt
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_path)

    # Tải base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Setup chat format (nếu cần)
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)

    # Áp dụng LoRA adapter từ mô hình fine-tuned vào base model
    peft_model = PeftModel.from_pretrained(base_model, args.peft_model_path)

    # Merge LoRA layers vào base model
    peft_model = peft_model.merge_and_unload()

    # Lưu lại mô hình đã merge để sử dụng cho suy luận sau này
    peft_model.save_pretrained(args.output_path, max_shard_size="2GB")
    tokenizer.save_pretrained(args.output_path)

    print(f"Merged model and tokenizer saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA model into base model")

    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="1TuanPham/T-VisStar-7B-v0.1", 
        help="Path to the base model"
    )
    parser.add_argument(
        "--peft_model_path", 
        type=str, 
        default="/home/ivirse/ivirse_all_data/namnt/llm/output_ckp/checkpoint-1145", 
        help="Path to the fine-tuned LoRA model"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="/home/ivirse/ivirse_all_data/namnt/llm/merge_model", 
        help="Path to save the merged model"
    )

    args = parser.parse_args()
    main(args)

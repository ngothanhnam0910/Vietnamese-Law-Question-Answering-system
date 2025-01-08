from transformers import AutoTokenizer, AutoModelForCausalLM
  
tokenizer = AutoTokenizer.from_pretrained("1TuanPham/T-VisStar-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("1TuanPham/T-VisStar-7B-v0.1")
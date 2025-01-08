# !pip install vllm==0.3.3
from vllm import LLM, SamplingParams
from datasets import load_dataset
import time
import pandas as pd
import tqdm

def print_outputs(outputs):
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("-" * 80)
    
    
llm = LLM(model="/home/ivirse/ivirse_all_data/namnt/llm/merge_model",
           max_model_len=2048,
           gpu_memory_utilization=0.8)

tokenizer = llm.get_tokenizer()
sampling = SamplingParams(max_tokens=512, seed=42, temperature=0)

def gen_answer(query, document):

    system_prompt =  """Bạn là một trợ lý thông minh, hãy trở lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan.
                            Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
                NOTE:  - Hãy chỉ trả lời nếu câu trả lời nằm trong tài liệu được truy xuất ra.
                       - Nếu không tìm thấy câu trả lời trong tài liệu truy xuất ra thì hãy trả về : "no" 
    
        # Context: {context} 
        """
    messages=[
    {"role": "system", "content":system_prompt.format(context = document)} ,
    {"role": "user", "content": query}
    ]
    outputs = llm.chat(messages,
                   sampling_params=sampling,
                   use_tqdm=False)
    results = [output.outputs[0].text for output in outputs]
    return results[0]

if __name__ == "__main__":
    df = pd.read_csv("/home/ivirse/namnt/final_project/gen_data/test.csv")
    results_df = pd.DataFrame(columns=["question", "bot_answer", "grouth_truth"])
    for index, row in tqdm.tqdm(df.iterrows()):
        if index <= 1000: 
            question = row['question']
            contexts = row['context']
            grouth_truth = row['answer']
            bot_answer = gen_answer(question, contexts)
            
            print(f"---------bot answer : {bot_answer}")
            # Lưu kết quả vào DataFrame
            results_df = pd.concat([
                results_df, 
                pd.DataFrame({"question": [question], "bot_answer": [bot_answer], "grouth_truth": [grouth_truth]})
            ], ignore_index=True)
            
            results_df.to_csv("/home/ivirse/namnt/final_project/gen_data/results.csv", index=False)
        else:
            continue




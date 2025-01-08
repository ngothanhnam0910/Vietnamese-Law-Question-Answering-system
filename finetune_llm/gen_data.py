import json
import pandas as pd
import ast
import openai
import os
import tqdm
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def gen_answer(query, document):
    prompt = f"""
    Là Một trợ lý ảo thông minh, hãy trả lời câu hỏi dựa trên các tài liệu được cung cấp. Câu trả lời phải gọn, chính xác nhưng vẫn đảm bảo đầy đủ những ý chính.
    # Question : 
    {query}

    # Tài liệu :
    {document}
    """
    response = openai.chat.completions.create(
        model= 'gpt-4o-mini',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0,
    )

    return response.choices[0].message.content


train_data = "train.csv"
output_file = "gen_data/train.csv"

# Đọc dữ liệu từ file train.csv
df = pd.read_csv(train_data)

# Nếu file output.csv đã tồn tại, đọc để tiếp tục ghi từ phần dừng trước đó
if os.path.exists(output_file):
    df_output = pd.read_csv(output_file)
else:
    df_output = pd.DataFrame(columns=["question", "context", "answer"])

# Duyệt từng dòng trong DataFrame
for index, row in tqdm.tqdm(df.iterrows()):
    # Kiểm tra nếu câu hỏi đã được xử lý trước đó
    if 10001 <=index and index <= 11000:
        question = row['question']
        contexts = ast.literal_eval(row['context'])
        full_context = "Dưới đây là toàn bộ thông tin về tài liệu : \n"
        for context in contexts:
            full_context += f"{context} \n"
        full_context += "Kết thúc phần thông tin tài liệu."

        try:
            # Tạo câu trả lời
            answer = gen_answer(question, full_context)
            
            # Thêm kết quả vào DataFrame output
            new_row = {"question": question, "context": full_context, "answer": answer}
            df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

            # Lưu ngay vào file CSV
            df_output.to_csv(output_file, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
        
    else: 
        continue

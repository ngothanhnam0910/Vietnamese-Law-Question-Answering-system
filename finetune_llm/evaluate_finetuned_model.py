import os
from llama_index.llms.openai import OpenAI
import pandas as pd
import tqdm
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

from llama_index.core.evaluation import CorrectnessEvaluator
from llama_index.llms.openai import OpenAI
evaluator = CorrectnessEvaluator(llm=llm)

list_score = []

df = pd.read_csv("/home/ivirse/namnt/final_project/gen_data/results.csv")
for index, row in tqdm.tqdm(df.iterrows()):
    # Kiểm tra nếu câu hỏi đã được xử lý trước đó
    if index <=1000 :
        question = row['question']
        bot_answer = row['bot_answer']
        grouth_truth = row['grouth_truth']
        
        result = evaluator.evaluate(
            query=question,
            response=bot_answer,
            reference=grouth_truth,
        )

        print(f"---------------")
        print(result.score)
        list_score.append(int(result.score))
        print(f"---------------")
        
    else:
        break
    
final_score = sum(list_score)/len(list_score)

print(f"final_score : {final_score}")
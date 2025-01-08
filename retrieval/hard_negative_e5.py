import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
import ast
import re
import json
import tqdm
import pickle

# Function convert to list python
def convert_to_list(s):
    s = s.strip('[]')  # Xóa dấu ngoặc vuông
    elements = s.split()  # Tách thành từng phần tử
    return [int(element) for element in elements]

def convert_str_to_list(input_str):    
    try:
        result = ast.literal_eval(input_str)
        return result
    except (ValueError, SyntaxError):
        print("Input is not a valid Python literal.")
        return None

def split_text_keeping_sentences(text, max_word_count):
    # Tách văn bản thành các câu
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        # Đếm số từ trong câu
        word_count = len(sentence.split())
        
        # Nếu thêm câu vào chunk hiện tại sẽ vượt quá số lượng từ tối đa
        if current_word_count + word_count > max_word_count:
            # Thêm chunk hiện tại vào danh sách chunks
            chunks.append(current_chunk.strip())
            current_chunk = sentence  # Bắt đầu một chunk mới
            current_word_count = word_count  # Đặt lại số lượng từ
        else:
            current_chunk += " " + sentence.strip() if current_chunk else sentence.strip()
            current_word_count += word_count

    # Thêm chunk còn lại nếu có
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

class QdrantSearch_e5:
    def __init__(self, host: str, collection_name: str, model_name: str, use_fp16: bool = True):
        self.client = QdrantClient(host)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name, device="cuda:0")
        
    def encode_query(self, query_text: str):
        """Encode the query text into dense and sparse vectors"""
        query_text = "query: "+ query_text
        dense_vec = self.model.encode(query_text, normalize_embeddings=True)
        return dense_vec

    def search(self, query_text: str, limit: int = 20):
        """Perform the search in Qdrant with the given query text and retrieve up to 50 results"""
        dense_vec = self.encode_query(query_text)
        
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=limit,
            )
        ]
        
        results = self.client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            with_payload=True,
            limit=limit,
        )
        
        return results

class QuestionInference:
    def __init__(self, csv_path: str, save_pair_path: str, qdrant_search: QdrantSearch_e5):
        self.csv_path = csv_path
        self.save_pair_path = save_pair_path
        self.qdrant_search = qdrant_search
    
    def load_questions(self):
        """Load questions and question_ids from CSV file"""
        self.questions = pd.read_csv(self.csv_path)
    
    def infer_and_save(self):
        """Infer each question and save results to a .txt file"""
        save_pairs = []
        for row in tqdm.tqdm(self.questions.itertuples(index=False)):
            question = row.question
            list_id = convert_to_list(row.cid)
            list_context = convert_str_to_list(row.context)

            for context in list_context:
                chunk_context = split_text_keeping_sentences(text=context, max_word_count=400)
                for chunk in chunk_context:
                    save_dict = {}
                    save_dict["question"] = "query: " + question
                    save_dict["document"] = "passage: " + chunk
                    save_dict["relevant"] = 1
                    save_pairs.append(save_dict)

            results = self.qdrant_search.search(query_text=question, limit=25)
            for result in results.points:
                infor_id = int(result.payload["infor_id"])
                if infor_id in list_id:
                    continue
                else:
                    text = result.payload["text"]
                    save_dict = {}
                    save_dict["question"] = "query: " + question
                    save_dict["document"] = "passage: " + text
                    save_dict["relevant"] = 0
                    save_pairs.append(save_dict)

        print(f"save pairs:{len(save_pairs)}")

        os.makedirs(self.save_pair_path, exist_ok=True)
        with open(os.path.join(self.save_pair_path, f"save_pairs_e5_top25.pkl"), "wb") as pair_file:
            pickle.dump(save_pairs, pair_file)

if __name__ == "__main__":
    # Đường dẫn file CSV đầu vào và file TXT đầu ra
    csv_path = 'train.csv'  # Đường dẫn đến file CSV của bạn
    output_path = '...'  # Đường dẫn đến file TXT đầu ra
    
    # Khởi tạo QdrantSearch
    qdrant_search = QdrantSearch_e5(
        host="http://localhost:6333",
        collection_name="law_with_e5_emb_not_finetune",
        model_name="intfloat/multilingual-e5-large",
        use_fp16=True
    )
    # # Khởi tạo và thực thi quá trình infer
    inference = QuestionInference(csv_path, output_path, qdrant_search)
    inference.load_questions()
    inference.infer_and_save()

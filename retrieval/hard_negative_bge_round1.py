import pandas as pd
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
import os
import ast
import re
import json
import tqdm
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

class QdrantSearch_bge:
    def __init__(self, host: str, collection_name: str, model_name: str, use_fp16: bool = True):
        self.client = QdrantClient(host)
        self.collection_name = collection_name
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        
    def encode_query(self, query_text: str):
        """Encode the query text into dense and sparse vectors"""
        emb = self.model.encode(query_text, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        emb_sparse = emb['lexical_weights']
        dense_vec = emb['dense_vecs']
        
        indices = list(emb_sparse.keys())
        values = list(emb_sparse.values())
        
        return dense_vec, indices, values

    def search(self, query_text: str, limit: int = 25):
        """Perform the search in Qdrant with the given query text and retrieve up to 50 results"""
        dense_vec, indices, values = self.encode_query(query_text)
        
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=indices,
                    values=values
                ),
                using="sparse",
                limit=limit,
            ),
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
    def __init__(self, csv_path: str, save_pair_path: str, qdrant_search: QdrantSearch):
        self.csv_path = csv_path
        self.save_pair_path = save_pair_path
        self.qdrant_search = qdrant_search
    
    def load_questions(self):
        """Load questions and question_ids from CSV file"""
        self.questions = pd.read_csv(self.csv_path)
    
    def infer_and_save(self):
        """Infer each question and save results to a .txt file"""
        file_name = "data_round1"
        with open(os.path.join(self.save_pair_path, file_name + '.json'), 'w') as output_file:
            for row in tqdm.tqdm(self.questions.itertuples(index=False)):
                question = row.question
                list_id = convert_to_list(row.cid)
                list_context = convert_str_to_list(row.context)
                # create_data for bge
                save_dict = {}
                save_dict["query"] = question
                save_dict["pos"] = []
                save_dict["neg"] = []
                for context in list_context:
                    chunk_context = split_text_keeping_sentences(text=context, max_word_count=400)
                    save_dict["pos"] += chunk_context

                results = self.qdrant_search.search(query_text=question, limit=25)
                for result in results.points:
                    infor_id = int(result.payload["infor_id"])
                    if infor_id in list_id:
                        continue
                    else:
                        text = result.payload["text"]
                        save_dict["neg"].append(text)

                output_file.write(json.dumps(save_dict,ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Đường dẫn file CSV đầu vào và file TXT đầu ra
    csv_path = 'train.csv'  # Đường dẫn đến file CSV của bạn
    output_path = '...'  # Đường dẫn đến file TXT đầu ra
    
    # Khởi tạo QdrantSearch
    qdrant_search = QdrantSearch(
        host="http://localhost:6333",
        collection_name="law_with_bge_round1",
        model_name="BAAI/bge-m3",
        use_fp16=True
    )
    # # Khởi tạo và thực thi quá trình infer
    inference = QuestionInference(csv_path, output_path, qdrant_search)
    inference.load_questions()
    inference.infer_and_save()

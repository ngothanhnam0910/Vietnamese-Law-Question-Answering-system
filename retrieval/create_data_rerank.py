
from qdrant_client import QdrantClient, models
from hard_negative_bge_round1 import QdrantSearch_bge
from hard_negative_e5 import QdrantSearch_e5
from hard_negative_bge_round1 import convert_to_list, convert_str_to_list, split_text_keeping_sentences
import os
import pandas as pd
import numpy as np
import json
import tqdm
import pickle

class QuestionInference:
    def __init__(self, csv_path: str, save_pair_path: str, qdrant_search_bge: QdrantSearch_bge, qdrant_search_e5: QdrantSearch_e5, qdrant_search_jina: QdrantSearch_jina):
        self.csv_path = csv_path
        self.save_pair_path = save_pair_path
        self.qdrant_search_bge = qdrant_search_bge
        self.qdrant_search_e5 = qdrant_search_e5
    
    def load_questions(self):
        """Load questions and question_ids from CSV file"""
        self.questions = pd.read_csv(self.csv_path)
    
    def infer_and_save(self):
        """Infer each question and save results to json file"""
        file_name = "data_reranking"
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

                results_bge = self.qdrant_search_bge.search(query_text=question, limit=25)
                result_e5 = self.qdrant_search_e5.search(query_text=question, limit=25)
                
                list_chunk_id = []
                for results in [results_bge, result_e5]:
                    for result in results.points:
                        # nếu là positive thì bỏ qua không append vào nữa.
                        infor_id = int(result.payload["infor_id"])
                        if infor_id in list_id:
                            continue

                        # nếu không phải id của positive thì check xem text này đã append hay chưa => chưa thì append vào list negative
                        else:
                            if result.payload["chunk_id"] not in list_chunk_id:
                                text = result.payload["text"]
                                save_dict["neg"].append(text)
                                list_chunk_id.append(result.payload["chunk_id"])
                            else:
                                continue

                output_file.write(json.dumps(save_dict,ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # Đường dẫn file CSV đầu vào và file TXT đầu ra
    csv_path = 'train_data.csv'  # Đường dẫn đến file CSV của bạn
    output_path = '/format_data/rerank'  # Đường dẫn đến file TXT đầu ra
    
    # Khởi tạo QdrantSearch
    qdrant_search_bge = QdrantSearch_bge(
        host="http://localhost:6333",
        collection_name="law_with_bge_round1",
        model_name="BAAI/bge-m3",
        use_fp16=True
    )
    qdrant_search_e5 = QdrantSearch_e5(
        host="http://localhost:6333",
        collection_name="law_with_e5_emb_not_finetune",
        model_name="intfloat/multilingual-e5-large",
        use_fp16=True
    )

    inference = QuestionInference(csv_path, output_path, qdrant_search_bge, qdrant_search_e5)
    inference.load_questions()
    inference.infer_and_save()
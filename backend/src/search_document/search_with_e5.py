from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import tqdm

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

    

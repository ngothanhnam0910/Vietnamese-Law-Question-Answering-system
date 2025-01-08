from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
import tqdm

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

    def search(self, query_text: str, limit: int = 20):
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



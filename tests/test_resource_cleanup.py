from app.core.vectordb.base import BaseVectorDB, BatchUpsertResult, VectorHit
from app.retrieval.repository import RetrievalRepository


class _FakeVectorDB(BaseVectorDB):
    def __init__(self):
        self.close_calls = 0

    @property
    def kind(self) -> str:
        return "fake"

    def health_check(self) -> bool:
        return True

    def ensure_collection(self, collection: str, *, embedding_dim: int, distance_metric: str = "cosine") -> None:
        return None

    def delete_collection(self, collection: str) -> None:
        return None

    def upsert(self, *, collection: str, doc_id: str, embedding, text: str, metadata) -> None:
        return None

    def batch_upsert(self, *, collection: str, doc_ids, embeddings, texts, metadatas) -> BatchUpsertResult:
        return BatchUpsertResult()

    def search(self, *, collection: str, query_embedding, top_k: int = 10, filters=None) -> list[VectorHit]:
        return []

    def exists(self, *, collection: str, doc_ids):
        return set()

    def get_by_ids(self, *, collection: str, doc_ids):
        return {}

    def delete(self, *, collection: str, doc_id: str) -> None:
        return None

    def delete_many(self, *, collection: str, doc_ids) -> int:
        return 0

    def count(self, *, collection: str) -> int:
        return 0

    def close(self) -> None:
        self.close_calls += 1


def test_retrieval_repository_close_is_idempotent():
    vectordb = _FakeVectorDB()
    repository = RetrievalRepository(db_session=None, vectordb=vectordb, collection="test")

    repository.close()
    repository.close()

    assert vectordb.close_calls == 1

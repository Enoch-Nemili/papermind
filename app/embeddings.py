"""
fastembed as a LangChain Embeddings object (what PGVector expects).

Replaces langchain-community's FastEmbedEmbeddings with the exact same calls, so vectors
match the ones already stored: documents go through model.embed(), queries through
model.query_embed() (which adds the model's query instruction for retrieval).
"""

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name, max_length=512, batch_size=256):
        # Imported lazily: loading fastembed + the model is slow, and tests don't need it.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, max_length=max_length)
        self._batch_size = batch_size

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=self._batch_size)]

    def embed_query(self, text):
        return next(iter(self._model.query_embed(text, batch_size=self._batch_size))).tolist()

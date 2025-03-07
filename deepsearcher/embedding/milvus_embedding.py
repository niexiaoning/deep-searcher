from typing import List

import numpy as np

from deepsearcher.embedding.base import BaseEmbedding

MILVUS_MODEL_DIM_MAP = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-base-zh-v1.5": 768,
    "BAAI/bge-small-zh-v1.5": 384,
    "GPTCache/paraphrase-albert-onnx": 768,
    "default": 768,  # 'GPTCache/paraphrase-albert-onnx',
    # see https://github.com/milvus-io/milvus-model/blob/4974e2d190169618a06359bcda040eaed73c4d0f/src/pymilvus/model/dense/onnx.py#L12
    "jina-embeddings-v3": 1024,  # required jina api key
    "BAAI/bge-m3": 1024,
}


class MilvusEmbedding(BaseEmbedding):
    def __init__(self, model: str = None, **kwargs) -> None:
        model_name = model
        from pymilvus import model

        if "model_name" in kwargs and (not model_name or model_name == "default"):
            model_name = kwargs.pop("model_name")

        if not model_name or model_name in [
            "default",
            "GPTCache/paraphrase-albert-onnx",
        ]:
            self.model = model.DefaultEmbeddingFunction(**kwargs)
        elif model_name == "bge-m3":
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel('BAAI/bge-m3')
        else:
            if model_name.startswith("jina-"):
                self.model = model.dense.JinaEmbeddingFunction(model_name, **kwargs)
            elif model_name.startswith("BAAI/"):
                self.model = model.dense.SentenceTransformerEmbeddingFunction(model_name, **kwargs)
            else:
                # Only support default model and BGE series model
                raise ValueError(f"Currently unsupported model name: {model_name}")

    def embed_query(self, text: str) -> List[float]:
        if self.model_name == "bge-m3":
            return self.model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=True)
        return self.model.encode_queries([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.model_name == "bge-m3":
            return self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        embeddings = self.model.encode_documents(texts)
        if isinstance(embeddings[0], np.ndarray):
            return [embedding.tolist() for embedding in embeddings]
        else:
            return embeddings

    @property
    def dimension(self) -> int:
        return self.model.dim  # or MILVUS_MODEL_DIM_MAP[self.model_name]

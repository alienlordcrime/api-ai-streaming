from chromadb import Collection, QueryResult
from chromadb.config import Settings
from langchain_chroma import Chroma

import chromadb

from api.repositories.ollama_embeddings_repository import OllamaEmbeddingsRepository

class VectorstoreLegalRepository():
    
    
    def __init__(self):
        
        server_onpremise= "localhost"
        
        self.chroma_client = chromadb.HttpClient(
            host=server_onpremise,
            port=8000,
            ssl=False,
            settings=Settings(
                chroma_server_ssl_enabled=False
            )
        )

        pass
    
    def get_vectorstore(self, name: str):
        
        _OllamaEmbeddingsRepository = OllamaEmbeddingsRepository()
        embedding_function= _OllamaEmbeddingsRepository.get_embedding_function()
        
        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name= name,
            embedding_function=embedding_function
        )
        
        return vectorstore
     
    def get_collection_name(self, name: str):
        
        _OllamaEmbeddingsRepository = OllamaEmbeddingsRepository()
        embedding_function= _OllamaEmbeddingsRepository.get_embedding_function()
        
        print("Aqui traigo el function", embedding_function)
        print("Vamos a retornar algo \n\n")
        
        self.chroma_client.delete_collection(name)
        
        return self.chroma_client.get_or_create_collection(
            name, 
            metadata= {
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 200
            },
            embedding_function= embedding_function 
        )
    
    def search_by_metadata(self, collection: Collection, options: dict) -> QueryResult:
        return collection.query(
            query_texts=[""],
            where= options
        )
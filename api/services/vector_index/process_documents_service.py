import uuid
import re

from api.repositories.ollama_embeddings_repository import OllamaEmbeddingsRepository
from api.repositories.vectorstore_legal_repository import VectorstoreLegalRepository
from api.services.ocr.ocr_functions_service import process_pdf_pages_ultra_optimized

class ProcessDocumentsService:
    
    def __init__(self):
        pass
    
    
    async def add_document_vector_datawarehouse(self, pdf_binary_data: bytes, filename: str):
        
        _vectorstoreLegalRepository= VectorstoreLegalRepository()
        collection= _vectorstoreLegalRepository.get_collection_name_embed_function('legal_contracts')
        results= _vectorstoreLegalRepository.search_by_metadata(collection, {
            "filename": filename
        })
        
        metadatas= results.get("metadatas", None)
        exist= any([any(meta for meta in element if meta['filename'] == filename) for element in metadatas])
            
        if not exist:
        
            reference_id= uuid.uuid4()
            _ollamaEmbeddingsRepository = OllamaEmbeddingsRepository()
        
            content_string, count_pages= await process_pdf_pages_ultra_optimized(pdf_binary_data)
            content_string = re.sub('\n',' ', content_string)
            
            texts, embedding_vectors= _ollamaEmbeddingsRepository.generate_by_long_text(content_string)
            
            collection.add(
                ids=[f"doc_{i}" for i in range(len(texts))],
                metadatas=[
                {
                    "filename": filename,
                    "paginas": count_pages,
                    "reference_id": str(reference_id),
                    "source": f"doc_{i}" 
                } for i in range(len(texts))
                ],
                embeddings=embedding_vectors,
                documents=texts
            )

        
        return {
            "message" : "Documento agregado"
        }
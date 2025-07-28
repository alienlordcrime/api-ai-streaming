
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from chromadb.utils.embedding_functions.chroma_langchain_embedding_function import create_langchain_embedding

class OllamaEmbeddingsRepository:
    
    def __init__(self):
        
        self.OLLAMA_SERVER= "http://10.0.0.14:11434"
        self.EMBEDDING_MODEL= "bge-m3:latest"
        
        pass
    
    def get_ollama_embedding_instance(self):
        return OllamaEmbeddings(
            base_url=self.OLLAMA_SERVER, 
            model=self.EMBEDDING_MODEL,
            temperature=0.4
        )
    
    def get_embedding_function(self):
        
        print("\n\nVamos por la función de embedding...\n\n")
        
        embeddings = OllamaEmbeddings(
            base_url=self.OLLAMA_SERVER, 
            model=self.EMBEDDING_MODEL,
            temperature=0.4
        )
        embedding_function = create_langchain_embedding(embeddings)
        
        print("\n\nsi la traemos\n\n")
        
        return embedding_function
    
    
    def generate_by_long_text(self, text: str):
        # 1. Instanciar embeddings de Ollama
        embeddings = OllamaEmbeddings(
            base_url=self.OLLAMA_SERVER,
            model=self.EMBEDDING_MODEL,
            temperature=0.4
        )

        # 2. Configurar el splitter semántico
        text_splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=2,                           # Ventana de 2 oraciones
            breakpoint_threshold_type="percentile",  # Usa percentiles para detectar cambios de tema
            breakpoint_threshold_amount=95.0,        # Umbral 95º percentil
            sentence_split_regex=r"(?<=[.?!])\s+"    # Regex para separar oraciones
        )

        # 3. Aplicar splitter al texto completo
        docs: list[Document] = text_splitter.create_documents([text])

        # 4. Extraer contenidos de cada chunk
        texts = [doc.page_content for doc in docs]

        # 5. Calcular vectores de embedding de cada chunk
        embedding_vectors = embeddings.embed_documents(texts)

        return texts, embedding_vectors
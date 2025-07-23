import asyncio
from langchain_core.documents import Document
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.retrievers import EnsembleRetriever
from langchain_core.callbacks.manager import AsyncCallbackManager

from api.repositories.vectorstore_legal_repository import VectorstoreLegalRepository
from api.services.models_llm.utils.streaming_callback_handler import StreamingCallbackHandler

from typing import AsyncGenerator, List
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

callback_handler = StreamingCallbackHandler()

def _summarize(docs: List[Document]) -> str:
    """Genera la cadena con los extractos para el prompt."""
    lines = [
        f"{'-'*10}\nExtracción # {i+1}: '{d.page_content}'."
        for i, d in enumerate(docs)
    ]
    return "\n".join(lines)


class RagLegalService:

    def __init__(self) -> None:
        
        logging.info("Creando el vector store")
        
        llm_url: str = "http://10.6.14.54:11434" 
        model_name: str = "gemma3n:e4b"
        
        self._vectorstore = (VectorstoreLegalRepository().get_vectorstore("legal_contracts"))
        
        self._retriever = self._build_ensemble_retriever()
        
        self._llm = self._build_llm(llm_url, model_name)
        self._prompt = self._build_prompt()
        self._rag_chain = self._build_chain()
        
        logging.info("Vector stores configurado")
        
    # ---------- Builders ----------
    
    def _build_ensemble_retriever(self) -> EnsembleRetriever:
        diverse = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "lambda_mult": 0.40},
        )
        relevant = self._vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "lambda_mult": 0.70},
        )
        return EnsembleRetriever(
            retrievers=[diverse, relevant],
            weights=[0.55, 0.45],
        )

    @staticmethod
    def _build_llm(url: str, model_name: str) -> ChatOllama:
        return ChatOllama(
            base_url=url,
            model=model_name,
            num_ctx=10_000,
            temperature=0.5,
            top_p=0.8,
            top_k=20,
            repeat_penalty=1.1,
            num_predict=200,
            streaming=True,
            callback_manager=AsyncCallbackManager([callback_handler])
        )

    @staticmethod
    def _build_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """
                Eres una IA RAG especializada en contratos y documentos legales de la Federación Mexicana de Fútbol.
                Utiliza terminología jurídica mexicana; responde en un máximo de 80 palabras.
                Pregunta: {question}

                Contexto:
                {context}
            """.strip()
        )

    def _build_chain(self):
        
        context_chain = (
            self._retriever
            | RunnableLambda(_summarize)
        )

        return (
            RunnableParallel(
                context=context_chain,
                question=RunnablePassthrough(),
            )
            | self._prompt
            | self._llm
            | StrOutputParser()
        )

    # ---------- Llamado al API ----------

    async def get_chat_completions(self, user_query: str) -> AsyncGenerator[str, None]:
        
        try:
            
            loop = asyncio.get_event_loop()
            
            def run_chain():
                return self._rag_chain.invoke(user_query)
            
            # Ejecutar en thread pool para no bloquear el event loop
            task = loop.run_in_executor(None, run_chain)
                        
            async for token in callback_handler.aiter():
                yield token
                
            # Esperar a que termine la ejecución
            await task
            
        except Exception as e:
            error_msg = f"data: Error durante el análisis: {str(e)}\n\n"
            yield error_msg
            yield f"data: [DONE]\n\n"
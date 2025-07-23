import asyncio
from typing import AsyncGenerator
from langchain_core.callbacks.base import AsyncCallbackHandler

class StreamingCallbackHandler(AsyncCallbackHandler):
    """Callback handler para capturar tokens y enviarlos a la queue"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """Se ejecuta al inicio del LLM"""
        self.done.clear()
        
    async def on_chat_model_start(self, serialized, messages, **kwargs):
        """Se ejecuta al inicio del chat model"""
        self.done.clear()
        
    async def on_llm_new_token(self, token: str, **kwargs):
        """Captura cada nuevo token generado"""
        if token:
            await self.queue.put(f"data: {token}\n\n")
    
    async def on_llm_end(self, response, **kwargs):
        """Se ejecuta al final del LLM"""
        await self.queue.put(f"data: [DONE]\n\n")
        self.done.set()
        
    async def on_llm_error(self, error, **kwargs):
        """Maneja errores del LLM"""
        await self.queue.put(f"data: Error: {str(error)}\n\n")
        self.done.set()
        
    async def aiter(self) -> AsyncGenerator[str, None]:
        """Genera tokens asincrónicamente"""
        while not self.done.is_set() or not self.queue.empty():
            try:
                # Esperar por un token con timeout
                token = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                yield token
            except asyncio.TimeoutError:
                # Verificar si ya terminamos
                if self.done.is_set():
                    break
                continue
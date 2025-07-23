
from typing import Dict, List
from pydantic import BaseModel, Field

class UserMessage(BaseModel):

    query: str = Field(
        ...,
        description="Pregunta que se desea responder respecto a los archivos."
    )

    history: List[Dict[str, str]] = Field(
        ...,
        description="Historial de mensajes registrados en el chat."
    )

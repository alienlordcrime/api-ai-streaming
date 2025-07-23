from typing import List, Dict
from pydantic import BaseModel, Field

class PDFAnalysisRequest(BaseModel):
    files: List[Dict[str, str]] = Field(
        ...,
        description="Lista de archivos con nombre y contenido."
    )
    history: List[Dict[str, str]] = Field(
        ...,
        description="Historial de mensajes registrados en el chat."
    )
    query: str = Field(
        ...,
        description="Pregunta que se desea responder respecto a los archivos."
    )
    chunk_size: int = Field(
        1_200,
        gt=0,
        description="Máximo de caracteres por fragmento de análisis."
    )
    chunk_overlap: int = Field(
        150,
        ge=0,
        description="Número de caracteres que se solapan entre fragmentos."
    )
    k_documents: int = Field(
        8,
        gt=0,
        description="Cantidad de documentos candidatos a considerar."
    )
    temperature: float = Field(
        0.1,
        ge=0,
        le=2,
        description="Temperatura del modelo de lenguaje."
    )

    class Config:
        schema_extra = {
            "example": {
                "files": [
                    {
                        "filename": "informe_anual.pdf",
                        "content": "Contenido extraído del PDF número uno..."
                    },
                    {
                        "filename": "resumen_ejecutivo.pdf",
                        "content": "Contenido extraído del PDF número dos..."
                    }
                ],
                "query": "¿Cuáles son los principales hallazgos del conjunto de documentos?",
                "chunk_size": 1_200,
                "chunk_overlap": 150,
                "k_documents": 8,
                "temperature": 0.1
            }
        }

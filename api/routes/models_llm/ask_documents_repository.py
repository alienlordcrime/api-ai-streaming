from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.routes.models_llm.schemas.documents_question import PDFAnalysisRequest
from api.services.models_llm.document_question_service import DocumentQuestionService

ask_documents_router = APIRouter(prefix="/ask_documents", tags=["LLM", "AI"])

_documentQuestionService= DocumentQuestionService()

"""LEER CONEXTO Y SACAR RESPUESTAS"""
@ask_documents_router.post("/completions")
def ask_document(request: PDFAnalysisRequest):
    
    # Validaciones básicas
    if not request.files:
        raise HTTPException(status_code=400, detail="Es necesario enviar archivos")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La consulta no puede estar vacía")
    
    
    # Headers para SSE (Server-Sent Events)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "X-Accel-Buffering": "no"  # Para nginx
    }
    
    return StreamingResponse(
        _documentQuestionService.analyze_pdf_streaming(request),
        media_type="text/event-stream",
        headers=headers,
        status_code=200
    )

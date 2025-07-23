import gc
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.routes.vector_index.schemas.user_message import UserMessage
from api.services.vector_index.process_documents_service import ProcessDocumentsService
from api.services.vector_index.rag_legal_service import RagLegalService

vector_index_router = APIRouter(prefix="/vector-index", tags=["Embedding", "AI"])

@vector_index_router.post("/document-add")
async def add_document_vector_warehouse(
    request: Request,
    background_tasks: BackgroundTasks,
    content_type: Optional[str] = Header(None),
    x_filename: Optional[str] = Header(None, alias="X-Filename"),
    _processDocumentsService: ProcessDocumentsService = Depends()
    ):
    
    
    if not content_type:
        return JSONResponse(
            content={"error": "Content-Type header is required"}, 
            status_code=400
        )
    
    if not x_filename:
        return JSONResponse(
            content={"error": "X-Filename header is required"}, 
            status_code=400
        )
    
    pdf_binary_data: bytes = await request.body()
    
    await _processDocumentsService.add_document_vector_datawarehouse(pdf_binary_data, x_filename)
    
    background_tasks.add_task(lambda: gc.collect())
    
    
    return


@vector_index_router.post("/completions")
def get_completions(
    request: Request,
    user_message: UserMessage
):
    
    _ragLegalService: RagLegalService  = request.app.state.rag_legal_service
    
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "X-Accel-Buffering": "no"
    }
    
    return StreamingResponse(
        _ragLegalService.get_chat_completions(user_message.query),
        media_type="text/event-stream",
        headers=headers,
        status_code=200
    )
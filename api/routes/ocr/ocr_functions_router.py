from typing import Optional
from fastapi import APIRouter, Request, BackgroundTasks, Header
from fastapi.responses import JSONResponse

import logging
import gc

from api.services.ocr.ocr_functions_service import process_pdf_pages_ultra_optimized

ocr_functions_router = APIRouter(prefix="/apply-ocr", tags=["Surya", "OCR", "AI"])

"""LEER PDF Y APLICAR OCR"""
@ocr_functions_router.put("/process")
async def read_pdf_optimized(
    request: Request,
    background_tasks: BackgroundTasks,
    content_type: Optional[str] = Header(None),
    x_filename: Optional[str] = Header(None, alias="X-Filename")
):
    # Validaciones tempranas
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
    
    try:
        logging.info(f"Procesando: {x_filename}, tipo: {content_type}")
        
        # Leer datos binarios
        pdf_binary_data = await request.body()
        file_size = len(pdf_binary_data)
        
        logging.info(f"Tamaño del archivo: {file_size} bytes")
        
        # Procesar PDF de forma optimizada
        (page_content, page_count) = await process_pdf_pages_ultra_optimized(pdf_binary_data)
        
        # Liberar referencia a datos binarios
        pdf_binary_data = None
        
        # Agregar tarea de limpieza en background
        background_tasks.add_task(lambda: gc.collect())
        
        return JSONResponse(content={
            "page_content": page_content,
            "metadata": {
                "source": x_filename,
                "page_count": page_count
            }
        })
        
    except Exception as e:
        logging.error(f"Error procesando {x_filename}: {str(e)}")
        return JSONResponse(
            content={"error": f"Error processing file: {str(e)}"}, 
            status_code=500
        )

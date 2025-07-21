from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from api.standalone.surya_service import SuryaPredictor

import logging
import os

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

logging.info("Iniciando API AI LLM")

@asynccontextmanager
async def lifespan(app):
    # Inicialización al startup
    predictor = SuryaPredictor()
    await predictor.get_predictors()  # Pre-cargar modelos
    
    # Configurar executor para procesamiento CPU-intensive
    app.state.executor = ProcessPoolExecutor(max_workers=min(4, os.cpu_count()))
    
    logging.info("Aplicación iniciada con recursos pre-cargados")
    yield
    
    # Limpieza al shutdown  
    app.state.executor.shutdown(wait=True)
    logging.info("Recursos liberados")

logging.info("Cargando Lifespan")

app = FastAPI(title="API AI LLM - DATA MANAGEMENT", version="0.1", lifespan= lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):        
    if isinstance(exc, HTTPException):
        
        content = {"error": exc.detail} if isinstance(exc.detail, str) else exc.detail
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers={"Access-Control-Allow-Origin": "*"}
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "No se logró procesar la solicitud"},
            headers={"Access-Control-Allow-Origin": "*"}
        )

logging.info("Registrando rutas")

from api.routes.ocr.ocr_functions_router import ocr_functions_router
app.include_router(ocr_functions_router)
        
logging.info("Configurando rutas")
        
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,  
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

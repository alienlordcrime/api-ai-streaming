from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request

app = FastAPI(title="API AI LLM - DATA MANAGEMENT", version="0.1")

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
        
from api.routes.ocr.ocr_functions_router import ocr_functions_router
app.include_router(ocr_functions_router)
        
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,  
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

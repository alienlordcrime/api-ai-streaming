from typing import Optional
from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

import fitz
import io

recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()

ocr_functions_router = APIRouter(prefix="/apply-ocr", tags=["Surya", "OCR", "AI"])

@ocr_functions_router.put("/process")
async def read_pdf(
    request: Request,
    content_type: Optional[str] = Header(None),
    x_filename: Optional[str] = Header(None, alias="X-Filename")
):
    
    # Verificar que se proporcionaron los headers requeridos
    if not content_type:
        return {"error": "Content-Type header is required"}
    
    if not x_filename:
        return {"error": "X-Filename header is required"}
    
    print(f"Nos llego: {x_filename}, de tipo: {content_type}")
    pdf_binary_data = await request.body()
    file_size = len(pdf_binary_data)
    
    print(f"Tamaño del archivo: {file_size}")
    
    (page_content, page_count)= process_pdf_pages(pdf_binary_data)
    
    return JSONResponse(content={
        "page_content": page_content,
        "metadata": {
            "source": x_filename,
            "page_count": page_count,
            # "extraction_method": "external_ocr",
            # "language": "es"
        }
    })


def convert_page_to_pil(page) -> Image.Image:
    # Configurar matriz para alta calidad (300 DPI)
    zoom = 300 / 72  # 300 DPI
    mat = fitz.Matrix(zoom, zoom)
    
    # Obtener pixmap de la página
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convertir a bytes de imagen
    img_bytes = pix.tobytes("png")
    
    # Convertir a PIL Image
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    return pil_image
    

def process_pdf_pages(pdf_binary_data: bytes) -> tuple:

    ## OCR con Surya
    content_pdf: str = ""
    count_pages: int = 0
    
    # Abrir el PDF desde los datos binarios
    pdf_doc = fitz.open(stream=pdf_binary_data, filetype="pdf")
    
    try:
        # Procesar cada página
        count_pages= len(pdf_doc)
        for page_num in range(count_pages):
            page = pdf_doc.load_page(page_num)
            
            # Convertir página a imagen PIL
            pil_image = convert_page_to_pil(page)
            
            # Aplicar OCR con Surya
            ocr_result = apply_surya_ocr(pil_image)
            content_pdf += f" {ocr_result}"
            
    finally:
        pdf_doc.close()
    
    return (content_pdf, count_pages)


def apply_surya_ocr(pil_image: Image.Image) -> str:
    
    predictions = recognition_predictor([pil_image], det_predictor=detection_predictor)
    text_extract: str = ""
    
    for prediction in predictions:
        for text_line in prediction.text_lines:
            text_extract += f" {text_line.text}"
    
    return text_extract
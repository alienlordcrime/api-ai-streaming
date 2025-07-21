from PIL import Image
import fitz
import io
import gc

from api.standalone.surya_service import SuryaPredictor


def convert_page_to_pil_optimized(page) -> Image.Image:
    """Convierte página PDF a PIL Image de forma optimizada"""
    # Usar DPI óptimo para Surya (no más de 300, puede ser contraproducente)
    zoom = 200 / 72  # 200 DPI es suficiente para Surya
    mat = fitz.Matrix(zoom, zoom)
    
    # Obtener pixmap optimizado
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convertir directamente a PIL sin pasos intermedios
    img_bytes = pix.tobytes("png")
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    # Liberar memoria del pixmap inmediatamente
    pix = None
    
    return pil_image

       
async def process_pdf_pages_optimized(pdf_binary_data: bytes) -> tuple:
    """Procesa páginas PDF de forma optimizada con manejo de memoria"""
    content_pdf = ""
    count_pages = 0
    
    # Usar context manager para garantizar limpieza
    pdf_doc = fitz.open(stream=pdf_binary_data, filetype="pdf")
    
    try:
        count_pages = len(pdf_doc)
        
        # Procesar páginas en lotes pequeños para evitar acumulación de memoria
        batch_size = 5  # Procesar de 5 en 5 páginas
        
        for i in range(0, count_pages, batch_size):
            batch_end = min(i + batch_size, count_pages)
            batch_texts = []
            
            # Procesar lote de páginas
            for page_num in range(i, batch_end):
                page = pdf_doc.load_page(page_num)
                
                # Convertir a PIL Image
                pil_image = convert_page_to_pil_optimized(page)
                
                # Aplicar OCR
                predictor = SuryaPredictor()
                recognition_predictor, detection_predictor = await predictor.get_predictors()
                
                predictions = recognition_predictor([pil_image], det_predictor=detection_predictor)
                
                text_extract = ""
                for prediction in predictions:
                    for text_line in prediction.text_lines:
                        text_extract += f" {text_line.text}"
                
                batch_texts.append(text_extract.strip())
                
                # Liberar memoria de la imagen inmediatamente
                pil_image.close() if hasattr(pil_image, 'close') else None
                pil_image = None
                
            # Agregar textos del lote
            content_pdf += " ".join(batch_texts) + " "
            
            # Forzar garbage collection después de cada lote
            gc.collect()
            
    finally:
        pdf_doc.close()
    
    return (content_pdf.strip(), count_pages)


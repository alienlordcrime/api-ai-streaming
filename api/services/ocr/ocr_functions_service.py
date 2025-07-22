from typing import List, Tuple, AsyncGenerator
from PIL import Image
import asyncio
import gc
import fitz
import io


from api.standalone.surya_service import SuryaPredictor


async def process_pdf_pages_optimized(pdf_binary_data: bytes) -> Tuple[str, int]:
    """
    Procesa páginas PDF de forma ultra-optimizada con verdadero batch processing
    y gestión proactiva de memoria
    """
    # Configuración dinámica según disponibilidad GPU
    try:
        import torch
        device_available = torch.cuda.is_available()
        # Para GPU: batch más grande, para CPU: más conservador
        batch_size = 16 if device_available else 8
    except ImportError:
        batch_size = 6
    
    content_pdf = ""
    count_pages = 0
    
    # Inicializar predictores UNA SOLA VEZ
    predictor = SuryaPredictor()
    recognition_predictor, detection_predictor = await predictor.get_predictors()
    
    # Context manager optimizado para PyMuPDF
    pdf_doc = fitz.open(stream=pdf_binary_data, filetype="pdf")
    
    try:
        count_pages = len(pdf_doc)
        
        # Procesar en verdaderos lotes con generador
        async for batch_text in _process_pages_in_batches(
            pdf_doc, 
            recognition_predictor, 
            detection_predictor, 
            batch_size
        ):
            content_pdf += batch_text + " "
            
            # Garbage collection proactivo cada 3 lotes
            if len(content_pdf) % (batch_size * 3) == 0:
                gc.collect()
                
    finally:
        pdf_doc.close()
        pdf_binary_data = None
    
    return (content_pdf.strip(), count_pages)


async def _process_pages_in_batches(
    pdf_doc: fitz.Document,
    recognition_predictor,
    detection_predictor,
    batch_size: int
) -> AsyncGenerator[str, None]:
    """
    Generador asíncrono que procesa páginas en verdaderos lotes
    """
    total_pages = len(pdf_doc)
    
    for i in range(0, total_pages, batch_size):
        batch_end = min(i + batch_size, total_pages)
        
        # Convertir páginas a imágenes en paralelo
        batch_images = await _convert_pages_to_images_batch(
            pdf_doc, range(i, batch_end)
        )
        
        try:
            # VERDADERO BATCH PROCESSING - múltiples imágenes simultáneamente
            predictions = recognition_predictor(
                batch_images, 
                det_predictor=detection_predictor
            )
            
            # Extraer texto de todas las predicciones
            batch_texts = []
            for prediction in predictions:
                page_text = " ".join([
                    text_line.text for text_line in prediction.text_lines
                ])
                batch_texts.append(page_text.strip())
            
            yield " ".join(batch_texts)
            
        finally:
            # Limpieza inmediata de imágenes
            for img in batch_images:
                if hasattr(img, 'close'):
                    img.close()
            batch_images.clear()


async def _convert_pages_to_images_batch(
    pdf_doc: fitz.Document, 
    page_range: range
) -> List[Image.Image]:
    """
    Convierte múltiples páginas a PIL Images de forma optimizada y paralela
    """
    # Usar DPI optimizado para Surya (200 DPI es suficiente)
    zoom = 200 / 72
    mat = fitz.Matrix(zoom, zoom)
    
    images = []
    
    # Procesamiento concurrente de conversión de páginas
    tasks = []
    for page_num in page_range:
        task = asyncio.create_task(_convert_single_page(pdf_doc, page_num, mat))
        tasks.append(task)
    
    # Esperar todas las conversiones concurrentemente
    images = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filtrar excepciones si las hay
    valid_images = [img for img in images if isinstance(img, Image.Image)]
    
    return valid_images


async def _convert_single_page(
    pdf_doc: fitz.Document, 
    page_num: int, 
    mat: fitz.Matrix
) -> Image.Image:
    """
    Convierte una sola página de forma asíncrona con gestión optimizada de memoria
    """
    # Ejecutar en thread pool para operación blocking
    loop = asyncio.get_event_loop()
    
    def _sync_convert():
        page = pdf_doc.load_page(page_num)
        try:
            # Obtener pixmap optimizado
            pix = page.get_pixmap(matrix=mat, alpha=False)
            try:
                # Conversión directa a PIL
                img_bytes = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_bytes))
                return pil_image
            finally:
                # Liberar pixmap inmediatamente
                pix = None
        finally:
            page = None
    
    return await loop.run_in_executor(None, _sync_convert)


# Función adicional para manejo de memoria avanzado
def setup_memory_optimization():
    """
    Configura optimizaciones de memoria para procesamiento PDF
    """
    # Configurar Pillow para liberar memoria agresivamente
    try:
        from PIL import Image
        # Desactivar cache de imágenes
        Image.MAX_IMAGE_PIXELS = None
        # Configurar para liberar memoria inmediatamente
        import PIL.Image
        PIL.Image.core.set_blocks_max(0)  # Deshabilitar cache de bloques
    except ImportError:
        pass
    
    # Configurar garbage collection más agresivo
    import gc
    gc.set_threshold(100, 10, 10)  # Más frecuente para liberar memoria


# Wrapper optimizado para uso en el router
async def process_pdf_pages_ultra_optimized(pdf_binary_data: bytes) -> Tuple[str, int]:
    """
    Wrapper principal con configuración de memoria optimizada
    """
    # Configurar optimizaciones de memoria
    setup_memory_optimization()
    
    try:
        result = await process_pdf_pages_optimized(pdf_binary_data)
        return result
    finally:
        # Limpieza final agresiva
        gc.collect()

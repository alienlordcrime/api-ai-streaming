from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

import asyncio
import logging

class SuryaPredictor:
    _instance = None
    _lock = asyncio.Lock()
    _recognition_predictor = None
    _detection_predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_predictors(self):
        async with self._lock:
            if self._recognition_predictor is None or self._detection_predictor is None:
                # Inicializar predictores una sola vez
                self._recognition_predictor = RecognitionPredictor()
                self._detection_predictor = DetectionPredictor()
                logging.info("Surya predictores inicializados")
        return self._recognition_predictor, self._detection_predictor

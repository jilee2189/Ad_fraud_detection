from adfrauddetection.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

from adfrauddetection.exception.exception import AdfrauddetectionException
from adfrauddetection.logging.logger import logging

class NetworkModel:
    def __init__(self,model):
        try:
            self.model = model
        except Exception as e:
            raise AdfrauddetectionException(e,sys)
    
    def predict(self,x):
        try:
            y_hat = self.model.predict(x)
            return y_hat
        except Exception as e:
            raise AdfrauddetectionException(e,sys)
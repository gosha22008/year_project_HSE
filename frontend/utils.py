import base64
from io import BytesIO

import streamlit as st
import pandas as pd
import requests
from PIL import Image

from logger import log_error, log_info
from config1 import FASTAPI_URL




def get_available_models():
    try:
        response = requests.get(f"{FASTAPI_URL}/list_models")
        if response.status_code == 200:
            log_info("Список доступных моделей получен.")
            return response.json()
        else:
            st.error(f"Ошибка при получении списка моделей: {response.text}")
            log_error(f"Ошибка при получении списка моделей: {response.text}")
            return []
    except Exception as e:
        st.error(f"Ошибка при взаимодействии с сервером: {e}")
        log_error(f"Ошибка при взаимодействии с сервером: {e}")
        return []


def train_model_with_api(config, file):
    """Вызов API для обучения модели."""
    with open(file, "rb") as f:
        files = {'file': f}
        response = requests.post(f"{FASTAPI_URL}/fit_new_model", data=config, files=files)
        
    return response


def make_prediction(model_name, df):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/model_predict",
            files={"file": df.to_csv(index=False).encode()},
            data={"model_name": model_name}
        )
        if response.status_code == 200:
            predictions = response.json()
            st.success("Предсказания успешно получены!")
            log_info("Предсказания успешно получены.")
            return pd.DataFrame(predictions)
        else:
            st.error(f"Ошибка при получении предсказаний: {response.text}")
            log_error(f"Ошибка при получении предсказаний: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ошибка при взаимодействии с сервером: {e}")
        log_error(f"Ошибка при взаимодействии с сервером: {e}")
        return None
    





def predict_model_with_api( file):
    """Вызов API для обучения модели."""
    with open(file, "rb") as f:
        files = {'file': f}
        response = requests.post(f"{FASTAPI_URL}/model_predict",  files=files)
        
    return response

import shutil
from preprocessing_data import preprocessing_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from datetime import datetime
import joblib
import os
from fastapi import HTTPException
import logging

#настройка логирования
logger = logging.getLogger(__name__)

saved_models_path = Path("saved_models/") # путь до директории saved_models

target = "is_fraud" # целевая переменная


def copy_best_model():
    '''
    При запуске сервера лучшая предобученная модель сохраняется в папку saved_models
    для быстрого доступа к методу предсказания.
    '''
    # Задать пути
    source_path = Path('best_pre-trained_model/best_pre-trained_model')  # Путь к исходному файлу
    # Копируем лучшую модель в сохранённе модели
    if os.path.exists("best_pre-trained_model/best_pre-trained_model"):
        shutil.copy(source_path, saved_models_path)
        logging.info("Предобученная модель успешно скопирована")


def train_model(data, config):
    '''
    Обучение модели LogisticRegression с поданными параметрами на загруженных данных,
    предвариетльно обработав их функцией препроцессингом. Сохранение обученной модели.
    '''
    try:
        # Преобразуем данные
        y = data[target]
        df_processed = preprocessing_data(data)
        logging.info("Данные учпешно обработаны препроцессингом")

        X = df_processed
        

        # Разделяем данные на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Разделение данных на треин и тест")


        if not config['class_weight']:
            config['class_weight'] = None

        # Инициализируем модель
        model = LogisticRegression(**config)
        logging.info("Инициализация модели с параметрами")

        # Обучаем модель
        model.fit(X_train, y_train)
        logging.info("Обучение модели")

        model_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_model(model, model_name)
        return model_name
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обучения модели: {e}")


def save_model(model, model_name):
    '''
    Сохранение модели на сервере (в директории saved_models)
    '''
    try:
        joblib.dump(model, os.path.join(saved_models_path,model_name))
        logging.info(f"Модель - model_name успешно сохранена")
    except Exception as e:
        logging.error(f"Ошибка сохранения модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения модели: {e}")
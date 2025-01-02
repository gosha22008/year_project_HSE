from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
from http import HTTPStatus
from typing import Dict, List, Optional, Union
import joblib
from pathlib import Path
import os
import pandas as pd
import io
from preprocessing_data import preprocessing_data
from sub_functions import copy_best_model, train_model
import logging
from log_config import set_logging


app = FastAPI(
    # docs_url="/api/openapi",
    # openapi_url="/api/openapi.json",
)


# Настройка логирования
set_logging()
logger = logging.getLogger(__name__)

# настройка активной модели по умолчанию
active_model = {} # словарь для активной модели 
if os.path.exists('best_pre-trained_model/best_pre-trained_model'):
    active_model["active_model"] = os.listdir('best_pre-trained_model')[0] # активная модель по умолчанию
    logging.info("Предобученная модель установлена активной")

saved_pipelines = Path("saved_pipelines/") # препроцессинг данных для модели

saved_models_path = Path("saved_models/") # path to saved models
if not os.path.exists(saved_models_path): # if not exist then make dir saved_models
    Path.mkdir("saved_models")
    logging.info("Создана директория saved_models")

copy_best_model() # копируем предобученную модель в saved_models

# Pydantic модели для валидации данных
class FitNewModelConfig(BaseModel):
    max_iter: int = Field(default=1000, gt=0, description="Максимальное количество итераций")
    C: float = Field(default=1, gt=0, description="Регуляризационный параметр")
    solver: str = Field(default="lbfgs", description='решатель')
    penalty: str = Field(default="l2", description="регуляризация")
    class_weight: Optional[Union[str, Dict[str, float]]] = Field(default=None, description="баланс классов")

class ApiResponse(BaseModel):
    message: str

class IdRequest(BaseModel):
    id: str

class IdResponse(BaseModel):
    id: str    

class PredictRequest(BaseModel):
    id: str
    X: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[float]

class ListIdResponse(BaseModel):
    models: List[IdResponse]

class DeleteResponse(BaseModel):
    message: str

class ActiveModelResponse(BaseModel):
    model: Union[str, None]

class FitResponse(BaseModel):
    status: str
    config: FitNewModelConfig
    model_name: str


async def get_config(
    max_iter: int = Form(1000, gt=0, description="Максимальное количество итераций"),
    C: float = Form(1.0, gt=0, description="Регуляризационный параметр"),
    solver: str = Form(default="lbfgs", description='решатель'),
    penalty: str = Form(default="l2", description="регуляризация"),
    class_weight: Optional[Union[str, Dict[str, float]]] = Form(default="balanced", description="баланс классов")
) -> FitNewModelConfig:
    '''
    Функция зависимость позволяющая использовать Pydantic модели для данных,
    поступающих через Form
    '''
    return FitNewModelConfig(
        max_iter=max_iter,
        C=C,
        solver=solver,
        penalty=penalty,
        class_weight=class_weight)

    
# API методы
@app.post("/fit_new_model", response_model=FitResponse, status_code=HTTPStatus.CREATED)
async def fit(
    config: FitNewModelConfig = Depends(get_config),
    file: Optional[UploadFile] = File(None)
):
    '''
    Обучение модели LogisticRegression на загруженном файле csv и сохранение обученной модели
    '''
    try:
        df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
        logging.info(f"Данные прочитаны из csv файла")
    except Exception as e:
        logging.error(f"Ошибка чтения файла {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла {e}")
    
    model_name = train_model(df, config.dict())
    logging.info(f"Модель {model_name} успешно обучена и сохранена")
    
    return FitResponse(
        status='OKEI',
        config=config,
        model_name=model_name
    )


@app.post("/set_active_model", response_model=ApiResponse, status_code=HTTPStatus.OK)
async def set_active_model(
    request:IdRequest
):
    '''
    Установка модели в статус активная по id(имени) модели
    '''
    saved_models = {f"{model}": model for model in os.listdir(saved_models_path)}
    if saved_models.get(request.id):
        active_model["active_model"] = request.id
        logging.info(f"Модель с {request} установлена активной")
        return ApiResponse(message=f"Модель с {request} установлена активной")
    else:
        logging.error(f"Ошибка. Такой {request} модели нет на сервере")
        raise HTTPException(status_code=404, detail=f"Ошибка. Такой {request} модели нет на сервере")


@app.get("/list_models", response_model=List[ListIdResponse])
async def get_list_models():
    '''
    Получение списка всех моделей на сервере (хранящихся в директории saved_models)
    '''
    response_list = []
    saved_models = os.listdir(saved_models_path)
    for model in saved_models:
        response_list.append(IdResponse(id=model))
    logging.info(f"Получение списка модели")
    return  [ListIdResponse(models=response_list)]


@app.get("/active_model", response_model=ActiveModelResponse)
async def get_active_model():
    '''
    Проверка установлена ли активная модель и какая именно модель установлена
    '''
    model_name = active_model.get("active_model")
    if model_name:
        logging.info(f"Активная модель- {model_name}")
        return ActiveModelResponse(model=model_name)
    else:
        logging.warning(f"Активная модель не задана")
        return ActiveModelResponse(model="Модель не задана")


@app.delete("/remove_all_models", response_model=List[DeleteResponse])
async def remove_all():
    '''
    Удаление всех моделей с сервера (из директории saved_models). Очистка активной модели.
    '''
    response_list = []
    saved_models = os.listdir(saved_models_path)
    for model in saved_models:
        try:
            os.remove(os.path.join(saved_models_path, model))
            logging.info(f"Model id: {model} removed")
            response_list.append(DeleteResponse(message=f"Model id: {model} removed"))
        except Exception as e:
            logging.error(f"Ошибка в удалении модели id: {model} - {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка в удалении модели id: {model} - {e}")
    global active_model
    active_model = {}
    return response_list


@app.delete("/remove_model", response_model=DeleteResponse)
async def remove_model(
    request: IdRequest
):
    '''
    Удаление модели по её id (имени) с сервера (из директории saved_models).
    Если модель была активной, то очистка активной модели.
    '''
    global active_model
    saved_models = os.listdir(saved_models_path)
    for model in saved_models:
        try:
            if active_model.get("active_model") == request.id:
                active_model = {}
                logging.info(f"Обнуление активной модели")
            if request.id == model:
                os.remove(os.path.join(saved_models_path, model))
                logging.info(f"Удаление модели {model} из директории saved_models")
        except Exception as e:
            logging.error(f"Ошибка в удалении модели id: {model} - {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка в удалении модели id: {model} - {e}")
    return DeleteResponse(message=f"Модель {request.id} успешно удалена")


@app.post("/model_predict", response_model=PredictResponse)
async def model_predict(
    file: Optional[UploadFile] = File(None)
):
    '''
    Предсказание активной модели. Подаётся файл csv формата.
    Данные обрабатываются препроцессиногом данных.
    Активная модель делает предсказания на обработанных данных.
    '''
    try:
        df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
        logging.info("Данные успешно считаны из файла")
    except Exception as e:
        logging.error(f"Ошибка чтения файла {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения файла {e}")
    
    if active_model.get("active_model"):
        model_name = active_model.get("active_model") 
    else:
        logging.warning(f"Активная модель не задана, установите!")
        raise HTTPException(status_code=404, detail=f"Активная модель не задана, установите!")
    
    data = preprocessing_data(df)

    try:
        model = joblib.load(f"saved_models/{model_name}")
        logging.info(f"Модель успешно загружена")
    except Exception as e:
        logging.error(f"модель не найдена {e}")
        raise HTTPException(status_code=404, detail=f"модель не найдена {e}")
    
    try:
        prediction = model.predict(data)
        logging.info(f"Успешное предсказание модели - {model_name}")
    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обучения модели: {e}")
    
    return PredictResponse(predictions=prediction)

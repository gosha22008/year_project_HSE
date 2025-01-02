import base64
from preprocessing import preprocess_data, get_plots
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import base64
import io
import logging
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import tempfile
import os
import requests
from config1 import FASTAPI_URL

from utils import  train_model_with_api, predict_model_with_api, get_available_models

st.title("ML Модель: Обучение и Предсказания")
st.write("Используйте это приложение для загрузки данных, обучения модели, получения метрик и выполнения предсказаний.")

# Загрузка данных или выбор данных сервера
st.header("Загрузка данных")

model_info = None
df = None


st.title("Загрузка CSV файла с данными")

# Элемент для загрузки файла
uploaded_file = st.file_uploader("Загрузите CSV файл с данными:", type="csv")

# Проверяем, был ли загружен файл
if uploaded_file is not None:
    # Читаем CSV файл
    df = pd.read_csv(uploaded_file)  # Можно использовать uploaded_file напрямую
    st.write(df)  # Отобразим загруженный DataFrame

    # Применяем функцию предобработки к загруженным данным
    try:
        df = preprocess_data(df)  # Предполагается, что эта функция определена
        st.write("Данные после предобработки:")
        st.write(df)
    except ValueError as e:
        st.error(f"Произошла ошибка при предобработке данных: {e}")
else:
    st.write("Пожалуйста, загрузите файл.")
# EDA
if uploaded_file:
    # Добавляем кнопку для выполнения EDA
    if st.button("Выполнить EDA"):
        st.write("Отправка данных на сервер для анализа...")
        plots = get_plots(df)

        if plots:
            st.header("Результаты EDA")

            if "distribution" in plots:
                st.subheader("Распределение признака")
                st.image(base64.b64decode(plots["distribution"]))

            if "correlation_matrix" in plots:
                st.subheader("Матрица корреляции")
                st.image(base64.b64decode(plots["correlation_matrix"]))

# Обучение модели
if uploaded_file: # or use_server_data:
    # Показываем "ручки управления" и кнопку для обучения модели
    st.header("Настройка параметров модели")
    max_iter = st.number_input("Максимальное количество итераций:", min_value=100, max_value=5000, value=1000)
    C = st.number_input("Параметр регуляризации (C):", min_value=0.01, max_value=10.0, value=1.0)
    solver = st.text_input("Solver", value="lbfgs")
    penalty = st.selectbox("Penalty", options=["l2", "none"])
    class_weight = st.text_input("Class Weight (e.g., 'balanced' or '{0: 0.5, 1: 0.5}')")  # Используйте text_input для class_weight


    st.header("Обучение модели")

    if st.button("Train Model"):
        if uploaded_file is not None:
            # Сохранение загруженного файла во временный файл
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_file_path = tmp_file.name
        
            # Подготовка данных для передачи в API
            config = {
                "max_iter": max_iter,
                "C": C,
                "solver": solver,
                "penalty": penalty,
                "class_weight": class_weight
            }
        
            # Вызов API для обучения модели
            response = train_model_with_api(config, temp_file_path)
            # Предполагаем, что response - это объект Response
            if response.ok:  # Проверка успешного ответа
                response_data = response.json()  # Получение данных в формате JSON
                model_name = response_data.get('moel_name')  # Получение значения по ключу 'moel_name'

                if model_name:
                    st.write(model_name)
                else:
                    st.write("Ключ 'moel_name' отсутствует в ответе.")
        
        # Удаление временного файла после использования
        os.remove(temp_file_path)


st.header("Выбор модели для предсказаний")

if st.button("Показать список моделей"):
    response = get_available_models()
    if response:
        # Извлечение списка моделей из ответа
        models = response[0]['models']  # Предполагая, что models - это список словарей
    
        # Извлекаем имена моделей
        model_names = [model['id'] for model in models]  # Предполагаем, что id - это имя модели

        # Создаем DataFrame для отображения в виде столбца
        df = pd.DataFrame(model_names, columns=["Имена моделей"])

        # Отображаем таблицу в Streamlit
        st.dataframe(df)
    else:
        st.write("Нет доступных моделей.")


st.header("Предсказания")
# Функция для установки активной модели


# Определение функции для установки активной модели
def set_active_model(model_id):
    response = requests.post(f"{FASTAPI_URL}/set_active_model", json={"id": model_id})
    if response.status_code == 200:
        return response.json().get('message', "Модель успешно установлена.")
    else:
        return f"Ошибка: {response.text}"

# Основной поток приложения Streamlit
st.title("Установка активной модели")

# Поле для ввода идентификатора модели
model_id = st.text_input("Введите идентификатор (имя) модели:")

# Кнопка для установки активной модели
if st.button("Установить активную модель"):
    if model_id:
        message = set_active_model(model_id)
        st.success(message)
    else:
        st.error("Пожалуйста, введите идентификатор модели.")


def get_active_model():
    response = requests.get(f"{FASTAPI_URL}/active_model")
    if response.status_code == 200:
        return response.json().get('model', "Неизвестная модель.")
    else:
        return f"Ошибка: {response.text}"

# Основной поток приложения Streamlit
st.title("Проверка активной модели")

# Кнопка для получения активной модели
if st.button("Проверить активную модель"):
    model = get_active_model()
    st.write("Активная модель:", model)

# Элемент для загрузки файла
uploaded_file_predict = st.file_uploader("Загрузите CSV файл с данными для предсказания:", type="csv")

if uploaded_file_predict is not None and st.button("Сделать предсказания"):
    # Сохранение загруженного файла во временный файл
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file_predict.getbuffer())
        temp_file_path = tmp_file.name
        
        
    # Вызов API для обучения модели
    response = predict_model_with_api( temp_file_path)
    # Предполагаем, что response - это объект Response
    if response.ok:  # Проверка успешного ответа
        response_data = response.json()  # Получение данных в формате JSON
        redictions = response.json().get('predictions', [])  # Получение значения по ключу 'moel_name'
        st.success("Предсказания получены!")

        if redictions:
            st.write(redictions)
        else:
            st.write("Предсказаний нет.")
        
    # Удаление временного файла после использования
    os.remove(temp_file_path)



# year_project_HSE
# Детектирование аномалий в данных

**Для тех, кто работал с реальными данными, не является секретом тот факт, что для построения хороших моделей данные требуются тщательно предобработать. В том числе избавиться от аномалий (лучше, но сложнее – их объяснить). Выдуманный банк хотел бы по транзакциям клиентов определять, является ли конкретная транзакция (или группа транзакций) подозрительной(-ными).**

---

*куратор* - Малюшитский Кирилл (@malyushitsky)

*участники проекта*:
- Егошин Юрий (@Yuri_Dmitrievich), (https://github.com/gosha22008)
- Глеб Лысенко (@glebly), (https://github.com/Gleb22001)
- Силиневич Илья (@uJlbI0IILuH), (https://github.com/kris01091980)
- Цыбакова Ольга (@olgasub57), (https://github.com/Olga57)

## ! Работа с моделями по 5 чекпоинту выложена здесь [backend/checkpoint5.ipynb](backend/checkpoint5.ipynb)

## Документация
- **! Презентация по 5 чекпоинту** выложена здесь [Year_project_presentation_5check_16032025.pdf](Year_project_presentation_5check_16032025.pdf), а также доступна в  [GoogleDocs](https://docs.google.com/presentation/d/1ravblR6tPc6wHUvsqJksDdq7RVtF4rdOXzkjq0BEJuo/edit?usp=sharing)
- Описание структуры проекта и инструкция по использованию сервиса описана в [report.pdf](report.pdf)

 
# Установка и развертывание

## I. Установка и запуск приложения

### Требования:
- Установленный **Docker** и **Docker Compose**.
- Операционная система с поддержкой контейнеризации (Linux, macOS, Windows).

Для установки необходимо ввести следующие команды в системе (удаленном сервере для развертывания):

```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo apt-get install docker-compose-plugin
```

### Шаги:

1. **Клонирование репозитория:**
   Скачайте проект и перейдите в папку:
   ```bash
   cd ml-api
   ```

2. **Сборка и запуск контейнеров:**   
   В файле `config.py`, находящимся в папке frontend указать действующий ip-адрес сервера

   Выполните команду:
   ```bash
   docker compose up --build
   ```
   Это запустит два сервиса:
   - **backend (FastAPI)** на порту `8000`.
   - **frontend (Streamlit)** на порту `8501`.

3. **Проверка статуса:**
   Убедитесь, что сервисы работают:
   - Перейдите в браузере на `http://localhost:8000/docs` для взаимодействия с API.
   - Перейдите на `http://localhost:8501` для работы с интерфейсом **Streamlit**.
   
   В случае развертывания на удаленном сервере нужно использовать соответственно `http://<ip-адрес>:8000/docs` и `http://<ip-адрес>:8501` 
---

## II. Использование сервиса

1. **API через FastAPI (backend)**:

   Необходимо перейти по адресу `http://<ip-адрес>:8000/docs` для изучения актуального API.
   
---
2. **Frontend через Streamlit:**
3. 
1. Перейдите на `http://<ip-адрес>:8501` для использования интерактивного интерфейса.
2. В интерфейсе доступны:
   - Загрузка данных для EDA и его визуализации. Действие происходит автоматически при загрузке данных.
   - Инструменты для запуска обучения модели и предсказаний.
   - Установка активной модели. Это означает, что для работы будет использована выбранная модель.

---

## III. Остановка сервисов
Для остановки всех контейнеров:
```bash
docker compose down
```

---



### Примечания:
- **Логи** приложений сохраняются в папке `frontend/logs` для Streamlit.
- **Данные и модели** хранятся в папке `backend/data`.

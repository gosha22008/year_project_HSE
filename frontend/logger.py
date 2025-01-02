import os
import logging
from logging.handlers import RotatingFileHandler

# Настройка директории для логов
from config1 import LOG_DIR

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Путь к файлу логов
log_file = os.path.join(LOG_DIR, "app.log")

# Создание обработчика логирования с ротацией
handler = RotatingFileHandler(
    log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # Максимальный размер 10MB, до 5 архивных файлов
)
handler.setLevel(logging.INFO)

# Настройка формата логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Добавление обработчика в корневой логгер
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)


def log_error(message):
    """
    Логирует сообщение уровня ERROR
    message: текст сообщения
    """
    logging.error(message)


def log_info(message):
    """
    Логирует сообщение уровня INFO
    message: текст сообщения
    """
    logging.info(message)

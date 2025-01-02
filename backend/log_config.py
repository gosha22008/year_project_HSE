import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def set_logging():
    # Создаем директорию logs, если она не существует
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Получаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Создаем обработчик для записи в файл с ротацией
    file_handler = RotatingFileHandler(
        logs_dir / "app.log",
        maxBytes=1_000_000, 
        backupCount=3,
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

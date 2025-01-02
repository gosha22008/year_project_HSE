import pandas as pd
import numpy as np
import joblib
from datetime import time


def preprocessing_data(data):
    '''
    Функция обработки данных для подачи в модель.
    '''
    try:
        data['name'] = data['first'] + " " +  data['last'] # создадим столбец name вместо двух столбцов first и last
    except Exception as e:
        raise ValueError(f"Ошибка при создании колонки name: {e}")

    try:
        # удалим столбцы Unnamed: 0, first,  last
        # Unnamed: 0 - повторение индекса
        # first и last - имя владельца считаю что ничего нам полезного не даст
        # trans_num - номер транзакции уникальный, тоже ничего не даст
        # unix_time - дублирует(наверное должен) колонку trans_date_trans_time
        data = data.drop(['Unnamed: 0', 'trans_num', 'first', 'last', 'unix_time'], axis=1) # 'first', 'last'
        # data.head(7)
    except Exception as e:
        raise ValueError(f"Ошибка при удалении колонок ['Unnamed: 0', 'trans_num', 'first', 'last', 'unix_time']: {e}")

    try:
        # переведём столбец trans_date_trans_time в datetime
        data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S')
        # data['trans_date_trans_time'].info()
    except Exception as e:
        raise ValueError(f"Ошибка при переводе в формат datetime: {e}")

    try:
        # разделим общую датувремя на год месяц день и отдельно время
        data['trans_year'] = data['trans_date_trans_time'].dt.year
        data['trans_month'] = data['trans_date_trans_time'].dt.month
        data['trans_day'] = data['trans_date_trans_time'].dt.day
        data['trans_time'] = data['trans_date_trans_time'].dt.time
        data = data.drop('trans_date_trans_time', axis=1) # удалим столбец с датой
    except Exception as e:
        raise ValueError(f"Ошибка при разделении на год месяц день: {e}")

    try:
        data.fillna(-1, inplace=True) # пока что заполняем пропущенные значения -1.
    except Exception as e:
        raise ValueError(f"Ошибка при заполнении пустых значений: {e}")

    try:
        # Рассчитываем возраст
        data['dob'] = pd.to_datetime(data['dob'])
        current_year = 2024
        data['age'] = current_year - data['dob'].dt.year
    except Exception as e:
        raise ValueError(f"Ошибка при расчете возраста клиента: {e}")

    try:
        # добавим колонки: morning, midday, evening, night - часть дня когда была сделана транзакция
        data['morning'] = np.where((time(6,0,0) <= data['trans_time']) & (data['trans_time'] <= time(12,0,0)), 1, 0)
        data['midday'] = np.where((time(12,0,0) < data['trans_time']) & (data['trans_time'] <= time(18,0,0)), 1, 0)
        data['evening'] = np.where((time(18,0,0) < data['trans_time']) & (data['trans_time'] <= time(23,59,59)), 1, 0)
        data['night'] = np.where((time(0,0,0) <= data['trans_time']) & (data['trans_time'] < time(6,0,0)), 1, 0)
    except Exception as e:
        raise ValueError(f"Ошибка при добавлении колонок morning, midday, evening, night: {e}")

    try:
        data['hour'] = data['trans_time'].apply(lambda x: x.hour)
    except Exception as e:
        raise ValueError(f"Ошибка в создании колонки hour: {e}")
    
    try:
        # Подсчет повторений для каждого значения в столбце 'name'
        count_series = data['name'].value_counts()
        # Создание нового столбца с количеством повторений
        data['name_count'] = data['name'].map(count_series)
    except Exception as e:
        raise ValueError(f"Ошибка в создании колонки name_count: {e}")

    try:
        preprocess = joblib.load("saved_pipelines/preprocess")
        data = preprocess.transform(data)
    except Exception as e:
        raise ValueError(f"Ошибка в загрузке pipeline или в обработке данных: {e}")
    
    try:
        data = pd.DataFrame(data, columns=preprocess.get_feature_names_out())
    except Exception as e:
        raise ValueError(f"Ошибка в создании выходного датафрейма: {e}")
    
    return data

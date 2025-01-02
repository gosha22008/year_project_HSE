from sklearn.preprocessing import LabelEncoder
import pandas as pd
import base64
import io
import logging
import seaborn as sns
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Функция для предобработки данных.
    Принимает сырые данные и возвращает предобработанные данные.
    """
    try:
        # Преобразование даты и времени
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['trans_year'] = df['trans_date_trans_time'].dt.year
        df['trans_month'] = df['trans_date_trans_time'].dt.month
        df['trans_day'] = df['trans_date_trans_time'].dt.day
        df['trans_hour'] = df['trans_date_trans_time'].dt.hour
        df['trans_minute'] = df['trans_date_trans_time'].dt.minute
        df['trans_second'] = df['trans_date_trans_time'].dt.second
        df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday
        df = df.drop(columns=['trans_date_trans_time'])
    except Exception as e:
        raise ValueError(f"Ошибка при преобразовании даты и времени: {e}")

    try:
        # Удаление ненужных столбцов
        columns_to_drop = [
            'Unnamed: 0', 'first', 'last', 'street', 'city', 'zip',
            'trans_num', 'merch_zipcode', 'cc_num', 'merchant', 'job'
        ]
        df = df.drop(columns=columns_to_drop, errors='ignore')
    except Exception as e:
        raise ValueError(f"Ошибка при удалении столбцов: {e}")

    try:
        # Кодирование категориальных переменных
        cat_features = ['category', 'gender', 'state']
        for col in cat_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    except Exception as e:
        raise ValueError(f"Ошибка при кодировании категориальных переменных: {e}")

    try:
        # Добавление признака "выходной день"
        df['is_weekend'] = df['trans_weekday'].apply(lambda x: 1 if x >= 5 else 0)
    except Exception as e:
        raise ValueError(f"Ошибка при добавлении признака 'выходной день': {e}")

    try:
        # Добавление признака "ночное время"
        df['is_night'] = df['trans_hour'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    except Exception as e:
        raise ValueError(f"Ошибка при добавлении признака 'ночное время': {e}")

    try:
        # Расчет возраста клиента
        df['dob'] = pd.to_datetime(df['dob'])
        df['birth_year'] = df['dob'].dt.year
        df['card_holder_age'] = df['trans_year'] - df['birth_year']
        df = df.drop(columns=['dob', 'birth_year'])
    except Exception as e:
        raise ValueError(f"Ошибка при расчете возраста клиента: {e}")

    try:
        # Удаление выбросов (опционально)
        outlier_threshold = 2700
        df = df[df['amt'] <= outlier_threshold]
    except Exception as e:
        raise ValueError(f"Ошибка при удалении выбросов: {e}")

    return df


def get_plots(df):
    """
    Генерирует графики на основе загруженного CSV файла.
    Возвращает словарь с графиками в формате base64.
    """
    plots = {}

    try:
        # Выбираем только числовые столбцы
        numeric_columns = df.select_dtypes(include=["float", "int"]).columns
        if numeric_columns.empty:
            logger.warning("Нет числовых столбцов для анализа")
            return {"plots": {}, "message": "Нет числовых столбцов для анализа"}

        # График распределения для первого числового столбца
        try:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.histplot(
                df['trans_month'],
                kde=True,
                ax=ax,
                color="skyblue",
                edgecolor="black",
                linewidth=1.5,
            )
            ax.set_title(
                f"Распределение: trans_month",
                fontsize=16,
                fontweight="bold",
                color="darkblue",
            )
            ax.set_xlabel(
                'Номер месяца',
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylabel(
                "Частота",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, linestyle="--", alpha=0.7)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plots["distribution"] = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
            logger.info(f"График распределения для {numeric_columns[0]} успешно создан")
        except Exception as e:
            logger.error(f"Ошибка при создании графика распределения: {e}")

        # Матрица корреляции
        try:
            if len(numeric_columns) > 1:
                relevant_columns = [col for col in numeric_columns if not col.lower().startswith("unnamed")]
                if relevant_columns:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    corr = df[relevant_columns].corr()
                    sns.heatmap(
                        corr,
                        annot=True,
                        fmt=".2f",
                        cmap="coolwarm",
                        annot_kws={"size": 10, "fontweight": "bold"},
                        cbar_kws={"shrink": 0.8, "aspect": 30},
                        linewidths=0.5,
                        square=True,
                        ax=ax,
                    )
                    ax.set_title(
                        "Матрица корреляции",
                        fontsize=18,
                        fontweight="bold",
                        color="darkblue",
                        pad=20,
                    )
                    ax.tick_params(axis="x", labelsize=12, rotation=45)
                    ax.tick_params(axis="y", labelsize=12)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight")
                    buf.seek(0)
                    plots["correlation_matrix"] = base64.b64encode(buf.read()).decode("utf-8")
                    plt.close()
                    logger.info("Матрица корреляции успешно создана")
        except Exception as e:
            logger.error(f"Ошибка при создании матрицы корреляции: {e}")

    except Exception as e:
        logger.error(f"Ошибка при получении графиков: {e}")
        return plots

    return plots


def perform_eda(file):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/eda",
            files={"file": file.getvalue()}
        )
        if response.status_code == 200:
            plots = response.json().get("plots", {})
            return plots
        else:
            st.error(f"Ошибка при выполнении EDA: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Ошибка при взаимодействии с сервером: {e}")
        return {}

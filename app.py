from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from transformers import pipeline
from database import init_db, save_news, get_news_from_db

# Загружаем переменные окружения
load_dotenv()

app = FastAPI()

# Инициализация базы данных
init_db()

# Модель для запроса данных
class ForecastRequest(BaseModel):
    query: str
    region: str = None

# Инициализация модели для анализа тональности
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# Функция для получения новостей
def get_news(query, region=None):
    api_key = os.getenv("NEWS_API_KEY")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "ru",
        "sortBy": "publishedAt",
        "pageSize": 100
    }
    if region:
        params["region"] = region
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Ошибка при запросе новостей")
    return response.json()

# Функция для анализа новостей
def analyze_news(news_data):
    # Преобразуем данные в DataFrame
    articles = news_data['articles']
    df = pd.DataFrame(articles)
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Анализ тональности каждой статьи
    df['sentiment'] = df['title'].apply(lambda x: sentiment_analyzer(x[:512])[0]['label'])
    
    # Сохраняем новости в базу данных
    save_news(articles)
    
    # Подсчет количества положительных, отрицательных и нейтральных статей
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    
    # Прогнозирование трендов (пока просто линейная регрессия)
    df['days'] = (df['publishedAt'] - df['publishedAt'].min()).dt.days
    X = df[['days']]
    y = np.arange(len(df))
    model = LinearRegression()
    model.fit(X, y)
    trend = model.coef_[0]
    
    return sentiment_counts, trend

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    news_data = get_news(request.query, request.region)
    sentiment_counts, trend = analyze_news(news_data)
    
    # Генерация прогноза
    if trend > 0:
        forecast_text = "Тренд положительный. Вероятно, ситуация будет улучшаться."
    else:
        forecast_text = "Тренд отрицательный. Вероятно, ситуация будет ухудшаться."
    
    # Генерация инструкции на основе тональности
    positive_count = sentiment_counts.get('POSITIVE', 0)
    negative_count = sentiment_counts.get('NEGATIVE', 0)
    
    if positive_count > negative_count:
        instruction = "Рекомендуется продолжить текущую политику."
    else:
        instruction = "Рекомендуется пересмотреть текущую политику."
    
    # Генерация графика
    plt.figure(figsize=(10, 5))
    plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['green', 'red', 'blue'])
    plt.title('Распределение тональности новостей')
    plt.xlabel('Тональность')
    plt.ylabel('Количество статей')
    plt.savefig('static/sentiment_plot.png')
    
    return {
        "forecast": forecast_text,
        "instruction": instruction,
        "sentiment_counts": sentiment_counts,
        "trend": trend,
        "plot_url": "/static/sentiment_plot.png"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

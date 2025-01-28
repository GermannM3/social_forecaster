import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('social_forecaster.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS news
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT,
                  content TEXT,
                  publishedAt TEXT,
                  sentiment TEXT)''')
    conn.commit()
    conn.close()

def save_news(articles):
    conn = sqlite3.connect('social_forecaster.db')
    c = conn.cursor()
    for article in articles:
        c.execute("INSERT INTO news (title, content, publishedAt, sentiment) VALUES (?, ?, ?, ?)",
                  (article['title'], article['content'], article['publishedAt'], article['sentiment']))
    conn.commit()
    conn.close()

def get_news_from_db():
    conn = sqlite3.connect('social_forecaster.db')
    c = conn.cursor()
    c.execute("SELECT * FROM news")
    rows = c.fetchall()
    conn.close()
    return rows

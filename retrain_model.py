import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def retrain_model():
    conn = sqlite3.connect('social_forecaster.db')
    df = pd.read_sql_query("SELECT * FROM news", conn)
    conn.close()
    
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['days'] = (df['publishedAt'] - df['publishedAt'].min()).dt.days
    X = df[['days']]
    y = np.arange(len(df))
    
    model = LinearRegression()
    model.fit(X, y)
    
    joblib.dump(model, 'models/linear_regression_model.pkl')

if __name__ == "__main__":
    retrain_model()

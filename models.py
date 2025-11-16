# backend/models.py
import sqlite3
import os
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = os.path.dirname(__file__)
DB = os.path.join(BASE, 'transactions.db')
MODEL_DIR = os.path.join(BASE, 'saved_models')
CAT_MODEL = os.path.join(MODEL_DIR, 'cat_model.pkl')
TFIDF = os.path.join(MODEL_DIR, 'tfidf.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

def get_conn():
    return sqlite3.connect(DB)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS transactions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT, amount REAL, description TEXT, category TEXT)''')
    conn.commit(); conn.close()

def add_transaction(date, amount, description, category):
    conn = get_conn(); c = conn.cursor()
    c.execute('INSERT INTO transactions(date,amount,description,category) VALUES(?,?,?,?)',
              (date,amount,description,category))
    conn.commit(); conn.close()

def get_transactions(limit=100):
    conn=get_conn(); c=conn.cursor()
    c.execute('SELECT id,date,amount,description,category FROM transactions ORDER BY date DESC LIMIT ?', (limit,))
    rows=c.fetchall(); conn.close()
    return [{'id':r[0],'date':r[1],'amount':r[2],'description':r[3],'category':r[4]} for r in rows]

def categorize_text(text):
    text = (text or '').lower()
    if os.path.exists(CAT_MODEL):
        try:
            tfidf = joblib.load(TFIDF)
            model = joblib.load(CAT_MODEL)
            return model.predict(tfidf.transform([text]))[0]
        except: pass
    rules={'groceries':['grocery','market'],'rent':['rent'],'transport':['uber','taxi'],
           'dining':['restaurant','cafe','coffee'],'entertainment':['netflix','movie'],
           'utilities':['electric','water','bill']}
    for cat,keys in rules.items():
        if any(k in text for k in keys): return cat
    return 'other'

def predict_expenses(months=3):
    conn=get_conn()
    df=pd.read_sql_query('SELECT date,amount FROM transactions',conn,parse_dates=['date'])
    conn.close()
    if df.empty:
        return [{'month':'N/A','pred':0}]*months
    df['ym']=df['date'].dt.to_period('M')
    m=df.groupby('ym')['amount'].sum().reset_index()
    m['idx']=range(len(m))
    X=m[['idx']].values; y=m['amount'].values
    if len(y)<2:
        last=float(y[-1])
        return [{'month':str(m['ym'].iloc[-1]+i+1),'pred':last} for i in range(months)]
    model=LinearRegression().fit(X,y)
    start=m['idx'].iloc[-1]+1
    return [{'month':str(m['ym'].iloc[-1]+i+1),
             'pred':float(model.predict([[start+i]])[0])} for i in range(months)]

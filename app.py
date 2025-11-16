# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from models import init_db, add_transaction, get_transactions, predict_expenses, categorize_text
import os

app = Flask(__name__)
CORS(app)
DB_PATH = os.path.join(os.path.dirname(__file__), 'transactions.db')

@app.route('/api/health')
def health():
    return jsonify({'status':'ok'})

@app.route('/api/transactions', methods=['GET','POST'])
def transactions():
    if request.method == 'POST':
        data = request.json
        if not all(k in data for k in ('date','amount','description')):
            return jsonify({'error':'missing fields'}),400
        category = categorize_text(data['description'])
        add_transaction(data['date'], float(data['amount']), data['description'], category)
        return jsonify({'status':'created','category':category}),201
    else:
        rows = get_transactions()
        return jsonify(rows)

@app.route('/api/predict', methods=['POST'])
def predict():
    body = request.json or {}
    months = int(body.get('months',3))
    forecast = predict_expenses(months)
    return jsonify({'forecast':forecast})

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

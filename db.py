import sqlite3
import datetime
import os

DB_PATH = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            model_used TEXT,
            prediction TEXT,
            confidence REAL,
            decision TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(filename, model_used, prediction, confidence, decision):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute('''
        INSERT INTO predictions (filename, model_used, prediction, confidence, decision, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, model_used, prediction, confidence, decision, timestamp))
    conn.commit()
    conn.close()

def get_history():
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 100')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

"""
TrueHire AI — Standalone Database Initializer
Run this once to create/reset the database.
Usage: python init_db.py
"""
import sqlite3, os

DB_FILE = 'truehire.db'

def init():
    print(f"\nInitializing database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)
    conn.execute('PRAGMA journal_mode=WAL')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        email      TEXT UNIQUE NOT NULL,
        password   TEXT NOT NULL,
        first_name TEXT,
        last_name  TEXT,
        role       TEXT DEFAULT 'jobseeker',
        resume_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS job_analysis (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        job_title        TEXT,
        company          TEXT,
        job_description  TEXT,
        risk_score       REAL,
        verdict          TEXT,
        signals          TEXT,
        fraud_probability REAL,
        model_version    TEXT DEFAULT '1.0',
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS job_recommendations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        job_title    TEXT,
        company      TEXT,
        location     TEXT,
        salary_range TEXT,
        match_score  REAL,
        verified     BOOLEAN,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    c.execute('''CREATE TABLE IF NOT EXISTS learning_paths (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id          INTEGER,
        target_role      TEXT,
        current_skills   TEXT,
        required_skills  TEXT,
        skill_gap        REAL,
        estimated_hours  INTEGER,
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    conn.commit()
    conn.close()

    tables = ['users', 'job_analysis', 'job_recommendations', 'learning_paths']
    for t in tables:
        print(f"  ✓ table: {t}")
    print(f"\n✅ Database ready: {os.path.abspath(DB_FILE)}\n")

if __name__ == '__main__':
    init()

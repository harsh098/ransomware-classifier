import sqlite3
import os

DB_NAME = os.getenv("LOGS_DB_NAME", "db.sqlite3")

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            TS INTEGER NOT NULL,
            PID INTEGER NOT NULL,
            TYPE TEXT NOT NULL,
            FLAG TEXT,
            PATTERN TEXT,
            \'OPEN\' INTEGER DEFAULT 0,
            \'CREATE\' INTEGER DEFAULT 0,
            \'DELETE\' INTEGER DEFAULT 0,
            ENCRYPT INTEGER DEFAULT 0,
            FILENAME TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

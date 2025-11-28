import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password_hash TEXT,
    created_at TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original TEXT,
    cleaned TEXT,
    prediction TEXT,
    confidence REAL,
    timestamp TEXT,
    user_id INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")

cursor.execute("PRAGMA table_info(history)")
columns = [col[1] for col in cursor.fetchall()]
if "user_id" not in columns:
    cursor.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")

conn.commit()
conn.close()

print("Database & tables created successfully!")

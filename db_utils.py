import os
import mysql.connector
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

MYSQL_CONFIG = {
    'host': os.environ.get("DB_HOST", "localhost"),
    'port': int(os.environ.get("DB_PORT", "3306")),
    'user': os.environ.get("DB_USER", "root"),
    'password': os.environ.get("DB_PASSWORD", ""),
    'database': os.environ.get("DB_NAME", "ecmschatbotdb")
}


@contextmanager
def get_db_connection():
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    try:
        yield conn
    finally:
        if conn.is_connected():
            conn.close()


def init_database():
    """Create tables if not exist."""
    create_knowledge = """
    CREATE TABLE IF NOT EXISTS ecms_knowledge (
        id INT AUTO_INCREMENT PRIMARY KEY,
        source VARCHAR(255) NOT NULL,
        section VARCHAR(100),
        chunk_text TEXT NOT NULL,
        embedding JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY unique_chunk (source, section, chunk_text(255))
    )
    """
    
    create_logs = """
    CREATE TABLE IF NOT EXISTS ecms_chatbot_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_input TEXT,
        response TEXT,
        timestamp DATETIME
    )
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(create_knowledge)
        cursor.execute(create_logs)
        conn.commit()
        cursor.close()
    print("✅ Database initialized.")
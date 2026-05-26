import json
import mysql.connector
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
INPUT_FILE = "ecms_content.json"

DB_CONFIG = {
    'host': os.environ.get("DB_HOST", "localhost"),
    'port': int(os.environ.get("DB_PORT", "3306")),
    'user': os.environ.get("DB_USER", ""),
    'password': os.environ.get("DB_PASSWORD", ""),
    'database': os.environ.get("DB_NAME", "ecmschatbotdb")
}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400,
    separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""]
)

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def init_database():
    create_knowledge = """
    CREATE TABLE IF NOT EXISTS ecms_knowledge (
        id INT AUTO_INCREMENT PRIMARY KEY,
        source VARCHAR(255) NOT NULL,
        section VARCHAR(100),
        chunk_text TEXT NOT NULL,
        embedding JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(create_knowledge)
    cursor.execute(create_logs)
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database initialized.")

def main():
    print("Loading content...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except FileNotFoundError:
        print(f" Error: {INPUT_FILE} not found. Run scrape.py first.")
        return

    all_chunks = []
    for source in sources:
        meta_header = f"SOURCE_ID: {source['source']}\nTITLE: {source['title']}\n\n"
        
        chunks = splitter.split_text(source['content'])
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            full_chunk = meta_header + chunk
            
            all_chunks.append({
                "source": source['source'],
                "section": source['title'],
                "chunk_index": i,
                "chunk_text": full_chunk
            })
    
    if not all_chunks:
        print("⚠️ No chunks created. Check your input file.")
        return

    print(f"Created {len(all_chunks)} chunks. Starting embedding...")
    
    embedder = OllamaEmbeddings(model=MODEL)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("TRUNCATE TABLE ecms_knowledge")
    conn.commit()
    
    batch_size = 5
    total = len(all_chunks)
    
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c['chunk_text'] for c in batch]
        
        print(f"  Processing Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
        try:
            vectors = embedder.embed_documents(texts)
            
            insert_data = []
            for chunk, vector in zip(batch, vectors):
                insert_data.append((
                    chunk['source'], 
                    chunk['section'], 
                    chunk['chunk_text'], 
                    json.dumps(vector)
                ))
            
            cursor.executemany(
                "INSERT INTO ecms_knowledge (source, section, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                insert_data
            )
            conn.commit()
            
        except Exception as e:
            print(f"    ✗ Batch failed: {e}")
            conn.rollback()
            continue
    
    cursor.close()
    conn.close()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ecms_knowledge")
    count = cursor.fetchone()[0]
    
    cursor.execute("SELECT DISTINCT source FROM ecms_knowledge LIMIT 10")
    samples = [r[0] for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    print(f"\n✅ Successfully stored {count} chunks in MySQL!")
    print(f"   Sample Sources: {', '.join(samples)}")

if __name__ == "__main__":
    init_database()
    main()
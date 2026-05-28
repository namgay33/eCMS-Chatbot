import json
import re
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
    create_btc = """
    CREATE TABLE IF NOT EXISTS btc_codes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        hs_code VARCHAR(20) NOT NULL,
        description TEXT,
        common_name TEXT,
        search_text TEXT NOT NULL,
        FULLTEXT INDEX idx_search (search_text)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(create_knowledge)
    cursor.execute(create_logs)
    cursor.execute(create_btc)
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized.")

def extract_btc_records(source):
    """Extract individual BTC records from scraped content."""
    records = []
    text = source.get('content', '')
    
    entries = re.findall(r'Entry (\d+):\s*Code:\s*([^\n]*)\s*Description:\s*([^\n]*)(?:\s*Common Names:\s*([^\n]*))?', text)
    
    for entry_num, code, desc, common in entries:
        code_clean = code.strip()
        desc_clean = desc.strip()
        common_clean = common.strip() if common else ''
        
        search_text = f"{code_clean} {desc_clean} {common_clean}".lower()
        
        records.append({
            'hs_code': code_clean,
            'description': desc_clean,
            'common_name': common_clean,
            'search_text': search_text
        })
    
    return records

def main():
    print("Loading content...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Run scrape.py first.")
        return

    all_chunks = []
    btc_records = []
    
    for source in sources:
        meta_header = f"SOURCE_ID: {source['source']}\nTITLE: {source['title']}\n\n"
        
        if source.get('type') == 'excel' and 'btc_' in source['source'].lower():
            records = extract_btc_records(source)
            btc_records.extend(records)
            print(f"Extracted {len(records)} BTC records from {source['source']}")
        
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
    
    if not all_chunks and not btc_records:
        print("No data found. Check your input file.")
        return

    print(f"Created {len(all_chunks)} chunks. Found {len(btc_records)} BTC records.")
    print("Starting embedding...")
    
    embedder = OllamaEmbeddings(model=MODEL)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("TRUNCATE TABLE ecms_knowledge")
    cursor.execute("TRUNCATE TABLE btc_codes")
    conn.commit()
    
    if btc_records:
        print(f"Inserting {len(btc_records)} BTC records...")
        insert_btc = "INSERT INTO btc_codes (hs_code, description, common_name, search_text) VALUES (%s, %s, %s, %s)"
        btc_data = [(r['hs_code'], r['description'], r['common_name'], r['search_text']) for r in btc_records]
        
        for i in range(0, len(btc_data), 1000):
            batch = btc_data[i:i+1000]
            cursor.executemany(insert_btc, batch)
            conn.commit()
            print(f"  Inserted batch {i//1000 + 1}")
    
    batch_size = 5
    total = len(all_chunks)
    
    for i in range(0, total, batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c['chunk_text'] for c in batch]
        
        print(f"Processing Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
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
            print(f"Batch failed: {e}")
            conn.rollback()
            continue
    
    cursor.close()
    conn.close()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ecms_knowledge")
    knowledge_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM btc_codes")
    btc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT DISTINCT source FROM ecms_knowledge LIMIT 10")
    samples = [r[0] for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    print(f"\nSuccessfully stored {knowledge_count} chunks in MySQL!")
    print(f"Successfully stored {btc_count} BTC records!")
    print(f"Sample Sources: {', '.join(samples)}")

if __name__ == "__main__":
    init_database()
    main()
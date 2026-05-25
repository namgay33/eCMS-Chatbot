import json
import mysql.connector
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db_utils import get_db_connection  # Import the context manager

MODEL = "llama3.2:3b"
INPUT_FILE = "ecms_content.json"

# Split on headings first to keep sections intact
splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400,
    separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""]
)

def main():
    print("Loading content...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            sources = json.load(f)
    except FileNotFoundError:
        print(f" Error: {INPUT_FILE} not found. Run scrap.py first.")
        return

    all_chunks = []
    for source in sources:
        # The content already has [TRADER/CUSTOMS] and ## Headings from scrap.py
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
        print("️ No chunks created. Check your input file.")
        return

    print(f"Created {len(all_chunks)} chunks. Starting embedding...")
    
    embedder = OllamaEmbeddings(model=MODEL)
    
    # USE 'with' statement here to properly handle the context manager
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Clear existing data
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

    # Final Stats - Use 'with' again for a new connection
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ecms_knowledge")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT DISTINCT source FROM ecms_knowledge LIMIT 10")
        samples = [r[0] for r in cursor.fetchall()]
        cursor.close()
        
        print(f"\n✅ Successfully stored {count} chunks in MySQL!")
        print(f"   Sample Sources: {', '.join(samples)}")

if __name__ == "__main__":
    main()
import json
import mysql.connector
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db_utils import get_db_connection

MODEL = "llama3.2:3b"
INPUT_FILE = "ecms_content.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=400,
    separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""]
)

def main():
    print("Loading content...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        sources = json.load(f)
    
    all_chunks = []
    for source in sources:
        header = f"DOCUMENT: {source['title']}\nSOURCE: {source['source']}\nTYPE: {source['type']}\n\n"
        chunks = splitter.split_text(source['content'])
        
        for i, chunk in enumerate(chunks):
            full_chunk = header + chunk
            all_chunks.append({
                "source": source['source'],
                "section": source['title'],
                "chunk_index": i,
                "chunk_text": full_chunk
            })
    
    print(f"Created {len(all_chunks)} chunks. Embedding...")
    
    embedder = OllamaEmbeddings(model=MODEL)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE ecms_knowledge")
        conn.commit()
        
        batch_size = 4
        total = len(all_chunks)
        
        for i in range(0, total, batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [c['chunk_text'] for c in batch]
            
            print(f"  Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
            
            try:
                vectors = embedder.embed_documents(texts)
                
                for chunk, vector in zip(batch, vectors):
                    cursor.execute(
                        "INSERT INTO ecms_knowledge (source, section, chunk_text, embedding) VALUES (%s, %s, %s, %s)",
                        (chunk['source'], chunk['section'], chunk['chunk_text'], json.dumps(vector))
                    )
                
                conn.commit()
            except Exception as e:
                print(f"    ✗ Batch failed: {e}")
                continue
        
        cursor.close()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ecms_knowledge")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT DISTINCT section FROM ecms_knowledge")
        sections = [r[0] for r in cursor.fetchall()]
        cursor.close()
    
    print(f"\n✅ Stored {count} chunks in MySQL!")
    print(f"   Sections: {', '.join(sections[:5])}{'...' if len(sections) > 5 else ''}")

if __name__ == "__main__":
    main()
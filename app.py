import os
import re
import json
import random
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request
import mysql.connector
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DB_CONFIG = {
    'host': os.environ.get("DB_HOST", "localhost"),
    'port': int(os.environ.get("DB_PORT", "3306")),
    'user': os.environ.get("DB_USER", ""),
    'password': os.environ.get("DB_PASSWORD", ""),
    'database': os.environ.get("DB_NAME", "ecmschatbotdb")
}

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

embedder = OllamaEmbeddings(model=OLLAMA_MODEL)
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)

RAG_PROMPT = """You are the eCMS Virtual Assistant. Answer based ONLY on the context provided.

CONTEXT:
{context}

RULES:
- Answer using ONLY the facts in the context
- NEVER invent contact details not in the context
- TPN means Tax Payer Number
- BTFN means Bhutan Trade FinNet used by RMA
- RMA means Royal Monetary Authority of Bhutan
- eCMS means Electronic Customs Management System (trade records, taxes, passengers travel record, etc)
- BTFN is not a specific module within the eCMS platform
- Both eCMS (trade records) and BTFN (related to monetary) are used by all the countries, not only India
- Not all services require full registration
- NEVER mention document names or .docx files
- For "overall process" or "how does trade work" questions, COMBINE information from multiple procedures (declaration, payment, manifest, clearance) into a step-by-step flow
- Explain the end-to-end process: registration → declaration → payment → customs approval → clearance
- DO NOT use Markdown formatting. Do NOT use bold (**text**) or italics (*text*). Use plain text only.
- Use simple numbered lists (1. Step one) without bolding the titles.

USER QUESTION: {question}

ANSWER:"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT)


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


def greet(sentence):
    GREET_INPUTS = ("hello", "kuzu zangpo la", "kuzu")
    GREET_RESPONSE = "Kuzu Zangpo La, Welcome to eCMS. How can I assist you?"
    sentence_lower = sentence.lower().strip()
    
    for word in GREET_INPUTS:
        if word.lower() == sentence_lower:
            return GREET_RESPONSE
    return None


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def clean_chunk_text(raw_text):
    text = re.sub(r'^\[(TRADER|CUSTOMS|GENERAL)\]\s+', '', raw_text)
    text = re.sub(r'DOCUMENT:.*?\nSOURCE:.*?\nTYPE:.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'\[TABLE\]\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def is_contact_query(question):
    q = question.lower()
    contact_words = ['contact', 'phone', 'email', 'call', 'reach', 'focal', 
                     'help desk', 'support', 'hotline', 'office', 'address',
                     'thimphu', 'gelephu', 'paro', 'samdrup', 'samtse', 
                     'phuntsholing', 'phuentsholing', 'kolkata', 'jonkhar',
                     'jongkhar', 'rrco']
    return any(w in q for w in contact_words)

def is_process_query(question):
    q = question.lower()
    process_words = ['process', 'how does', 'overall', 'end to end', 'workflow', 
                     'import export', 'trade process', 'how do i trade', 'complete process']
    return any(w in q for w in process_words)

def parse_contact_table(text):
    contacts = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('SOURCE:') or line.startswith('TITLE:'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if any(h in line.lower() for h in ['sl#', 'department', 'name', 'phone number', 'email']):
            continue
        if len(parts) >= 4:
            dept = parts[1] if len(parts) > 1 else ''
            location = 'Thimphu'
            if 'Thimphu' in dept: location = 'Thimphu'
            elif 'Gelephu' in dept: location = 'Gelephu'
            elif 'Paro' in dept: location = 'Paro'
            elif 'Samdrup' in dept or 'Jongkhar' in dept: location = 'Samdrup Jongkhar'
            elif 'Samtse' in dept: location = 'Samtse'
            elif 'Phuntsholing' in dept or 'Phuentsholing' in dept or 'Phuntsoling' in dept: location = 'Phuntsholing'
            elif 'Kolkata' in dept or 'Kolkatta' in dept: location = 'Kolkata'
            
            phone = ''
            email = ''
            for p in parts:
                if '@' in p: email = p.strip()
                elif re.search(r'\d{5,}', p): phone = p.strip()
            
            if phone or email:
                contacts.append({'department': dept, 'location': location, 'phone': phone or 'N/A', 'email': email or 'N/A'})
    return contacts


def get_contacts_by_location(question, all_contacts):
    q_lower = question.lower()
    location_map = {
        'thimphu': ['thimphu'], 'gelephu': ['gelephu'], 'paro': ['paro'],
        'samdrup': ['samdrup', 'jongkhar', 'samdrup jongkhar', 'samdrupjongkhar', 'sj'],
        'samdrup jongkhar': ['samdrup', 'jongkhar', 'samdrup jongkhar', 'samdrupjongkhar', 'sj'],
        'samtse': ['samtse'], 'phuntsholing': ['phuntsholing', 'phuentsholing', 'phuntsoling', 'pling', 'p/ling'],
        'kolkata': ['kolkata', 'kolkatta', 'kolka','calcutta'],
    }
    asked_location = None
    for loc_key, variants in location_map.items():
        if any(v in q_lower for v in variants):
            asked_location = loc_key
            break
    if asked_location:
        filtered = [c for c in all_contacts if c['location'].lower().replace(' ', '') == asked_location.replace(' ', '')]
        if not filtered:
            filtered = [c for c in all_contacts if asked_location.replace(' ', '') in c['location'].lower().replace(' ', '')]
        return filtered
    return all_contacts


def format_contacts(contacts):
    if not contacts:
        return "I don't have contact information for that location. Please visit https://www.ecms.gov.bt/contact-us"
    lines = []
    for c in contacts:
        loc = c.get('location', 'N/A')
        dept = c.get('department', 'N/A')
        phone = c.get('phone', 'N/A')
        email = c.get('email', 'N/A')
        lines.append(f"📍 {loc}")
        if dept and dept != loc:
            lines.append(f"   Department: {dept}")
        lines.append(f"   Phone: {phone}")
        lines.append(f"   Email: {email}")
        lines.append("")
    return "\n".join(lines).strip()


def search_knowledge(question, top_k=4, threshold=0.55):
    question_vector = embedder.embed_query(question)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT source, section, chunk_text, embedding FROM ecms_knowledge")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    scored = []
    q_lower = question.lower()
    
    # Keywords that should trigger specific document filtering
    exemption_keywords = ['exemption', 'duty free', 'waiver', 'quota']
    is_exemption_query = any(kw in q_lower for kw in exemption_keywords)

    for row in rows:
        emb = json.loads(row['embedding'])
        score = cosine_similarity(question_vector, emb)
        
        # Boost score significantly if it's an exemption query and the source matches
        if is_exemption_query:
            source_lower = row['source'].lower()
            if any(kw in source_lower for kw in exemption_keywords):
                score += 0.2  # Artificial boost to prioritize these docs

        scored.append({
            'score': score, 
            'source': row['source'], 
            'section': row['section'], 
            'text': clean_chunk_text(row['chunk_text'])
        })
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter by threshold
    results = [r for r in scored[:top_k] if r['score'] >= threshold]
    
    # Fallback: if no results meet threshold, take the best one anyway to avoid empty response
    if not results and scored:
        results = [scored[0]]
        
    return results


def get_response(question):
    q_lower = question.lower().strip()
    
    if q_lower == 'kuzu zangpo la' or q_lower == 'kuzuzangpola':
        return "Kuzu Zangpo La, Welcome to eCMS. How can I assist you?"
    if q_lower == 'hi' or q_lower == 'hello':
        return "Welcome to eCMS! How can I assist you?"
    if q_lower == 'bye' or q_lower == 'goodbye':
        return "Goodbye! Have a smooth customs experience."
    if q_lower == 'thanks' or q_lower == 'thank you':
        return "You are welcome!"
    
    if is_contact_query(q_lower):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT chunk_text FROM ecms_knowledge WHERE source LIKE %s", ("%contact%",))
        contact_rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if contact_rows:
            all_contacts = []
            for row in contact_rows:
                text = clean_chunk_text(row['chunk_text'])
                contacts = parse_contact_table(text)
                all_contacts.extend(contacts)
            
            seen = set()
            unique_contacts = []
            for c in all_contacts:
                key = (c['location'], c['phone'], c['email'])
                if key not in seen and (c['phone'] != 'N/A' or c['email'] != 'N/A'):
                    seen.add(key)
                    unique_contacts.append(c)
            
            filtered = get_contacts_by_location(question, unique_contacts)
            if filtered:
                return format_contacts(filtered)
            if unique_contacts:
                return format_contacts(unique_contacts)
    
    if is_process_query(q_lower):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT chunk_text FROM ecms_knowledge WHERE source LIKE %s OR source LIKE %s OR source LIKE %s OR source LIKE %s OR source LIKE %s",
            ("%Registration%", "%Declaration%", "%Payment%", "%Clearance%", "%Manifest%")
        )
        process_rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if process_rows:
            process_text = "\n\n".join([clean_chunk_text(r['chunk_text'])[:1500] for r in process_rows[:10]])
            
            process_prompt = """You are the eCMS Virtual Assistant. Describe the OVERALL import/export process using eCMS.

Use the following procedure steps to build a complete end-to-end workflow:

{context}

RULES:
- Combine steps from different procedures into ONE coherent flow
- Start with trader registration
- Then declaration creation
- Then payment
- Then customs processing/approval
- Finally clearance/release
- Use simple numbered steps
- NEVER mention document names
- DO NOT use Markdown formatting. Use plain text only.

USER QUESTION: {question}

ANSWER:""".format(context=process_text, question=question)
            
            response = llm.invoke(process_prompt)
            return response.strip()

    chunks = search_knowledge(question)
    if not chunks:
        return "I don't have information on that eCMS procedure. Please visit https://www.ecms.gov.bt/contact-us for assistance."
    
    context_parts = []
    for c in chunks:
        lines = c['text'].split('\n')
        if lines and lines[0].startswith('##'):
            heading = lines[0].replace('##', '').strip()
            body = '\n'.join(lines[1:])
            context_parts.append(f"[{heading}]\n{body[:1800]}")
        else:
            context_parts.append(c['text'][:2000])
    
    context = "\n\n---\n\n".join(context_parts)
    chain = rag_prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    
    # Clean up Markdown
    answer = re.sub(r'\*\*', '', answer)
    answer = re.sub(r'\*', '', answer)
    
    # Clean up Document references
    answer = re.sub(r'\(?Document \d+:.*\.docx\)?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\[?Source:.*?\]?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'according to .*?document', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'as per .*?document', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\(.*\.docx.*\)', '', answer, flags=re.IGNORECASE)
    
    # Detect Hallucination/Assumptions
    hallucination_indicators = [
        "based on general assumptions",
        "context does not provide",
        "i don't have specific information",
        "please contact eCMS support",
        "not found in the context"
    ]
    if any(indicator in answer.lower() for indicator in hallucination_indicators):
        return "I don't have specific information on that procedure in my current database. Please contact eCMS support for detailed guidance."

    answer = re.sub(r'\n+', '\n', answer)
    answer = re.sub(r' {2,}', ' ', answer)
    return answer.strip()


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/get")
def get_bot_response():
    user_response = request.args.get('msg', '').strip()
    
    if not user_response:
        return {"message": "Please type your question.", "highlighted_questions": get_highlighted_questions()}
    
    user_lower = user_response.lower()
    
    if user_lower == 'bye':
        response_message = "Goodbye! Take Care <3"
    elif user_lower in ('thanks', 'thank you'):
        response_message = "You are Welcome.."
    else:
        greeting = greet(user_lower)
        if greeting:
            response_message = greeting
        else:
            response_message = get_response(user_response)
    
    insert_chat_log(user_response, response_message)
    return {"message": response_message, "highlighted_questions": get_highlighted_questions()}


def get_highlighted_questions():
    return [
        "⚖️ What is eCMS?",
        "📝 How to register in eCMS?",
        "💰 How is eCMS related to BTFN?",
        "❓ How does eCMS work?"
    ]


def insert_chat_log(user_input, response):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO ecms_chatbot_logs (user_input, response, timestamp) VALUES (%s, %s, %s)"
        cursor.execute(query, (user_input, response, timestamp))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error logging: {e}")


@app.route("/admin/stats")
def stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as total FROM ecms_knowledge")
        knowledge = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as total FROM ecms_chatbot_logs")
        logs = cursor.fetchone()['total']
        cursor.close()
        conn.close()
        return {"knowledge_chunks": knowledge, "chat_logs": logs}
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    init_database()
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"🚀 eCMS Chatbot: http://127.0.0.1:{port}")
    app.run(use_reloader=False, debug=False, port=port)
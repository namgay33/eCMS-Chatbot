import os
import re
import json
import random
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, session
import mysql.connector
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ecms-chatbot-secret-key")

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
- BIRMS means Bhutan Integrated Revenue Management System
- CID means Citizen Identity
- BTC means Bhutan Trade Classification (HS Code system used in Bhutan)
- eCMS means Electronic Customs Management System (trade records, taxes, passengers travel record, etc)
- BTFN is not a specific module within the eCMS platform
- Payment in eCMS is done through BIRMS
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY unique_chunk (source, section, chunk_text(255))
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    create_logs = """
    CREATE TABLE IF NOT EXISTS ecms_chatbot_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_input TEXT,
        response TEXT,
        timestamp DATETIME
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
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
    text = re.sub(r'^\[(TRADER|CUSTOMS|GENERAL|TARIFF)\]\s+', '', raw_text)
    text = re.sub(r'SOURCE_ID:.*?\nTITLE:.*?\n\n', '', text, flags=re.DOTALL)
    text = re.sub(r'SOURCE:.*?\nTITLE:.*?\nTYPE:.*?\n\n', '', text, flags=re.DOTALL)
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
                     'jongkhar', 'rrco', 'department', 'customs office']
    return any(w in q for w in contact_words)


def is_process_query(question):
    q = question.lower()
    process_words = ['process', 'how does', 'overall', 'end to end', 'workflow', 
                     'import export', 'trade process', 'how do i trade', 'complete process',
                     'step by step', 'procedure', 'workflow']
    return any(w in q for w in process_words)


def is_btc_query(question):
    q = question.lower()
    btc_words = [
        'btc', 'hs code', 'h.s code', 'h.s. code', 'harmonized system',
        'trade classification', 'commodity code', 'tariff code', 'classification code',
        'customs code', 'import code', 'export code', 'hsn', 'schedule b',
        'code for', 'btc for', 'hs for', 'hscode', 'hs-code',
        'commodity', 'tariff', 'classification'
    ]
    has_code_pattern = bool(re.search(r'\b\d{6,10}\b', question))
    return any(w in q for w in btc_words) or has_code_pattern


def is_btc_info_query(question):
    """Check if user is asking for statistics/info about BTC database."""
    q = question.lower()
    info_words = ['how many', 'total', 'count', 'number of', 'list all', 'show all', 
                  'all codes', 'all hscodes', 'all btc', 'database size']
    return any(w in q for w in info_words)


def is_btc_followup(question):
    q = question.lower()
    followup_words = ['only one', 'single', 'best', 'top', 'final', 'main', 'one code', 
                      'which code', 'give me', 'show me', 'what is the code', 'the code']
    return any(w in q for w in followup_words)


def extract_search_terms(question):
    q_lower = question.lower()
    fillers = ['what', 'is', 'the', 'code', 'for', 'btc', 'hs', 'h.s', 'please', 'tell', 
               'me', 'about', 'find', 'search', 'lookup', 'of', 'give', 'only', 'one', 
               'final', 'best', 'single', 'top', 'show', 'which', 'main', 'and', 'list',
               'its', 'matching', 'too', 'also']
    words = re.findall(r'\b\w+\b', q_lower)
    terms = [w for w in words if w not in fillers and len(w) >= 2]
    return terms


def search_btc_codes(question, context_terms=None):
    terms = extract_search_terms(question)
    
    if not terms and context_terms:
        terms = context_terms
    
    code_match = re.search(r'\b(\d{4,10})\b', question)
    search_code = code_match.group(1) if code_match else None
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    results = []
    
    if search_code:
        cursor.execute(
            "SELECT hs_code, description, common_name FROM btc_codes WHERE hs_code LIKE %s LIMIT 10",
            (f"{search_code}%",)
        )
        exact_matches = cursor.fetchall()
        if exact_matches:
            results.extend(exact_matches)
    
    if terms and len(results) < 10:
        like_conditions = ' OR '.join(['search_text LIKE %s'] * len(terms))
        like_values = [f"%{t}%" for t in terms]
        
        cursor.execute(
            f"SELECT hs_code, description, common_name FROM btc_codes WHERE {like_conditions} LIMIT 50",
            like_values
        )
        text_matches = cursor.fetchall()
        
        seen = {r['hs_code'] for r in results}
        for r in text_matches:
            if r['hs_code'] not in seen:
                results.append(r)
                seen.add(r['hs_code'])
    
    cursor.close()
    conn.close()
    
    return results, terms


def get_btc_stats():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT COUNT(*) as total FROM btc_codes")
    total = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(DISTINCT LEFT(hs_code, 2)) as chapters FROM btc_codes")
    chapters = cursor.fetchone()['chapters']
    
    cursor.execute("SELECT hs_code, description FROM btc_codes ORDER BY hs_code LIMIT 5")
    first_few = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    lines = []
    lines.append(f"BTC/HS Code Database Statistics:")
    lines.append(f"Total codes: {total}")
    lines.append(f"Chapters covered: {chapters}")
    lines.append("")
    lines.append("First few codes:")
    for r in first_few:
        lines.append(f"  {r['hs_code']} - {r['description']}")
    
    return "\n".join(lines)


def score_btc_result(result, terms, search_code):
    score = 0
    code = result['hs_code'].lower()
    desc = result['description'].lower()
    common = result['common_name'].lower() if result['common_name'] else ''
    
    if search_code and search_code in code:
        score += 100
        if code.startswith(search_code):
            score += 50
    
    for term in terms:
        if term in desc:
            score += 25
        if term in code:
            score += 20
        if term in common:
            score += 5
    
    desc_words = desc.split()
    for term in terms:
        if any(term == w or term in w for w in desc_words):
            score += 15
    
    if len(code) >= 6 and search_code and code.startswith(search_code[:4]):
        score += 10
    
    if 'other' in desc and len(terms) > 0:
        score -= 10
    
    return score


def format_btc_results(results, question, terms, single_mode=False, list_matches=False):
    if not results:
        code_match = re.search(r'\b(\d{4,10})\b', question)
        if code_match:
            return f"No BTC/HS code found starting with {code_match.group(1)}. Please verify the code or check the official BTC reference."
        
        if terms:
            return f"No results found for '{' '.join(terms)}'. Try searching with exact product name or BTC code number."
        
        return "Please provide a product name or BTC/HS code number to search."
    
    code_match = re.search(r'\b(\d{4,10})\b', question)
    search_code = code_match.group(1) if code_match else None
    
    scored_results = []
    for r in results:
        s = score_btc_result(r, terms, search_code)
        scored_results.append((s, r))
    
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    if single_mode and not list_matches:
        best = scored_results[0][1]
        code = best['hs_code']
        desc = best['description']
        common = best['common_name']
        
        lines = []
        lines.append(f"The BTC/HS Code is: {code}")
        lines.append(f"Description: {desc}")
        if common:
            common_short = common[:150] + "..." if len(common) > 150 else common
            lines.append(f"Common Names: {common_short}")
        
        return "\n".join(lines)
    
    if single_mode and list_matches:
        best = scored_results[0][1]
        code = best['hs_code']
        desc = best['description']
        common = best['common_name']
        
        lines = []
        lines.append(f"The BTC/HS Code is: {code}")
        lines.append(f"Description: {desc}")
        if common:
            common_short = common[:150] + "..." if len(common) > 150 else common
            lines.append(f"Common Names: {common_short}")
        
        other_matches = [x[1] for x in scored_results[1:6]]
        if other_matches:
            lines.append("")
            lines.append("Other matching codes:")
            for r in other_matches:
                lines.append(f"  {r['hs_code']} - {r['description']}")
        
        return "\n".join(lines)
    
    lines = []
    lines.append("Here are the matching BTC/HS Code entries:")
    lines.append("")
    
    for i, (score, r) in enumerate(scored_results[:10], 1):
        code = r['hs_code']
        desc = r['description']
        common = r['common_name']
        
        lines.append(f"{i}. Code: {code}")
        lines.append(f"   Description: {desc}")
        if common:
            common_display = common[:200] + "..." if len(common) > 200 else common
            lines.append(f"   Common Names: {common_display}")
        lines.append("")
    
    if len(scored_results) > 10:
        lines.append(f"... and {len(scored_results) - 10} more matches.")
    
    return "\n".join(lines)


def parse_contact_table(text):
    contacts = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(line.startswith(x) for x in ['SOURCE:', 'TITLE:', 'TYPE:', 'EXTRA', 'PAGE CONTENT:', '##']):
            continue
        if any(h in line.lower() for h in ['sl#', 's.no', 'serial', 'department', 'name', 'phone number', 'email', 'designation', 'location']):
            if len(line) < 50:
                continue
        
        parts = [p.strip() for p in line.split('|')]
        
        full_line = ' '.join(parts).lower()
        location = 'Thimphu'
        
        location_keywords = {
            'Thimphu': ['thimphu'],
            'Gelephu': ['gelephu'],
            'Paro': ['paro'],
            'Samdrup Jongkhar': ['samdrup', 'jongkhar', 'sj'],
            'Samtse': ['samtse'],
            'Phuntsholing': ['phuntsholing', 'phuentsholing', 'phuntsoling', 'pling', 'p/ling'],
            'Kolkata': ['kolkata', 'kolkatta', 'calcutta'],
        }
        
        for loc_name, keywords in location_keywords.items():
            if any(kw in full_line for kw in keywords):
                location = loc_name
                break
        
        phone = ''
        email = ''
        department = ''
        
        for p in parts:
            p_lower = p.lower()
            if '@' in p and '.' in p:
                email = p.strip()
            elif re.search(r'[\d\+\-\s]{7,}', p) and any(c.isdigit() for c in p):
                phone = p.strip()
            elif len(p) > 3 and not any(x in p_lower for x in ['phone', 'email', 'fax']):
                if not department and len(p) > 5:
                    department = p.strip()
        
        if phone or email:
            contacts.append({
                'department': department or location,
                'location': location,
                'phone': phone or 'N/A',
                'email': email or 'N/A'
            })
    
    return contacts


def get_contacts_by_location(question, all_contacts):
    q_lower = question.lower()
    location_map = {
        'thimphu': ['thimphu'],
        'gelephu': ['gelephu'],
        'paro': ['paro'],
        'samdrup': ['samdrup', 'jongkhar', 'samdrup jongkhar', 'samdrupjongkhar', 'sj'],
        'samdrup jongkhar': ['samdrup', 'jongkhar', 'samdrup jongkhar', 'samdrupjongkhar', 'sj'],
        'samtse': ['samtse'],
        'phuntsholing': ['phuntsholing', 'phuentsholing', 'phuntsoling', 'pling', 'p/ling'],
        'kolkata': ['kolkata', 'kolkatta', 'kolka', 'calcutta'],
    }
    
    asked_location = None
    for loc_key, variants in location_map.items():
        if any(v in q_lower for v in variants):
            asked_location = loc_key
            break
    
    if asked_location:
        asked_normalized = asked_location.replace(' ', '').lower()
        filtered = []
        for c in all_contacts:
            contact_normalized = c['location'].lower().replace(' ', '')
            if asked_normalized == contact_normalized:
                filtered.append(c)
        
        if not filtered:
            for c in all_contacts:
                if asked_normalized in c['location'].lower().replace(' ', ''):
                    filtered.append(c)
        
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
        if dept and dept != loc and dept != 'N/A':
            lines.append(f"   Department: {dept}")
        lines.append(f"   Phone: {phone}")
        lines.append(f"   Email: {email}")
        lines.append("")
    
    return "\n".join(lines).strip()


def search_knowledge(question, top_k=5, threshold=0.55):
    question_vector = embedder.embed_query(question)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT source, section, chunk_text, embedding FROM ecms_knowledge")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    scored = []
    q_lower = question.lower()
    
    source_keywords = {
        'registration': ['registration', 'register', 'sign up', 'account creation', 'tpn', 'tax payer'],
        'declaration': ['declaration', 'declare', 'import', 'export', 'shipment', 'cargo', 'goods'],
        'payment': ['payment', 'pay', 'birms', 'tax', 'duty', 'fee', 'revenue'],
        'clearance': ['clearance', 'clear', 'release', 'approve', 'customs approval'],
        'manifest': ['manifest', 'cargo manifest', 'passenger manifest', 'road manifest', 'air cargo', 'transport'],
        'offence': ['offence', 'appeal', 'penalty', 'violation', 'fine', 'infringement'],
        'exemption': ['exemption', 'duty free', 'waiver', 'quota', 'annual exemption'],
        'valuation': ['valuation', 'value', 'price', 'assess', 'determine value'],
        'risk': ['risk', 'selectivity', 'red channel', 'green channel', 'yellow channel', 'inspection'],
        'report': ['report', 'operational', 'statistics', 'analytics', 'dashboard'],
        'system admin': ['system admin', 'administrator', 'user management', 'role', 'permission', 'access'],
        'act': ['act', 'exemption act', 'annual exemption', 'legislation'],
        'pca': ['pca', 'post clearance', 'audit', 'verification after clearance'],
        'refund': ['refund', 'repayment', 'claim back', 'reimbursement'],
        'duty free': ['duty free', 'quota', 'liquor', 'tobacco', 'allowance'],
        'contact': ['contact', 'phone', 'email', 'office', 'help desk', 'support', 'focal'],
        'tariff': ['tariff', 'hs code', 'btc', 'classification', 'duty rate', 'tax rate', 'commodity', 'harmonized'],
        'warehouse': ['warehouse', 'bonded', 'storage', 'depot']
    }
    
    relevant_sources = set()
    for source, keywords in source_keywords.items():
        if any(kw in q_lower for kw in keywords):
            relevant_sources.add(source)
    
    for row in rows:
        emb = json.loads(row['embedding'])
        score = cosine_similarity(question_vector, emb)
        source_lower = row['source'].lower()
        
        if relevant_sources:
            for src in relevant_sources:
                if src.replace(' ', '_') in source_lower or src in source_lower:
                    score += 0.25
                    break
        
        if 'srs_' in source_lower:
            score += 0.05
        
        scored.append({
            'score': score,
            'source': row['source'],
            'section': row['section'],
            'text': clean_chunk_text(row['chunk_text'])
        })
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    results = [r for r in scored[:top_k] if r['score'] >= threshold]
    if not results and scored:
        results = scored[:3]
    
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
    
    if is_btc_info_query(question):
        return get_btc_stats()
    
    if is_btc_query(question) or is_btc_followup(question):
        context_terms = None
        single_mode = any(w in q_lower for w in ['only one', 'single', 'best', 'top', 'final', 'main', 'one code', 'which code'])
        list_matches = any(w in q_lower for w in ['list', 'matching', 'and', 'also', 'too'])
        
        if is_btc_followup(question) and not is_btc_query(question):
            context_terms = session.get('last_btc_terms', None)
            if not context_terms:
                return "What product are you looking for the BTC/HS code? Please mention the product name."
        
        results, terms = search_btc_codes(question, context_terms)
        
        if terms:
            session['last_btc_terms'] = terms
        
        return format_btc_results(results, question, terms, single_mode=single_mode, list_matches=list_matches)
    
    if is_contact_query(q_lower):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT chunk_text, source FROM ecms_knowledge WHERE source = %s OR source LIKE %s OR source LIKE %s",
            ("contact_webpage", "%contact%", "%focal%")
        )
        contact_rows = cursor.fetchall()
        
        if not contact_rows:
            cursor.execute("SELECT chunk_text, source FROM ecms_knowledge")
            all_rows = cursor.fetchall()
            contact_rows = []
            for row in all_rows:
                text_lower = row['chunk_text'].lower()
                if any(x in text_lower for x in ['phone', 'email', '@', 'thimphu', 'gelephu', 'paro', 'samdrup', 'samtse', 'phuntsholing', 'kolkata', 'department', 'customs office']):
                    contact_rows.append(row)
        
        cursor.close()
        conn.close()
        
        if contact_rows:
            all_contacts = []
            for row in contact_rows:
                text = clean_chunk_text(row['chunk_text'])
                contacts = parse_contact_table(text)
                all_contacts.extend(contacts)
            
            if not all_contacts:
                for row in contact_rows:
                    text = clean_chunk_text(row['chunk_text'])
                    lines = text.split('\n')
                    for line in lines:
                        if any(x in line.lower() for x in ['+975', 'phone', 'tel', 'email', '@']):
                            parts = [p.strip() for p in line.split('|') if p.strip()]
                            if len(parts) >= 2:
                                phone = ''
                                email = ''
                                for p in parts:
                                    if '@' in p:
                                        email = p
                                    elif any(c.isdigit() for c in p) and len(p) > 5:
                                        phone = p
                                if phone or email:
                                    all_contacts.append({
                                        'department': parts[0] if parts else 'General',
                                        'location': 'Bhutan',
                                        'phone': phone or 'N/A',
                                        'email': email or 'N/A'
                                    })
            
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
        
        chunks = search_knowledge(question)
        if chunks:
            context_parts = []
            for c in chunks[:3]:
                context_parts.append(c['text'][:2000])
            
            context = "\n\n".join(context_parts)
            contact_prompt = """You are the eCMS Virtual Assistant. Extract and list contact information from the context.

CONTEXT:
{context}

RULES:
- List only contacts found in the context
- Group by location if possible
- NEVER invent contact details
- Use plain text only, no Markdown

USER QUESTION: {question}

ANSWER:""".format(context=context, question=question)
            
            response = llm.invoke(contact_prompt)
            cleaned = re.sub(r'\*\*', '', response)
            cleaned = re.sub(r'\*', '', cleaned)
            return cleaned.strip()
    
    if is_process_query(q_lower):
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT chunk_text FROM ecms_knowledge WHERE source LIKE %s OR source LIKE %s OR source LIKE %s OR source LIKE %s OR source LIKE %s OR source LIKE %s",
            ("%Registration%", "%Declaration%", "%Payment%", "%Clearance%", "%Manifest%", "%Application%")
        )
        process_rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if process_rows:
            process_text = "\n\n".join([clean_chunk_text(r['chunk_text'])[:1500] for r in process_rows[:12]])
            
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
- NEVER mention document names or .docx files
- DO NOT use Markdown formatting. Use plain text only.
- Keep it concise but complete.

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
    
    answer = re.sub(r'\*\*', '', answer)
    answer = re.sub(r'\*', '', answer)
    answer = re.sub(r'__', '', answer)
    answer = re.sub(r'_', '', answer)
    
    answer = re.sub(r'\(?Document \d+:.*\.docx\)?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\[?Source:.*?\]?', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'according to .*?document', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'as per .*?document', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'\(.*\.docx.*\)', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'SOURCE_ID:.*', '', answer, flags=re.IGNORECASE)
    
    hallucination_indicators = [
        "based on general assumptions",
        "context does not provide",
        "i don't have specific information",
        "please contact eCMS support",
        "not found in the context",
        "i apologize",
        "i'm sorry",
        "i do not have"
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
        "❓ How does eCMS work?",
        "📋 What is the BTC code for horses?"
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
        cursor.execute("SELECT COUNT(*) as total FROM btc_codes")
        btc = cursor.fetchone()['total']
        cursor.close()
        conn.close()
        return {"knowledge_chunks": knowledge, "chat_logs": logs, "btc_codes": btc}
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    init_database()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ecms_knowledge")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM btc_codes")
    btc_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    if count == 0:
        print("WARNING: No knowledge chunks found in database!")
        print("Run: python build_db.py")
    else:
        print(f"Knowledge base loaded: {count} chunks")
    
    if btc_count == 0:
        print("WARNING: No BTC codes found!")
    else:
        print(f"BTC codes loaded: {btc_count} records")
    
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"eCMS Chatbot: http://127.0.0.1:{port}")
    app.run(use_reloader=False, debug=False, port=port)
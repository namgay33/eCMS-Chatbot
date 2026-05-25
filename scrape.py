import os
import json
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from docx import Document

OUTPUT_FILE = "ecms_content.json"
DOCUMENTS_FOLDER = "./documents"
URLS_FILE = "./urls.txt"

TRADER_PATTERNS = [
    r'^1\.', r'^2\.1', r'^2\.2', r'^2\.3', r'^3\.1', r'^4\.', 
    r'Annual Exemption', r'^Declaration', r'^Duty free application',
    r'^Duty free quota$', r'^Exemption', r'^FE Export', r'^PCA', r'^TIER'
]

CUSTOMS_PATTERNS = [
    r'^C\.', r'^D\.', r'^E\.', r'^F\.', r'^G\.', r'^H\.', r'^i\.', r'^I\.',
    r'^ii\.', r'^J\.', r'^K\.', r'^L\.', r'^M\.'
]

def detect_side(filename):
    for pattern in TRADER_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return 'TRADER'
    for pattern in CUSTOMS_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return 'CUSTOMS'
    return 'GENERAL'

def clean_filename(filename):
    name = Path(filename).stem
    name = re.sub(r'^\d+(\.\d+)?\s*[\.\-]?\s*', '', name)
    name = re.sub(r'^[A-Za-z]\.\s*', '', name)
    name = re.sub(r'^ii\.', '', name)
    name = re.sub(r'^FE\s+', '', name)
    return name.strip()

def extract_docx_enhanced(path):
    doc = Document(path)
    sections = []
    current_section = {"heading": "General", "content": []}
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        style_name = para.style.name.lower() if para.style else ""
        is_heading = (
            'heading' in style_name or 
            'title' in style_name or
            (len(text) < 120 and not text.endswith('.') and text[0].isupper())
        )
        
        if is_heading and current_section["content"]:
            sections.append(current_section)
            current_section = {"heading": text, "content": []}
        else:
            current_section["content"].append(text)
    
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            cells = [cell.text.strip() if cell.text else "" for cell in row.cells]
            if any(cells):
                table_rows.append(" | ".join(c for c in cells if c))
        
        if table_rows:
            current_section["content"].append("[TABLE]\n" + "\n".join(table_rows))
    
    if current_section["content"]:
        sections.append(current_section)
    
    full_text = ""
    for sec in sections:
        content = "\n".join(sec["content"])
        full_text += f"\n\n## {sec['heading']}\n{content}"
    
    return full_text.strip()

def scrape_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove scripts/styles but KEEP tables and their structure
        for tag in soup(['script', 'style']):
            tag.decompose()
        
        title = soup.title.string.strip() if soup.title else url
        
        # For contact pages, preserve table structure as text
        if 'contact' in url.lower():
            # Find the contact table specifically
            tables = soup.find_all('table')
            contact_text = f"SOURCE: {url}\nTITLE: {title}\n\n"
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                        contact_text += row_text + "\n"
                contact_text += "\n"
            
            # Also get any extra text after tables
            main = soup.find('main') or soup.find('div', class_=re.compile('content|main'))
            if main:
                extra = main.get_text(separator='\n', strip=True)
                # Remove table content duplicates
                extra_lines = [l for l in extra.split('\n') if l.strip() and l.strip() not in contact_text]
                if extra_lines:
                    contact_text += "\nEXTRA INFO:\n" + "\n".join(extra_lines[:20])
            
            return {
                "source": url,
                "title": "Contact Us",
                "content": contact_text[:25000],
                "type": "url",
                "side": "GENERAL"
            }
        
        # Normal pages
        main = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        text = main.get_text(separator='\n', strip=True) if main else soup.get_text(separator='\n', strip=True)
        
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return {
            "source": url,
            "title": title,
            "content": text[:15000],
            "type": "url",
            "side": "GENERAL"
        }
    except Exception as e:
        print(f"  ✗ Failed: {url} — {e}")
        return None

def main():
    all_data = []
    
    doc_folder = Path(DOCUMENTS_FOLDER)
    if doc_folder.exists():
        docx_files = sorted(doc_folder.glob("*.docx"))
        print(f"📁 Found {len(docx_files)} DOCX files")
        
        for file_path in docx_files:
            side = detect_side(file_path.name)
            clean_name = clean_filename(file_path.name)
            
            print(f"  📄 [{side}] {file_path.name} → '{clean_name}'")
            
            text = extract_docx_enhanced(file_path)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) > 50:
                contextual_text = f"[{side}] {clean_name}\n\n{text}"
                
                all_data.append({
                    "source": f"{side}_{clean_name}",
                    "title": clean_name,
                    "content": contextual_text,
                    "type": "document",
                    "side": side
                })
                print(f"    ✓ {len(text)} chars")
            else:
                print(f"    ⚠️ Empty or too short")
    
    url_file = Path(URLS_FILE)
    if url_file.exists():
        urls = [u.strip() for u in url_file.read_text().splitlines() 
                if u.strip() and not u.strip().startswith('#')]
        
        print(f"\n🌐 Found {len(urls)} URLs")
        for url in urls:
            print(f"  🌐 {url}")
            result = scrape_url(url)
            if result:
                all_data.append(result)
                print(f"    ✓ {len(result['content'])} chars")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(all_data)} sources")
    print(f"   Trader: {sum(1 for d in all_data if d.get('side') == 'TRADER')}")
    print(f"   Customs: {sum(1 for d in all_data if d.get('side') == 'CUSTOMS')}")
    print(f"   General: {sum(1 for d in all_data if d.get('side') == 'GENERAL')}")

if __name__ == "__main__":
    main()
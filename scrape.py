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
    r'Exemption', 
    r'Duty free', 
    r'Annual Exemption',
    r'^1\.', r'^2\.1', r'^2\.2', r'^2\.3', r'^3\.1', r'^4\.', 
    r'^Declaration', r'^FE Export', r'^PCA', r'^TIER'
]

CUSTOMS_PATTERNS = [
    r'^C\.', r'^D\.', r'^E\.', r'^F\.', r'^G\.', r'^H\.', r'^i\.', r'^I\.',
    r'^ii\.', r'^J\.', r'^K\.', r'^L\.', r'^M\.'
]

def detect_side(filename):
    filename_lower = filename.lower()
    
    if any(x in filename_lower for x in ['exemption', 'duty free', 'refund', 'quota']):
        return 'TRADER'
        
    if any(x in filename_lower for x in ['registration', 'declaration', 'payment', 'pca', 'application']):
        return 'TRADER'
        
    if any(x in filename_lower for x in ['clearance', 'offence', 'risk', 'valuation', 'admin', 'system']):
        return 'CUSTOMS'

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
            (len(text) < 100 and not text.endswith('.') and text[0].isupper() and not text[0].isdigit())
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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        print(f"    → Fetching {url}...")
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        
        # Check if we got redirected to a different page
        if resp.url != url:
            print(f"    → Redirected to {resp.url}")
        
        resp.raise_for_status()
        
        # Check if we actually got HTML
        content_type = resp.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            print(f"    ⚠ Non-HTML content: {content_type}")
            return None
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        title = soup.title.string.strip() if soup.title else url
        
        # BROADER contact detection - catch contact, contacts, contact-us, contactus, focal, helpdesk
        url_lower = url.lower()
        is_contact_page = any(x in url_lower for x in ['contact', 'focal', 'helpdesk', 'help-desk', 'support'])
        
        # Also check page title/content for contact indicators if URL doesn't match
        if not is_contact_page and title:
            title_lower = title.lower()
            is_contact_page = any(x in title_lower for x in ['contact', 'focal person', 'help desk', 'support'])
        
        if is_contact_page:
            print(f"    ✓ Detected as CONTACT page")
            
            # Extract ALL tables (contact info is usually in tables)
            tables = soup.find_all('table')
            contact_text = f"SOURCE: {url}\nTITLE: {title}\nTYPE: CONTACT_PAGE\n\n"
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                        contact_text += row_text + "\n"
                contact_text += "\n"
            
            # Also extract any paragraph/div text that might contain contact info
            main = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|body'))
            if main:
                # Get all paragraphs and divs with text
                text_elements = main.find_all(['p', 'div', 'li'])
                extra_lines = []
                for elem in text_elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 10:
                        # Look for phone/email/location indicators
                        if any(indicator in text.lower() for indicator in ['phone', 'tel', 'email', '@', 'thimphu', 'gelephu', 'paro', 'samdrup', 'samtse', 'phuntsholing', 'kolkata', 'office', 'department']):
                            extra_lines.append(text)
                
                if extra_lines:
                    contact_text += "\nEXTRA CONTACT INFO:\n" + "\n".join(extra_lines[:50])
            
            # If no tables found, just grab all text
            if not tables:
                text = soup.get_text(separator='\n', strip=True)
                text = re.sub(r'\n+', '\n', text)
                contact_text += "\nPAGE CONTENT:\n" + text[:10000]
            
            return {
                "source": "contact_webpage",
                "title": "Contact Us",
                "content": contact_text[:30000],
                "type": "url",
                "side": "GENERAL"
            }
        
        # Non-contact pages
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
        
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Network error: {url} — {e}")
        return None
    except Exception as e:
        print(f"    ✗ Failed: {url} — {e}")
        return None

def main():
    all_data = []
    
    # Process DOCX files
    doc_folder = Path(DOCUMENTS_FOLDER)
    if doc_folder.exists():
        docx_files = sorted(doc_folder.glob("*.docx"))
        print(f" Found {len(docx_files)} DOCX files")
        
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
                print(f"    ⚠ Empty or too short")
    
    # Process URLs
    url_file = Path(URLS_FILE)
    if url_file.exists():
        urls = [u.strip() for u in url_file.read_text().splitlines() 
                if u.strip() and not u.strip().startswith('#')]
        
        print(f"\n Found {len(urls)} URLs")
        
        for url in urls:
            print(f"  🌐 {url}")
            result = scrape_url(url)
            if result:
                all_data.append(result)
                print(f"    ✓ Source: {result['source']} | {len(result['content'])} chars")
            else:
                print(f"    ✗ Skipped (failed or empty)")
    else:
        print(f"\n ⚠ URLs file not found: {URLS_FILE}")
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(all_data)} sources to {OUTPUT_FILE}")
    print(f"   Trader: {sum(1 for d in all_data if d.get('side') == 'TRADER')}")
    print(f"   Customs: {sum(1 for d in all_data if d.get('side') == 'CUSTOMS')}")
    print(f"   General: {sum(1 for d in all_data if d.get('side') == 'GENERAL')}")
    
    # Show contact sources specifically
    contact_sources = [d['source'] for d in all_data if 'contact' in d['source'].lower()]
    if contact_sources:
        print(f"   Contact sources: {contact_sources}")
    else:
        print(f"   ⚠ No contact sources found!")
        print(f"   Check your urls.txt file and ensure the contact page URL is listed.")

if __name__ == "__main__":
    main()
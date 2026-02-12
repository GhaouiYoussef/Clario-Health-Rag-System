import os
import re
import json
import unicodedata
from collections import Counter
from copy import deepcopy

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not installed. Some cleaning functions will use regex fallback.")

# Default configuration
# Assuming the script is run from project root, or we can use relative paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ABREVS_DIR = os.path.join(PROJECT_ROOT, "data", "documents", "abrevs")

class AbbreviationExpander:
    def __init__(self, file_name, abrevs_dir=ABREVS_DIR):
        self.abrev_map = {}
        # Ensure directory exists or handle gracefully
        abrev_file = os.path.join(abrevs_dir, f"{file_name}.json")

        if not os.path.exists(abrev_file):
            # Try checking if file_name already has extension or path, purely basename
            base_name = os.path.basename(file_name).replace('.pdf', '').strip()
            abrev_file = os.path.join(abrevs_dir, f"{base_name}.json")

        if not os.path.exists(abrev_file):
            # print(f"No abbreviation file found for {file_name} at {abrev_file}. Abbreviations will not be expanded.") # Verbose
            self.pattern = re.compile(r"(?!x)x")  # Matches nothing
            return

        try:
            with open(abrev_file, "r", encoding="utf-8") as f:
                self.abrev_map = json.load(f)
        except Exception as e:
            print(f"Error loading abbreviation file {abrev_file}: {e}")
            self.pattern = re.compile(r"(?!x)x")
            return

        # Sort keys by length descending to match longest first
        sorted_keys = sorted(self.abrev_map.keys(), key=len, reverse=True)
        if sorted_keys:
            self.pattern = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
            )
        else:
            self.pattern = re.compile(r"(?!x)x")

    def expand(self, text):
        if not self.abrev_map:
            return text
        return self.pattern.sub(
            lambda m: self.abrev_map[m.group(0)], text
        )

# Noise Learning Constants
WORDS = 5
ALLOWED_PUNCT_SET = set(".,;:?!'\"()-")
ALLOWED_PUNCT_REGEX = r"\.\,\;\:\?\!\'\"\(\)\-"

def normalize(text):
    # Remove leading numbers/whitespace to help noise learning
    text_no_num = re.sub(r'^\s*\d+\s+', '', text)
    return re.sub(r'\s+', ' ', text_no_num.lower()).strip()

def make_start_pattern(noise_text):
    words = noise_text.split()
    return re.compile(r"^\s*(?:\d+\s+)?\s*" + r"\s+".join(map(re.escape, words)), re.IGNORECASE)

def make_end_pattern(noise_text):
    words = noise_text.split()
    return re.compile(r"\s+".join(map(re.escape, words)) + r"\s*$", re.IGNORECASE)

def learn_noise_direction(pages_text, direction='start'):
    noise_patterns = []
    current_texts = list(pages_text)
    total_docs = len(current_texts)
    
    if total_docs == 0:
        return []

    max_iterations = 10 
    
    for _ in range(max_iterations):
        found_layer = False
        best_candidate = None
        
        # Check varying lengths from WORDS down to 1
        for length in range(WORDS, 0, -1):
            candidates = []
            for text in current_texts:
                words = text.split()
                if not words: continue
                
                phrase = None
                if direction == 'start':
                    if len(words) >= length:
                        phrase = " ".join(words[:length])
                else: # end
                    if len(words) >= length:
                        phrase = " ".join(words[-length:])
                
                if phrase:
                    candidates.append(phrase)
            
            if not candidates:
                continue
                
            counts = Counter(candidates)
            cand, count = counts.most_common(1)[0]
            
            # Threshold: > 15% of pages
            if count / total_docs > 0.15:
                best_candidate = cand
                found_layer = True
                break
        
        if found_layer and best_candidate:
            noise_patterns.append(best_candidate)
            # Strip this noise
            for i, text in enumerate(current_texts):
                if direction == 'start':
                    if text.startswith(best_candidate):
                         current_texts[i] = text[len(best_candidate):].strip()
                else:
                    if text.endswith(best_candidate):
                         current_texts[i] = text[:-len(best_candidate)].strip()
        else:
            break
            
    return noise_patterns

def learn_noise(pages_map):
    page_nums = sorted(pages_map.keys())
    normalized_pages = [normalize(pages_map[p]) for p in page_nums]
    
    noise_starts = learn_noise_direction(normalized_pages, 'start')
    noise_ends = learn_noise_direction(normalized_pages, 'end')
    
    return noise_starts, noise_ends

def apply_noise(pages_map, noise_starts, noise_ends):
    cleaned = {}
    page_nums = sorted(pages_map.keys())
    
    start_patterns = [make_start_pattern(n) for n in noise_starts]
    end_patterns = [make_end_pattern(n) for n in noise_ends]

    for p in page_nums:
        text = pages_map[p]
        changed = True
        while changed:
            changed = False
            for pattern in start_patterns:
                match = pattern.match(text)
                if match:
                    text = text[match.end():].lstrip()
                    changed = True
            
            for pattern in end_patterns:
                match = pattern.search(text)
                if match:
                    text = text[:match.start()].rstrip()
                    changed = True
        cleaned[p] = text

    return cleaned

def clean_document_chapters(chapters):
    """
    Cleans a list of chapters belonging to a SINGLE document.
    Aggregates pages to learn noise globally, then applies cleaning.
    """
    if not chapters:
        return []
        
    # --- 1. Aggregation ---
    global_page_map = {}
    
    for ch in chapters:
        pages = None
        if "pages" in ch:
            pages = ch["pages"]
        else:
            # Handle potential nested format
            keys = list(ch.keys())
            if keys:
                first_val = ch[keys[0]]
                if isinstance(first_val, dict) and "pages" in first_val:
                    pages = first_val["pages"]
        
        if pages:
            for p_num, p_text in pages.items():
                global_page_map[p_num] = p_text
                
    if not global_page_map:
        return chapters

    # --- 2. Global Learning ---
    noise_starts, noise_ends = learn_noise(global_page_map)
    
    # --- 3. Global Application ---
    cleaned_global_pages = apply_noise(global_page_map, noise_starts, noise_ends)
    
    # --- 4. Post-processing & Redistribution ---
    leading_digit_pattern = re.compile(r'^\s*\d+\s+')
    leading_non_letter_pattern = re.compile(r'^[^a-zA-Z]+')
    
    new_chapters = deepcopy(chapters)
    
    for ch in new_chapters:
        pages_ref = None
        if "pages" in ch:
            pages_ref = ch["pages"]
        else:
            keys = list(ch.keys())
            if keys:
                first_val = ch[keys[0]]
                if isinstance(first_val, dict) and "pages" in first_val:
                    pages_ref = first_val["pages"]
                    
        if pages_ref is None:
            continue
            
        for p_num in list(pages_ref.keys()):
            if p_num in cleaned_global_pages:
                new_text = cleaned_global_pages[p_num]
                new_text = leading_digit_pattern.sub('', new_text)
                new_text = leading_non_letter_pattern.sub('', new_text)
                pages_ref[p_num] = new_text

        # --- Remove Last Page of Chapter if garbage ---
        if pages_ref:
            try:
                # Assuming integer keys, but they might be strings in dict
                int_keys = []
                for k in pages_ref.keys():
                    if isinstance(k, int): int_keys.append(k)
                    elif str(k).isdigit(): int_keys.append(int(k))
                
                if int_keys:
                    max_key = max(int_keys)
                    # We need to access via the original key type
                    lookup_key = max_key
                    if max_key not in pages_ref and str(max_key) in pages_ref:
                        lookup_key = str(max_key)
                    
                    if lookup_key in pages_ref:
                        last_text = pages_ref[lookup_key]
                        if len(last_text.split()) < 10:
                            del pages_ref[lookup_key]
            except Exception:
                pass
                
    return new_chapters

def restore_word_boundaries(text: str) -> str:
    if not text:
        return text

    # letter followed by uppercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # letter followed by digit
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
    # digit followed by letter
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    # punctuation followed by letter
    text = re.sub(r'([.,;:?!])([A-Za-z])', r'\1 \2', text)
    # fix cases like "surveyalso"
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_text_post(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    cleaned_chars = []
    
    # Custom filtering based on original notebook
    for char in text:
        if char.isalnum():
            cleaned_chars.append(char)
        elif char.isspace():
            cleaned_chars.append(" ")
        elif char in ALLOWED_PUNCT_SET:
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ") # replace others with space

    text = "".join(cleaned_chars)
    text = restore_word_boundaries(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def nltk_clean(text):
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            print("NLTK 'punkt' not found, downloading...")
            nltk.download("punkt")
            nltk.download("punkt_tab")
            sentences = sent_tokenize(text)
    else:
        sentences = re.split(r"[.!?]", text)

    cleaned = []
    for s in sentences:
        s = re.sub(rf"[^a-zA-Z0-9{ALLOWED_PUNCT_REGEX}\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) > 3:
            cleaned.append(s)

    return " ".join(cleaned)

def post_clean_document_structure(data: dict) -> dict:
    cleaned_data = deepcopy(data)
    for doc_id, sections in cleaned_data.items():
        for section in sections:
            if "title" in section:
                section["title"] = clean_text_post(section["title"])
            if "pages" in section:
                for page_num, content in section["pages"].items():
                    section["pages"][page_num] = clean_text_post(content)
    return cleaned_data

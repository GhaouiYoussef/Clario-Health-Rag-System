import fitz  # PyMuPDF

import os
from typing import List, Dict, Any, Tuple
from src.models import SourceDocument
import re

def extract_pdf_elements(file_path: str, images_output_dir: str) -> List[SourceDocument]:
    """
    Extracts text, headlines, page numbers, and images from a PDF.
    
    Args:
        file_path: Path to the PDF file.
        images_output_dir: Directory to save extracted images.
        
    Returns:
        List of SourceDocument (Pydantic) with rich metadata.
    """
    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)

    doc = fitz.open(file_path)
    file_name = os.path.basename(file_path)
    documents = []

    for page_num, page in enumerate(doc):
        # 1. Extract Text and identify "headlines"
        # We'll consider a "headline" to be any text block with a font size significantly larger than the average.
        # This is a heuristic.
        
        text_blocks = page.get_text("dict")["blocks"]
        plain_text = page.get_text("text")  # For the main content
        
        headlines = []
        # Calculate average font size (heuristic)
        font_sizes = []
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
        
        if font_sizes:
            avg_font_size = sum(font_sizes) / len(font_sizes)
            # Threshold: let's say 1.2x average is a "heading" candidate, or hardcode typical sizes
            # A simpler approach for now: grab the largest text on the page if it's > 14pt (typical doc body is 10-12)
            
            for block in text_blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["size"] > avg_font_size * 1.2 and span["size"] > 12:
                                text = span["text"].strip()
                                if len(text) > 3: # Ignore tiny artifacts
                                    headlines.append(text)

        # 2. Extract Images
        image_list = page.get_images(full=True)
        saved_image_paths = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image
            image_filename = f"{os.path.splitext(file_name)[0]}_p{page_num+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(images_output_dir, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            saved_image_paths.append(image_path)

        # 3. Create Document Object
        # We consolidate headers into a single string for metadata
        headlines_str = " | ".join(sorted(list(set(headlines)), key=headlines.index)) # preserve order, remove dupes
        
        metadata = {
            "source": file_name,
            "page": page_num + 1,
            "headlines": headlines_str,
            "has_images": len(saved_image_paths) > 0,
            "image_paths": str(saved_image_paths),  # Chroma metadata must be simpler types (str, int, float, bool)
        }
        
        doc_obj = SourceDocument(
            page_content=plain_text,
            metadata=metadata
        )
        documents.append(doc_obj)

    doc.close()
    return documents

def load_documents_custom(directory_path: str, images_output_dir: str) -> List[SourceDocument]:
    """Loads PDFs using the custom extraction method."""
    all_documents = []
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            try:
                print(f"Processing (Custom) {filename}...")
                docs = extract_pdf_elements(file_path, images_output_dir)
                all_documents.extend(docs)
                print(f"  - Extracted {len(docs)} pages with {sum(1 for d in docs if d.metadata['has_images'])} containing images.")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return all_documents

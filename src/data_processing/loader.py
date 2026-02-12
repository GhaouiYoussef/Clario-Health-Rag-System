import os
import json
import fitz  # PyMuPDF
# from PIL import Image
# import io
from .cleaner import AbbreviationExpander, nltk_clean
from src.utils import get_file_name

# def extract_images(doc, doc_name, image_root="images"):
#     image_map = {}
#    
#     # Ensure image_root is handled correctly relative to execution or passed val
#     doc_folder = os.path.join(image_root, doc_name.replace(".pdf", ""))
#     os.makedirs(doc_folder, exist_ok=True)
#
#     for page_index in range(len(doc)):
#         page = doc[page_index]
#         page_number = page_index + 1
#         image_list = []
#
#         # get_images(full=True) returns list of (xref, smask, width, height, bpc, colorspace, alt. colorspace, name, filter, referencer)
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             try:
#                 base = doc.extract_image(xref)
#                 image_bytes = base["image"]
#                 ext = base["ext"]
#
#                 img_name = f"page_{page_number}_img_{img_index + 1}.{ext}"
#                 img_path = os.path.join(doc_folder, img_name)
#
#                 with open(img_path, "wb") as f:
#                     f.write(image_bytes)
#
#                 image_list.append(img_path)
#             except Exception as e:
#                 print(f"Failed to extract image {img_index} on page {page_number}: {e}")
#
#         if image_list:
#             image_map[page_number] = image_list
#
#     return image_map

def get_chapters(doc_source, chapter_font_size, start_page=None, end_page=None, image_root="images"):
    """
    Extracts chapters from PDF(s).
    doc_source: Path to file or directory of PDFs.
    chapter_font_size: Font size used to identify chapter headers.
    """
    # handle AbbreviationExpander directory? It picks up from default in cleaner.py
    
    files_to_process = []
    if os.path.isdir(doc_source):
        files_to_process = [
            os.path.join(doc_source, f)
            for f in os.listdir(doc_source)
            if f.lower().endswith(".pdf")
        ]
    elif os.path.isfile(doc_source):
        files_to_process = [doc_source]
    else:
        print(f"Invalid source: {doc_source}")
        return []

    all_chapters = []

    for doc_path in files_to_process:
        try:
            doc = fitz.open(doc_path)
        except Exception as e:
            print(f"Error opening {doc_path}: {e}")
            continue
            
        doc_name = os.path.basename(doc_path)
        file_name_clean = get_file_name(doc_path)

        # ---- IMAGE EXTRACTION (DISABLED) ----
        # image_map = extract_images(doc, doc_name, image_root=image_root)
        
        # Abbreviation setup
        expander = AbbreviationExpander(file_name_clean)

        current_chapter = None

        for page_num, page in enumerate(doc):
            page_number = page_num + 1
            
            # Optional page range filter if single doc
            if start_page and page_number < start_page: continue
            if end_page and page_number > end_page: break

            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception as e:
                print(f"Error getting text on page {page_number}: {e}")
                continue

            for block in blocks:
                if block["type"] != 0: # 0 = text
                    continue

                for line in block["lines"]:
                    for span in line["spans"]:
                        raw_text = span["text"].strip()
                        if not raw_text:
                            continue

                        # ---- CHAPTER TITLE DETECTION ----
                        if int(span["size"]) == chapter_font_size:
                            
                            if current_chapter is not None:
                                merged_text = "".join(
                                    current_chapter["pages"].values()
                                ).strip()

                                if merged_text == "":
                                    # Append multi-line title
                                    current_chapter["title"] += " " + raw_text
                                    continue
                                else:
                                    all_chapters.append(current_chapter)

                            current_chapter = {
                                "doc_name": doc_name,
                                "title": raw_text,
                                "pages": {},
                                # "images": {}
                            }
                            continue

                        # ---- NORMAL CONTENT ----
                        if current_chapter is not None:
                            # Apply NLTK cleaning and Abbreviations
                            # Note: The notebook version in #VSC-2716edf1 calls `nltk_clean(raw_text)`
                            # It DOES NOT call `expander.expand` in that *specific* latest cell override.
                            # BUT earlier version did.
                            # The user probably wants the best version. 
                            # The latest "Re-extract" cell used `nltk_clean` but missed `expander`?
                            # Or maybe `nltk_clean` was deemed sufficient?
                            # I will include expander because it was a specific feature class created.
                            
                            expanded_text = expander.expand(raw_text)
                            clean_text = nltk_clean(expanded_text)
                            
                            if clean_text:
                                current_chapter["pages"].setdefault(page_number, "")
                                current_chapter["pages"][page_number] += clean_text + " "

                            # ---- MAP IMAGES TO CHAPTER (DISABLED) ----
                            # if page_number in image_map:
                            #    current_chapter["images"].setdefault(
                            #        page_number, image_map[page_number]
                            #    )

        if current_chapter is not None:
            all_chapters.append(current_chapter)

        doc.close()

    return all_chapters

def save_chapters_to_json(chapters, output_path="output/chapters.json"):
    serializable = []

    for ch in chapters:
        serializable.append({
            "doc_name": ch["doc_name"],
            "title": ch["title"],
            "pages": {str(k): v for k, v in ch["pages"].items()},
            # "images": {
            #    str(k): v for k, v in ch.get("images", {}).items()
            # }
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(serializable)} chapters to {output_path}")

import re

def split_text_smart(text: str, chunk_size=1000, overlap=200):
    """
    Smart chunking that respects paragraph and sentence boundaries.
    """
    if not text: return []
    
    # Note: notebook mentioned "1. Pre-clean ligatures" here but no code was present.
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Determine Cut Point (End)
        if end < text_len:
            chunk_str = text[start:end]
            min_search = int(len(chunk_str) * 0.6)
            break_index = -1
            
            # Priority 1: Double Newline (Paragraph)
            last_para = chunk_str.rfind('\n\n')
            if last_para > min_search:
                break_index = last_para + 2
                
            # Priority 2: Sentence End
            if break_index == -1:
                for i in range(len(chunk_str) - 1, min_search, -1):
                    if chunk_str[i] in '.!?' and (i + 1 == len(chunk_str) or chunk_str[i+1].isspace()):
                        break_index = i + 1
                        break
            
            # Priority 3: Space
            if break_index == -1:
                last_space = chunk_str.rfind(' ')
                if last_space > min_search:
                    break_index = last_space + 1
            
            # Fallback
            if break_index != -1:
                end = start + break_index

        # Add Chunk
        valid_chunk = text[start:end].strip()
        if valid_chunk:
            chunks.append(valid_chunk)
            
        # Determine Next Start (Overlap)
        if end == text_len:
            break
            
        next_target = end - overlap
        
        # Align `next_target` to the START of a sentence or word
        # Look backwards from next_target for a punctuation or newline
        align_found = False
        
        # 1. Look for paragraph break before target
        para_start = text.rfind('\n\n', start, next_target)
        if para_start != -1:
             start = para_start + 2
             align_found = True
        
        # 2. Look for sentence end
        if not align_found:
            # Simple scan for '. '
            sent_start = -1
            for i in range(next_target, start, -1):
                if text[i] in '.!?' and (i+1 < text_len and text[start].isspace()): # Corrected from text[i+1] to check bounds/isspace? 
                    # Notebook logic:
                    # if text[i] in '.!?' and (i+1 < text_len and text[i+1].isspace()):
                    #    sent_start = i + 1
                    #    break
                    if i + 1 < text_len and text[i+1].isspace():
                         sent_start = i + 1
                         break
            
            # Note: Re-implementing exactly as notebook logic
            sent_start = -1
            for i in range(next_target, start, -1):
                if text[i] in '.!?' and (i+1 < text_len and text[i+1].isspace()):
                    sent_start = i + 1
                    break
            
            if sent_start != -1:
                start = sent_start
                # Skip leading whitespace for new start
                while start < text_len and text[start].isspace():
                    start += 1
                align_found = True
                
        # 3. Fallback to space
        if not align_found:
            space_start = text.rfind(' ', start, next_target)
            if space_start != -1:
                start = space_start + 1
            else:
                start = next_target

    return chunks

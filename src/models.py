from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SummaryRequest(BaseModel):
    slug: str
    max_words: int = Field(default=500, ge=100, le=2000)

class SubtopicSummary(BaseModel):
    title: str
    slug: str
    summary: str
    original_word_count: int
    summary_word_count: int

class TopicSummary(BaseModel):
    title: str
    slug: str
    classification: Optional[str] = None
    subtopics: List[SubtopicSummary]
    combined_word_count: int

class SummaryResponse(BaseModel):
    success: bool
    module_title: str
    module_description: str
    topics: List[TopicSummary]
    total_word_count: int
    summary_word_count: int
    error: Optional[str] = None
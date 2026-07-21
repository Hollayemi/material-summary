import os
import re
import hashlib
from typing import List, Dict, Any, Optional
import json
import httpx

class SummaryService:
    def __init__(self, use_local_model: bool = False):
        """
        Initialize the summary service.
        
        Args:
            use_local_model: If True, use local transformers model. 
                            If False, use Hugging Face Inference API (free, no GPU needed)
        """
        self.use_local_model = use_local_model
        self.tokenizer = None
        self.model = None
        self.summarizer = None
        self.model_name = None
        
        # Hugging Face API settings (free tier)
        self.hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "")
        self.hf_api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        
        if self.use_local_model:
            self._initialize_local_model()
        else:
            print("Using Hugging Face Inference API (no local model required)")

    def _initialize_local_model(self):
        """Initialize local summarization model (fallback option)"""
        try:
            print("Initializing local summarization model...")
            # Use smaller model for memory constraints
            self.model_name = "t5-small"  # Much smaller than BART
            
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # CPU
                framework="pt"
            )
            print("Local model initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize local model: {e}")
            self.summarizer = None
            # Fallback to API mode
            self.use_local_model = False
            print("Falling back to Hugging Face Inference API")

    def _chunk_text(self, text: str, max_chunk_length: int = 512) -> List[str]:
        """Split text into chunks for processing"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_chunk_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    async def _summarize_with_api(self, text: str, max_words: int) -> str:
        """Summarize using Hugging Face Inference API (free)"""
        if not text or not text.strip():
            return ""
        
        # If text is short enough, return it as is
        if len(text.split()) <= max_words:
            return text
        
        try:
            # Prepare text for API (limit length)
            words = text.split()
            if len(words) > 1024:
                text = " ".join(words[:1024])
            
            # Calculate max/min lengths for API
            input_word_count = len(text.split())
            max_length = min(max_words, input_word_count // 2)
            min_length = min(max_length // 2, 50)
            
            # Prepare headers
            headers = {"Authorization": f"Bearer {self.hf_api_token}"} if self.hf_api_token else {}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.hf_api_url,
                    headers=headers,
                    json={
                        "inputs": text,
                        "parameters": {
                            "max_length": max_length,
                            "min_length": min_length,
                            "do_sample": False,
                            "truncation": True
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result and isinstance(result, list) and len(result) > 0:
                        summary = result[0].get("summary_text", "")
                        # Ensure we don't exceed max_words
                        summary_words = summary.split()
                        if len(summary_words) > max_words:
                            return " ".join(summary_words[:max_words]) + "..."
                        return summary
                
                # If API fails, fallback to extraction
                print(f"API summarization failed: {response.status_code}")
                return self._extractive_summary(text, max_words)
                
        except Exception as e:
            print(f"API summarization error: {e}")
            return self._extractive_summary(text, max_words)

    def _extractive_summary(self, text: str, max_words: int) -> str:
        """Extractive summarization - picks most important sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return text[:max_words * 5]  # Rough character limit
        
        # Score sentences by length (more words = more informative)
        scored_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 3:  # Ignore very short sentences
                score = len(words)
                # Bonus for sentences with key legal terms
                legal_terms = ["contract", "offer", "acceptance", "court", "held", "agreement", "binding", "revoke"]
                for term in legal_terms:
                    if term.lower() in sentence.lower():
                        score += 5
                scored_sentences.append((score, sentence))
        
        # Sort by score descending
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Select sentences until we reach max_words
        selected = []
        word_count = 0
        for _, sentence in scored_sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= max_words:
                selected.append(sentence)
                word_count += sentence_words
            elif not selected and sentence_words > max_words:
                # If first sentence is too long, truncate
                selected.append(" ".join(sentence.split()[:max_words]) + "...")
                break
            else:
                # Try to add partial sentence to reach limit
                remaining = max_words - word_count
                if remaining > 10:
                    partial = " ".join(sentence.split()[:remaining]) + "..."
                    selected.append(partial)
                break
        
        return " ".join(selected) if selected else text[:max_words * 5]

    def _summarize_text(self, text: str, max_words: int) -> str:
        """Summarize a single text using available method"""
        if not text or not text.strip():
            return ""
        
        # If text is already short enough
        if len(text.split()) <= max_words:
            return text
        
        # Use local model if available
        if self.use_local_model and self.summarizer:
            try:
                input_word_count = len(text.split())
                max_length = min(max_words, input_word_count // 2)
                min_length = min(max_length // 2, 50)
                
                # Chunk long text
                chunks = self._chunk_text(text, 512)
                summaries = []
                
                for chunk in chunks:
                    if len(chunk.split()) < 30:
                        summaries.append(chunk)
                        continue
                        
                    result = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )
                    if result and len(result) > 0:
                        summaries.append(result[0]['summary_text'])
                
                combined = " ".join(summaries)
                final_words = combined.split()
                if len(final_words) > max_words:
                    return " ".join(final_words[:max_words]) + "..."
                return combined
                
            except Exception as e:
                print(f"Local model summarization error: {e}")
                # Fall back to API or extraction
        
        # Try API summarization
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self._summarize_with_api(text, max_words)
            )
            loop.close()
            return result
        except Exception as e:
            print(f"Error in API summarization: {e}")
            # Final fallback: extractive summary
            return self._extractive_summary(text, max_words)

    def _generate_subtopic_summary(self, subtopic: Dict[str, Any], max_words: int) -> Dict[str, Any]:
        """Generate summary for a single subtopic"""
        title = subtopic.get("title", "")
        notes = subtopic.get("notes", "")
        slug = subtopic.get("slug", "")
        
        original_word_count = len(notes.split())
        
        # Determine appropriate length for this subtopic
        if original_word_count > 1000:
            subtopic_max_words = min(max_words // 3, 300)
        elif original_word_count > 500:
            subtopic_max_words = min(max_words // 4, 200)
        else:
            subtopic_max_words = min(max_words // 5, 150)
        
        subtopic_max_words = max(subtopic_max_words, 50)
        
        summary = self._summarize_text(notes, subtopic_max_words)
        
        return {
            "title": title,
            "slug": slug,
            "summary": summary,
            "original_word_count": original_word_count,
            "summary_word_count": len(summary.split())
        }

    async def generate_summary(self, material_data: Dict[str, Any], max_words: int) -> Dict[str, Any]:
        """Generate full course summary"""
        module = material_data.get("module", {})
        topics = material_data.get("topics", [])
        
        topic_summaries = []
        total_summary_words = 0
        
        for topic in topics:
            subtopics = topic.get("subtopics", [])
            if not subtopics:
                continue
                
            topic_title = topic.get("title", "")
            topic_slug = topic.get("slug", "")
            topic_classification = topic.get("classification", "")
            
            # Calculate total words in topic
            total_topic_words = sum(len(st.get("notes", "").split()) for st in subtopics)
            if total_topic_words == 0:
                continue
            
            subtopic_summaries = []
            for subtopic in subtopics:
                subtopic_word_count = len(subtopic.get("notes", "").split())
                if subtopic_word_count == 0:
                    continue
                
                # Proportional allocation
                proportion = subtopic_word_count / total_topic_words if total_topic_words > 0 else 0
                subtopic_max_words = max(int(proportion * max_words), 30)
                
                summary_data = self._generate_subtopic_summary(subtopic, subtopic_max_words)
                subtopic_summaries.append(summary_data)
                total_summary_words += summary_data["summary_word_count"]
            
            if subtopic_summaries:
                topic_summaries.append({
                    "title": topic_title,
                    "slug": topic_slug,
                    "classification": topic_classification,
                    "subtopics": subtopic_summaries,
                    "combined_word_count": sum(s["summary_word_count"] for s in subtopic_summaries)
                })
        
        return {
            "module_title": module.get("title", ""),
            "module_description": module.get("description", ""),
            "topics": topic_summaries,
            "total_word_count": total_summary_words,
            "summary_word_count": total_summary_words
        }
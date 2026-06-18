"""
Knowledge Base implementation for semantic search over Q&A data.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass

import numpy as np
from loguru import logger
from fastembed import TextEmbedding


@dataclass
class KnowledgeItem:
    """Represents a single Q&A/KB item."""

    id: str
    questions: List[str]
    answer: str
    implications: List[str]
    source: str
    # Optional metadata (used by Lofi KB format)
    tags: Optional[List[str]] = None
    kind: Optional[str] = None  # e.g., "faq", "how-to", "product"
    last_updated: Optional[str] = None
    question_embeddings: Optional[List[np.ndarray]] = None


class KnowledgeBase:
    """Knowledge base for semantic search over Q&A data."""

    def __init__(
        self, data_dir: str = "data/canon", model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.model: Optional[TextEmbedding] = None
        self.items: List[KnowledgeItem] = []
        self.question_embeddings: List[np.ndarray] = []
        self._initialized = False

    def _load_data(self) -> None:
        """Load KB data from JSON files in the data directory.

        Supports two formats:
        - Canon format: { source: str, qa: [ {questions: [..], answer: str, implication: [..]} ] }
        - Lofi format: [ { id, type, questions: str|[str], answer, tags: [..], last_updated } ]
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return

        item_id = 0
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Canon format (dict with qa list)
                if isinstance(data, dict) and isinstance(data.get("qa"), list):
                    source = data.get("source", str(json_file))
                    qa_items = data.get("qa", [])
                    for qa_item in qa_items:
                        questions_raw = qa_item.get("questions", [])
                        if isinstance(questions_raw, str):
                            questions = [questions_raw]
                        else:
                            questions = list(questions_raw or [])
                        answer = qa_item.get("answer", "")
                        implications_raw = qa_item.get("implication", [])
                        if isinstance(implications_raw, str):
                            implications = [implications_raw]
                        else:
                            implications = list(implications_raw or [])

                        if questions and answer:
                            item = KnowledgeItem(
                                id=f"item_{item_id}",
                                questions=questions,
                                answer=answer,
                                implications=implications,
                                source=source,
                            )
                            self.items.append(item)
                            item_id += 1

                # Lofi format (list of dicts)
                elif isinstance(data, list):
                    for obj in data:
                        if not isinstance(obj, dict):
                            continue
                        q_raw = obj.get("questions")
                        if isinstance(q_raw, str):
                            questions = [q_raw]
                        elif isinstance(q_raw, list):
                            questions = [
                                str(q) for q in q_raw if isinstance(q, (str, bytes))
                            ]
                        else:
                            questions = []
                        answer = obj.get("answer", "")
                        # Derive simple implications/steps from answer lines (bullets or backticked routes)
                        implications: List[str] = []
                        try:
                            lines = str(answer).splitlines()
                            for ln in lines:
                                ln_strip = ln.strip()
                                if not ln_strip:
                                    continue
                                if ln_strip.startswith("-") or ln_strip.startswith("•"):
                                    implications.append(ln_strip)
                                else:
                                    ticks = re.findall(r"`([^`]+)`", ln_strip)
                                    if ticks:
                                        implications.append(ln_strip)
                        except Exception:
                            pass
                        if questions and answer:
                            item = KnowledgeItem(
                                id=str(obj.get("id") or f"item_{item_id}"),
                                questions=questions,
                                answer=answer,
                                implications=implications,
                                source=str(json_file.name),
                                tags=obj.get("tags"),
                                kind=obj.get("type"),
                                last_updated=obj.get("last_updated"),
                            )
                            self.items.append(item)
                            item_id += 1

                else:
                    logger.debug(f"Skipping unrecognized KB format: {json_file}")

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue

        logger.info(f"Loaded {len(self.items)} knowledge items from {self.data_dir}")

    def _initialize_model(self) -> None:
        """Initialize the embedding model (fastembed)."""
        try:
            self.model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            raise

    def _generate_embeddings(self) -> None:
        """Generate embeddings for all questions."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        all_questions = []
        for item in self.items:
            all_questions.extend(item.questions)

        if not all_questions:
            logger.warning("No questions found to embed")
            return

        try:
            # fastembed returns an iterator of list[float]; convert to numpy array
            embeddings = np.array(list(self.model.embed(all_questions)))
            logger.info(f"Generated embeddings for {len(all_questions)} questions")

            # Store embeddings per question
            embedding_idx = 0
            for item in self.items:
                item.question_embeddings = []
                for _ in item.questions:
                    item.question_embeddings.append(embeddings[embedding_idx])
                    embedding_idx += 1

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def initialize(self) -> None:
        """Initialize the knowledge base by loading data and generating embeddings."""
        if self._initialized:
            return

        try:
            self._load_data()
            self._initialize_model()
            self._generate_embeddings()
            self._initialized = True
            logger.info("Knowledge base initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Tuple[KnowledgeItem, float]]:
        """
        Search for the most similar questions using semantic similarity.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of tuples containing (KnowledgeItem, similarity_score)
        """
        if not self._initialized:
            self.initialize()

        if not self.model or not self.items:
            return []

        try:
            # Generate embedding for the query
            query_embedding = np.array(list(self.model.embed([query])))[0]

            # Calculate similarities
            similarities = []
            for item in self.items:
                if (
                    not hasattr(item, "question_embeddings")
                    or not item.question_embeddings
                ):
                    continue

                # Find the best matching question for this item
                best_similarity = 0.0
                for question_embedding in item.question_embeddings:
                    # Cosine similarity via NumPy
                    denom = (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(question_embedding)
                    ) or 1.0
                    similarity = float(
                        np.dot(query_embedding, question_embedding) / denom
                    )
                    best_similarity = max(best_similarity, similarity)

                similarities.append((item, best_similarity))

            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def get_item_by_id(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by its ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None


# Global instance
_kb_instance: Optional[KnowledgeBase] = None
_lofi_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "canon")
        _kb_instance = KnowledgeBase(data_dir=data_dir)
    return _kb_instance


def get_lofi_knowledge_base() -> KnowledgeBase:
    """Get the global Lofi product knowledge base instance."""
    global _lofi_kb_instance
    if _lofi_kb_instance is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "lofi")
        _lofi_kb_instance = KnowledgeBase(data_dir=data_dir)
    return _lofi_kb_instance

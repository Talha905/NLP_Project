"""
Recursive text chunking with configurable overlap and boundary tracking.
"""

from typing import List, Optional
from pydantic import BaseModel
from .document_loader import Document


class TextChunk(BaseModel):
    """Represents a discrete text chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    source_filename: str = ""


class RecursiveChunker:
    """
    Recursively chunks text using paragraph, sentence, and word boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _approx_token_count(self, text: str) -> int:
        """Approximates token count (avg ~4 characters per token)."""
        return max(1, len(text) // 4)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive split logic."""
        if not separators:
            return [text]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        result = []
        for s in splits:
            if not s:
                continue
            if len(s) > self.chunk_size and remaining_seps:
                result.extend(self._split_text(s, remaining_seps))
            else:
                result.append(s)
        return result

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Splits a Document into TextChunk objects with overlap."""
        text = document.content
        if not text:
            return []

        raw_splits = self._split_text(text, self.separators)

        chunks: List[TextChunk] = []
        current_chunk_pieces: List[str] = []
        current_len = 0
        current_start = 0

        for piece in raw_splits:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current_chunk_pieces:
                combined_text = " ".join(current_chunk_pieces).strip()
                if combined_text:
                    chunk_id = f"{document.doc_id}_chunk_{len(chunks)}"
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=document.doc_id,
                            text=combined_text,
                            start_char=current_start,
                            end_char=current_start + len(combined_text),
                            token_count=self._approx_token_count(combined_text),
                            source_filename=document.filename
                        )
                    )
                # Overlap logic: keep trailing pieces that fit within overlap budget
                overlap_pieces = []
                overlap_len = 0
                for p in reversed(current_chunk_pieces):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk_pieces = overlap_pieces
                current_len = overlap_len
                current_start = max(0, current_start + len(combined_text) - overlap_len)

            current_chunk_pieces.append(piece)
            current_len += piece_len

        # Final chunk
        if current_chunk_pieces:
            combined_text = " ".join(current_chunk_pieces).strip()
            if combined_text:
                chunk_id = f"{document.doc_id}_chunk_{len(chunks)}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        text=combined_text,
                        start_char=current_start,
                        end_char=current_start + len(combined_text),
                        token_count=self._approx_token_count(combined_text),
                        source_filename=document.filename
                    )
                )

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """Splits multiple documents into TextChunks."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

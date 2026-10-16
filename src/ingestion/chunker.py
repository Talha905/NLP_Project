"""
Recursive text chunking with configurable overlap and boundary tracking.
"""

from typing import List, Optional
from pydantic import BaseModel
from .document_loader import Document


class TextChunk(BaseModel):
    """Represents a discrete text chunk with metadata and hierarchical parent linkage."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    source_filename: str = ""
    is_summary: bool = False
    parent_id: Optional[str] = None
    parent_text: Optional[str] = None


class RecursiveChunker:
    """
    Recursively chunks text using paragraph, sentence, and word boundaries.
    Supports standard flat chunking and Parent-Child Hierarchical Chunking.
    """

    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 40,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 150,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def _approx_token_count(self, text: str) -> int:
        """Approximates token count (avg ~4 characters per token)."""
        return max(1, len(text) // 4)

    def _split_text(self, text: str, separators: List[str], max_size: int) -> List[str]:
        """Recursive split logic respecting max_size."""
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
            if len(s) > max_size and remaining_seps:
                result.extend(self._split_text(s, remaining_seps, max_size))
            else:
                result.append(s)
        return result

    def _create_chunks_from_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        doc_id: str,
        filename: str,
        id_prefix: str = "chunk",
        start_offset: int = 0
    ) -> List[TextChunk]:
        raw_splits = self._split_text(text, self.separators, max_size=chunk_size)
        chunks: List[TextChunk] = []
        current_pieces: List[str] = []
        current_len = 0
        current_start = start_offset

        for piece in raw_splits:
            piece_len = len(piece)
            if current_len + piece_len > chunk_size and current_pieces:
                combined_text = " ".join(current_pieces).strip()
                if combined_text:
                    cid = f"{doc_id}_{id_prefix}_{len(chunks)}"
                    chunks.append(
                        TextChunk(
                            chunk_id=cid,
                            doc_id=doc_id,
                            text=combined_text,
                            start_char=current_start,
                            end_char=current_start + len(combined_text),
                            token_count=max(1, len(combined_text) // 4),
                            source_filename=filename
                        )
                    )
                overlap_pieces = []
                overlap_len = 0
                for p in reversed(current_pieces):
                    if overlap_len + len(p) <= chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_pieces = overlap_pieces
                current_len = overlap_len
                current_start = max(0, current_start + len(combined_text) - overlap_len)

            current_pieces.append(piece)
            current_len += piece_len

        if current_pieces:
            combined_text = " ".join(current_pieces).strip()
            if combined_text:
                cid = f"{doc_id}_{id_prefix}_{len(chunks)}"
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        text=combined_text,
                        start_char=current_start,
                        end_char=current_start + len(combined_text),
                        token_count=max(1, len(combined_text) // 4),
                        source_filename=filename
                    )
                )

        return chunks

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """Standard flat document chunking."""
        return self._create_chunks_from_text(
            text=document.content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            doc_id=document.doc_id,
            filename=document.filename
        )

    @staticmethod
    def _detect_table_header(text: str) -> Optional[str]:
        """Detects if a document contains a table header line (e.g. rosters, marks, tabular records)."""
        lines = text.split("\n")[:50]
        for line in lines:
            cleaned = line.strip().lower()
            if not cleaned:
                continue
            markers = ["roll no", "sr. no", "student name", "total isa", "aptitude", "marks out of"]
            matches = sum(1 for m in markers if m in cleaned)
            if matches >= 2:
                return line.strip()
            if line.strip().startswith("|") and line.strip().endswith("|") and ("---" not in line):
                return line.strip()
        return None

    def chunk_document_hierarchical(self, document: Document) -> List[TextChunk]:
        """
        Creates child chunks (~250 chars) for search indexing, each linked to
        an expanded parent section (~1200 chars) for LLM context generation.
        """
        text = document.content
        if not text:
            return []

        # 1. Create parent sections (~1200 chars)
        parents = self._create_chunks_from_text(
            text=text,
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            doc_id=document.doc_id,
            filename=document.filename,
            id_prefix="parent"
        )

        # Detect tabular columns header and inject into parent sections missing it
        table_header = self._detect_table_header(text)
        if table_header:
            header_prefix = f"[TABLE COLUMNS REFERENCE: {table_header}]\n\n"
            for parent in parents:
                if table_header.lower() not in parent.text.lower():
                    parent.text = header_prefix + parent.text

        all_child_chunks: List[TextChunk] = []

        # 2. For each parent section, create child chunks (~250 chars)
        for p_idx, parent in enumerate(parents):
            child_chunks = self._create_chunks_from_text(
                text=parent.text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                doc_id=document.doc_id,
                filename=document.filename,
                id_prefix=f"p{p_idx}_child",
                start_offset=parent.start_char
            )
            for child in child_chunks:
                child.parent_id = parent.chunk_id
                child.parent_text = parent.text
                all_child_chunks.append(child)

        print(f"[CHUNKER] [HIERARCHICAL] Created {len(parents)} parent sections (~{self.parent_chunk_size}c) and {len(all_child_chunks)} child chunks (~{self.chunk_size}c) for '{document.filename}'")
        return all_child_chunks

    def chunk_documents(self, documents: List[Document], hierarchical: bool = True) -> List[TextChunk]:
        """Splits multiple documents into TextChunks, defaulting to hierarchical chunking."""
        all_chunks = []
        for doc in documents:
            if hierarchical:
                all_chunks.extend(self.chunk_document_hierarchical(doc))
            else:
                all_chunks.extend(self.chunk_document(doc))
        return all_chunks

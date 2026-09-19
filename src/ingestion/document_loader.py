"""
Document loading utilities for text, markdown, and PDF files.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents an ingested document with source metadata."""
    doc_id: str
    filename: str
    source_path: str
    content: str
    char_count: int = 0

    def model_post_init(self, __context):
        if not self.char_count:
            self.char_count = len(self.content)


class DocumentLoader:
    """Loads text, markdown, and PDF documents from files or directories."""

    @staticmethod
    def load_file(file_path: str | Path) -> Optional[Document]:
        """Loads a single document file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        content = ""

        try:
            if ext in [".txt", ".md", ".markdown"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            elif ext == ".pdf":
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages_text = []
                for idx, page in enumerate(reader.pages):
                    page_str = page.extract_text() or ""
                    if page_str.strip():
                        pages_text.append(page_str)
                content = "\n\n".join(pages_text)
            else:
                # Try reading as plain text
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            if not content.strip():
                return None

            return Document(
                doc_id=path.stem,
                filename=path.name,
                source_path=str(path.resolve()),
                content=content.strip(),
                char_count=len(content.strip())
            )
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None

    @classmethod
    def load_directory(cls, dir_path: str | Path, extensions: Optional[List[str]] = None) -> List[Document]:
        """Loads all matching documents from a directory."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        if extensions is None:
            extensions = [".txt", ".md", ".markdown", ".pdf"]

        docs: List[Document] = []
        for file in sorted(path.glob("**/*")):
            if file.is_file() and file.suffix.lower() in extensions:
                doc = cls.load_file(file)
                if doc and len(doc.content) > 20:
                    docs.append(doc)
        return docs

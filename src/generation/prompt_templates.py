"""
Prompt templates for grounded RAG generation, factual constraint enforcement,
and Chain-of-Thought reasoning.
"""

from typing import List
from ..ingestion.chunker import TextChunk


class PromptBuilder:
    """Formats retrieved context and user query into structured LLM prompts."""

    STRICT_GROUNDED_SYSTEM_PROMPT = """You are a helpful, precise fact-based AI assistant.
Your task is to answer the user's question using ONLY the provided reference context.
CRITICAL RULES:
1. Answer concisely and accurately based strictly on the provided context.
2. Rely strictly on facts mentioned in the context. Do NOT extrapolate or introduce external facts.
3. If the context contains tables, rosters, marks, scores, or student records, locate the matching row for the entity/name/ID and extract the requested fields (such as Roll Number, Name, Marks, Scores, or Status). In marksheets, questions about 'marks' or 'scores' refer to the evaluation columns (such as Total ISA, Marks out of 10, CRED Total, Aptitude); extract and report these values.
4. If the context does NOT contain information about the requested person or entity, state clearly: "The provided context does not contain sufficient evidence to answer this question."
5. Do NOT speculate or make up information.
"""

    COT_SYSTEM_PROMPT = """You are a rigorous analytical AI assistant.
Answer the user's question step-by-step using only the provided context.
First, identify the relevant facts or tabular records in the context.
Second, synthesize the facts to formulate your answer.
If the context does not contain the answer, state that the context lacks the required information.
"""

    @classmethod
    def format_context_chunks(cls, retrieved_chunks: List[TextChunk]) -> str:
        """Formats retrieved chunks with clear source boundaries, expanding to parent context if present."""
        if not retrieved_chunks:
            return "No relevant context found."

        formatted_pieces = []
        seen_parents = set()

        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.source_filename or chunk.doc_id

            # If parent context is present, use the complete expanded section
            if getattr(chunk, "parent_text", None) and getattr(chunk, "parent_id", None):
                if chunk.parent_id in seen_parents:
                    continue  # Deduplicate to prevent feeding the same parent paragraph twice
                seen_parents.add(chunk.parent_id)
                body = chunk.parent_text.strip()
                tag = f"[Source Section {idx} (Expanded Parent Context) | ID: {chunk.parent_id} | Doc: {source}]"
            else:
                body = chunk.text.strip()
                tag = f"[Source Chunk {idx} | ID: {chunk.chunk_id} | Doc: {source}]"

            formatted_pieces.append(f"{tag}\n{body}")

        return "\n\n".join(formatted_pieces)

    @classmethod
    def build_grounded_prompt(cls, query: str, chunks: List[TextChunk]) -> str:
        """Constructs prompt with strict grounded instructions."""
        context_str = cls.format_context_chunks(chunks)
        prompt = f"""### CONTEXT:
{context_str}

### QUESTION:
{query}

### ANSWER:"""
        return prompt

    @classmethod
    def build_cot_prompt(cls, query: str, chunks: List[TextChunk]) -> str:
        """Constructs a Chain-of-Thought grounded prompt."""
        context_str = cls.format_context_chunks(chunks)
        prompt = f"""### CONTEXT:
{context_str}

### QUESTION:
{query}

### INSTRUCTIONS:
Think step-by-step. Refer only to the facts present in the context.

### REASONING AND ANSWER:"""
        return prompt

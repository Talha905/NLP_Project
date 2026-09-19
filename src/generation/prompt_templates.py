"""
Prompt templates for grounded RAG generation, factual constraint enforcement,
and Chain-of-Thought reasoning.
"""

from typing import List
from ..ingestion.chunker import TextChunk


class PromptBuilder:
    """Formats retrieved context and user query into structured LLM prompts."""

    STRICT_GROUNDED_SYSTEM_PROMPT = """You are a rigorous, fact-based AI assistant.
Your task is to answer the user's question using ONLY the provided reference context.
CRITICAL RULES:
1. If the answer is directly supported by the context, answer concisely and accurately.
2. Rely strictly on facts mentioned in the context. Do NOT extrapolate or introduce external facts.
3. If the context does not contain sufficient evidence to answer the question, explicitly state: "The provided context does not contain sufficient evidence to answer this question."
4. Do NOT speculate or make up information.
"""

    COT_SYSTEM_PROMPT = """You are a rigorous analytical AI assistant.
Answer the user's question step-by-step using only the provided context.
First, identify the relevant facts in the context.
Second, synthesize the facts to formulate your answer.
If the context does not contain the answer, state that the context lacks the required information.
"""

    @classmethod
    def format_context_chunks(cls, retrieved_chunks: List[TextChunk]) -> str:
        """Formats retrieved chunks with clear source boundaries."""
        if not retrieved_chunks:
            return "No relevant context found."

        formatted_pieces = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.source_filename or chunk.doc_id
            formatted_pieces.append(
                f"[Source Chunk {idx} | ID: {chunk.chunk_id} | Doc: {source}]\n{chunk.text.strip()}"
            )
        return "\n\n".join(formatted_pieces)

    @classmethod
    def build_grounded_prompt(cls, query: str, chunks: List[TextChunk]) -> str:
        """Constructs prompt with strict grounded instructions."""
        context_str = cls.format_context_chunks(chunks)
        prompt = f"""### CONTEXT:
{context_str}

### QUESTION:
{query}

### INSTRUCTIONS:
Answer the question based strictly on the context above. If the context does not contain the answer, reply: "The provided context does not contain sufficient evidence to answer this question."

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

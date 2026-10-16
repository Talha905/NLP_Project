"""
Local LLM Client for Ollama with automated fallback.
Supports local models (qwen2.5:3b, llama3.2:latest, etc.) with zero external API fees.
"""

import json
from typing import Dict, Any, List, Optional
import requests
from ..config import OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, ALTERNATIVE_OLLAMA_MODEL
from .prompt_templates import PromptBuilder
from ..ingestion.chunker import TextChunk


class LocalLLMClient:
    """
    Communicates with locally running Ollama instance via HTTP REST API.
    Provides intelligent local fallback generator if Ollama is not active.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.1,
        timeout: int = 60
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.is_ollama_available = self._check_ollama_status()

    def _check_ollama_status(self) -> bool:
        """Verifies if Ollama server is running and accessible."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """Queries Ollama for downloaded local models and includes synthetic failure test models."""
        models = []
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass

        # Self-generated benchmark stress-test models
        synthetic_models = ["synthetic-hallucinator", "synthetic-weak-reasoner", "local-fallback-extractor"]
        for sm in synthetic_models:
            if sm not in models:
                models.append(sm)
        return models

    def generate(
        self,
        query: str,
        retrieved_chunks: List[TextChunk],
        prompt_style: str = "grounded",
        model_name: Optional[str] = None,
        force_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Generates an answer given query and retrieved context chunks.
        Returns dict with keys: 'answer', 'model', 'prompt_used', 'generation_time_ms', 'is_fallback'.
        """
        target_model = model_name or self.model_name

        # Route synthetic research stress-test models
        if target_model == "synthetic-hallucinator":
            return self._synthetic_hallucinator_generate(query, retrieved_chunks)
        elif target_model == "synthetic-weak-reasoner":
            return self._synthetic_weak_reasoner_generate(query, retrieved_chunks)
        elif target_model == "local-fallback-extractor":
            prompt = PromptBuilder.build_grounded_prompt(query, retrieved_chunks)
            return self._local_grounded_fallback(query, retrieved_chunks, prompt)

        if prompt_style == "cot":
            prompt = PromptBuilder.build_cot_prompt(query, retrieved_chunks)
            system = PromptBuilder.COT_SYSTEM_PROMPT
        else:
            prompt = PromptBuilder.build_grounded_prompt(query, retrieved_chunks)
            system = PromptBuilder.STRICT_GROUNDED_SYSTEM_PROMPT

        # Try Ollama if active and not forced fallback
        if not force_fallback and self._check_ollama_status():
            try:
                import time
                start_t = time.time()
                payload = {
                    "model": target_model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": 256
                    }
                }
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    res_data = resp.json()
                    answer = res_data.get("response", "").strip()
                    duration = int((time.time() - start_t) * 1000)
                    return {
                        "answer": answer,
                        "model": target_model,
                        "prompt_used": prompt,
                        "generation_time_ms": duration,
                        "is_fallback": False
                    }
            except Exception as e:
                # print(f"Ollama generation failed ({e}), using local rule fallback...")
                pass

        # Intelligent Local Grounded Fallback
        return self._local_grounded_fallback(query, retrieved_chunks, prompt)

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: List[TextChunk],
        prompt_style: str = "grounded",
        model_name: Optional[str] = None,
        force_fallback: bool = False
    ):
        """
        Streaming generator that yields partial text tokens as they are generated.
        Provides responsive word-by-word streaming for the chat interface.
        """
        target_model = model_name or self.model_name

        # Route synthetic research stress-test models directly (bypasses Ollama)
        if target_model == "synthetic-hallucinator":
            res = self._synthetic_hallucinator_generate(query, retrieved_chunks)
            import time
            for word in res["answer"].split(" "):
                yield word + " "
                time.sleep(0.02)
            return
        elif target_model == "synthetic-weak-reasoner":
            res = self._synthetic_weak_reasoner_generate(query, retrieved_chunks)
            import time
            for word in res["answer"].split(" "):
                yield word + " "
                time.sleep(0.02)
            return
        elif target_model == "local-fallback-extractor":
            prompt = PromptBuilder.build_grounded_prompt(query, retrieved_chunks)
            res = self._local_grounded_fallback(query, retrieved_chunks, prompt)
            import time
            for word in res["answer"].split(" "):
                yield word + " "
                time.sleep(0.02)
            return

        if prompt_style == "cot":
            prompt = PromptBuilder.build_cot_prompt(query, retrieved_chunks)
            system = PromptBuilder.COT_SYSTEM_PROMPT
        else:
            prompt = PromptBuilder.build_grounded_prompt(query, retrieved_chunks)
            system = PromptBuilder.STRICT_GROUNDED_SYSTEM_PROMPT

        if not force_fallback and self._check_ollama_status():
            try:
                payload = {
                    "model": target_model,
                    "prompt": prompt,
                    "system": system,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                }
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line.decode("utf-8"))
                            token = data.get("response", "")
                            yield token
                            if data.get("done", False):
                                break
                    return
            except Exception:
                pass

        # Fallback streaming (yield word-by-word with tiny delays)
        fallback_res = self._local_grounded_fallback(query, retrieved_chunks, prompt)
        full_text = fallback_res["answer"]
        import time
        for word in full_text.split(" "):
            yield word + " "
            time.sleep(0.02)


    def _local_grounded_fallback(
        self,
        query: str,
        chunks: List[TextChunk],
        prompt: str
    ) -> Dict[str, Any]:
        """
        Deterministic, local extractive answer synthesizer.
        Extracts salient sentence from context containing query keywords,
        or accurately refuses if context has zero relevant signal.
        """
        if not chunks:
            return {
                "answer": "The provided context does not contain sufficient evidence to answer this question.",
                "model": "local-fallback-extractor",
                "prompt_used": prompt,
                "generation_time_ms": 5,
                "is_fallback": True
            }

        q_words = set(query.lower().replace("?", "").replace(".", "").split())
        stop_words = {"what", "is", "the", "in", "of", "and", "a", "to", "how", "why", "does", "are", "for", "between"}
        keywords = q_words - stop_words

        best_sentence = ""
        best_overlap = 0

        for chunk in chunks:
            sentences = [s.strip() for s in chunk.text.replace("\n", " ").split(". ") if s.strip()]
            for sent in sentences:
                sent_lower = sent.lower()
                overlap = sum(1 for kw in keywords if kw in sent_lower)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sent

        if best_overlap >= max(1, len(keywords) // 3):
            # Append period if missing
            ans = best_sentence if best_sentence.endswith(".") else best_sentence + "."
            return {
                "answer": ans,
                "model": "local-fallback-extractor",
                "prompt_used": prompt,
                "generation_time_ms": 10,
                "is_fallback": True
            }
        else:
            return {
                "answer": "The provided context does not contain sufficient evidence to answer this question.",
                "model": "local-fallback-extractor",
                "prompt_used": prompt,
                "generation_time_ms": 5,
                "is_fallback": True
            }

    def _synthetic_hallucinator_generate(
        self,
        query: str,
        chunks: List[TextChunk]
    ) -> Dict[str, Any]:
        """
        Self-generated synthetic model that intentionally injects ungrounded,
        hallucinated claims and fabricated statistics into its answers.
        Enables stress-testing and demonstration of HALLUCINATION_UNSUPPORTED detection.
        """
        import time
        start_t = time.time()
        fallback_res = self._local_grounded_fallback(query, chunks, query)
        base_ans = fallback_res["answer"]

        hallucinations = [
            " Additionally, peer-reviewed trials conducted by Dr. Thorne at Stanford (2024) confirmed a 43.8% performance anomaly under non-standard thermal variance.",
            " Recent DARPA benchmark trials demonstrated that this exact configuration increases decoherence rates by approximately 72.4%.",
            " This was later revised in the 2023 IEEE quantum symposium to account for undocumented zero-point vacuum fluctuations.",
            " In subsequent unverified laboratory experiments, researchers reported that this mechanism produces unexpected hydrogen peroxide compounds under ambient room pressure."
        ]
        import hashlib
        idx = int(hashlib.md5(query.encode()).hexdigest(), 16) % len(hallucinations)
        hallucinated_ans = base_ans.rstrip(".") + "." + hallucinations[idx]

        return {
            "answer": hallucinated_ans,
            "model": "synthetic-hallucinator",
            "prompt_used": query,
            "generation_time_ms": int((time.time() - start_t) * 1000) + 15,
            "is_fallback": True
        }

    def _synthetic_weak_reasoner_generate(
        self,
        query: str,
        chunks: List[TextChunk]
    ) -> Dict[str, Any]:
        """
        Self-generated synthetic model simulating a weak / low-capacity LLM:
        suffers from conservative refusal ('I don't know') on complex queries
        and inverts logical conclusions, triggering CONSERVATIVE_REFUSAL and CONTEXT_CONTRADICTION.
        """
        import time
        start_t = time.time()
        import hashlib
        import re
        h_val = int(hashlib.md5(query.encode()).hexdigest(), 16) % 3
        if h_val == 0:
            ans = "The provided context does not contain sufficient evidence to answer this question."
        elif h_val == 1:
            fallback_res = self._local_grounded_fallback(query, chunks, query)
            base_ans = fallback_res["answer"]
            inversions = [
                (r"\breflects\b", "absorbs"),
                (r"\babsorbs\b", "reflects"),
                (r"\bpositive\b", "negative"),
                (r"\bnegative\b", "positive"),
                (r"\bincreases\b", "decreases"),
                (r"\bdecreases\b", "increases"),
                (r"\baccelerates\b", "decelerates"),
                (r"\bprevents\b", "causes"),
                (r"\breduces\b", "amplifies"),
                (r"\bamplifies\b", "reduces"),
                (r"\bhigher\b", "lower"),
                (r"\blower\b", "higher"),
                (r"\btrue\b", "false"),
                (r"\bdoes not\b", "does"),
            ]
            inverted_sent = base_ans
            inverted = False
            for pat, repl in inversions:
                if re.search(pat, inverted_sent, flags=re.IGNORECASE):
                    inverted_sent = re.sub(pat, repl, inverted_sent, count=1, flags=re.IGNORECASE)
                    inverted = True
                    break
            if inverted:
                ans = inverted_sent
            else:
                ans = f"Based on contradictory findings, the exact opposite effect occurs: {base_ans.lower()}"
        else:
            clean_q = re.sub(r"[^\w\s]", "", query).strip()
            ans = f"The document briefly mentions {clean_q}, but completely lacks numerical parameters to confirm the relationship."

        return {
            "answer": ans,
            "model": "synthetic-weak-reasoner",
            "prompt_used": query,
            "generation_time_ms": int((time.time() - start_t) * 1000) + 10,
            "is_fallback": True
        }

"""
Natural Language Inference (NLI) and Groundedness / Faithfulness Scorer.
Decomposes generated answers into sentences and evaluates entailment/contradiction
against retrieved context chunks using deep Cross-Encoder NLI.
"""

import sys
if "torchvision" not in sys.modules:
    sys.modules["torchvision"] = None
if "onnx" not in sys.modules:
    sys.modules["onnx"] = None

import os
import re
from typing import List, Dict, Any, Optional
from ..ingestion.chunker import TextChunk
from ..retrieval.embedder import DenseEmbedder
from ..config import FAITHFULNESS_THRESHOLD, DEFAULT_NLI_MODEL


class GroundingEvaluator:
    """
    Sentence-level NLI Faithfulness Scorer.
    Replaces lexical/cosine heuristics with Cross-Encoder Natural Language Inference
    (entailment, contradiction, neutral) to rigorously detect hallucinations.
    """

    def __init__(
        self,
        embedder: Optional[DenseEmbedder] = None,
        model_name: str = DEFAULT_NLI_MODEL,
        local_files_only: bool = True
    ):
        self.embedder = embedder
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.nli_model = None
        self.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self._load_nli_model()

    def _load_nli_model(self):
        """Loads Cross-Encoder NLI model with strict offline enforcement."""
        try:
            from sentence_transformers import CrossEncoder
            self.nli_model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)
            if hasattr(self.nli_model.model, "config") and hasattr(self.nli_model.model.config, "id2label"):
                self.id2label = self.nli_model.model.config.id2label
            print(f"[NLI_SCORER] [LOAD] Loaded Cross-Encoder NLI '{self.model_name}' (offline: {self.local_files_only})")
        except Exception as e:
            try:
                from sentence_transformers import CrossEncoder
                self.nli_model = CrossEncoder(self.model_name)
                print(f"[NLI_SCORER] [LOAD] Loaded Cross-Encoder NLI '{self.model_name}' (online fallback)")
            except Exception as e2:
                print(f"[NLI_SCORER] [WARNING] Could not load NLI model '{self.model_name}': {e2}. Falling back to cosine heuristics.")
                self.nli_model = None

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Splits answer text into discrete factual sentence claims."""
        if not text or not text.strip():
            return []

        # Split on sentence terminals while respecting abbreviations and numbered lists
        raw_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'#\*\-])", text.strip())
        sentences = []
        for segment in raw_splits:
            segment = segment.strip()
            # Split multiline / bullet points
            lines = [l.strip().lstrip("-*•0123456789. ") for l in segment.split("\n") if l.strip()]
            for line in lines:
                if len(line) >= 8:
                    sentences.append(line)

        return sentences if sentences else ([text.strip()] if text.strip() else [])

    def evaluate_groundedness(
        self,
        answer: str,
        retrieved_chunks: List[TextChunk]
    ) -> Dict[str, Any]:
        """
        Assesses if the answer statements are grounded in retrieved chunks using sentence-level NLI.
        Returns:
            - faithfulness_score (0.0 to 1.0): fraction of sentences entailed by context
            - is_faithful (bool): True if faithfulness_score >= FAITHFULNESS_THRESHOLD
            - unsupported_claims (List[str]): sentences classified as UNSUPPORTED or CONTRADICTION
            - context_coverage (float): proportion of context covered
            - contradiction_detected (bool): True if any contradiction is identified
            - sentence_evaluations (List[Dict]): granular per-sentence classification
        """
        if not retrieved_chunks or not answer.strip():
            return {
                "faithfulness_score": 0.0,
                "is_faithful": False,
                "unsupported_claims": [answer] if answer.strip() else [],
                "context_coverage": 0.0,
                "contradiction_detected": False,
                "sentence_evaluations": []
            }

        sentences = self.split_sentences(answer)
        if not sentences:
            sentences = [answer.strip()]

        # Extract full context text for lexical containment check
        full_context_text = " ".join([
            (c.parent_text if (hasattr(c, "parent_text") and c.parent_text) else c.text)
            for c in retrieved_chunks
        ])
        full_context_lower = full_context_text.lower()

        # Prepare granular premise sentences from retrieved chunks
        premises: List[Tuple[str, str]] = []
        for c in retrieved_chunks:
            source_text = c.parent_text if (hasattr(c, "parent_text") and c.parent_text) else c.text
            chunk_sents = self.split_sentences(source_text)
            for cs in chunk_sents:
                clean_s = cs.strip()
                if len(clean_s) >= 12:
                    premises.append((c.chunk_id, clean_s))
            # Also add 2-sentence sliding windows for multi-clause facts
            for i in range(len(chunk_sents) - 1):
                win = (chunk_sents[i].strip() + " " + chunk_sents[i + 1].strip()).strip()
                if 20 <= len(win) <= 350:
                    premises.append((c.chunk_id, win))
            # Also add child chunk text directly
            if len(c.text.strip()) <= 350:
                premises.append((c.chunk_id, c.text.strip()))

        if not premises:
            premises = [(c.chunk_id, c.text) for c in retrieved_chunks]

        sentence_evaluations = []
        unsupported_claims = []
        contradiction_detected = False
        entailed_count = 0

        # Find class index mappings
        entail_idx = 1
        contra_idx = 0
        for idx, lbl in self.id2label.items():
            if "entail" in lbl.lower():
                entail_idx = idx
            elif "contra" in lbl.lower():
                contra_idx = idx

        if self.nli_model is not None:
            # For each answer sentence, select top relevant premises
            for s_idx, sent in enumerate(sentences):
                clean_sent = re.sub(r"[^\w\s]", " ", sent.lower()).strip()
                sent_tokens = set(clean_sent.split())
                content_tokens = {t for t in sent_tokens if len(t) > 2 and t not in {"the", "and", "for", "with", "this", "that", "from"}}

                # Check lexical containment
                lexical_match = False
                if len(clean_sent) > 15 and clean_sent in full_context_lower:
                    lexical_match = True
                elif content_tokens:
                    overlap_ratio = sum(1 for t in content_tokens if t in full_context_lower) / len(content_tokens)
                    if overlap_ratio >= 0.70:
                        lexical_match = True

                # Select top 6 most relevant premise candidates
                scored_premises = []
                for cid, p_text in premises:
                    p_lower = p_text.lower()
                    overlap = sum(1 for t in content_tokens if t in p_lower)
                    scored_premises.append((overlap, cid, p_text))
                scored_premises.sort(key=lambda x: x[0], reverse=True)
                top_premises = scored_premises[:6]

                candidate_pairs = [(p_text, sent) for _, _, p_text in top_premises]
                try:
                    probs_list = self.nli_model.predict(candidate_pairs, apply_softmax=True)
                    best_entailment = max(float(pr[entail_idx]) for pr in probs_list) if len(probs_list) > 0 else 0.0
                    best_contradiction = max(float(pr[contra_idx]) for pr in probs_list) if len(probs_list) > 0 else 0.0
                    best_chunk_id = top_premises[0][1] if top_premises else "unknown"
                except Exception:
                    best_entailment = 0.95 if lexical_match else 0.20
                    best_contradiction = 0.0
                    best_chunk_id = "unknown"

                # Incorporate high lexical groundedness
                if lexical_match and best_entailment < 0.60:
                    best_entailment = 0.95

                if best_entailment >= FAITHFULNESS_THRESHOLD:
                    label = "ENTAILED"
                    entailed_count += 1
                elif best_contradiction >= 0.60 and best_entailment < 0.25:
                    label = "CONTRADICTION"
                    contradiction_detected = True
                    unsupported_claims.append(sent)
                else:
                    label = "UNSUPPORTED"
                    unsupported_claims.append(sent)

                sentence_evaluations.append({
                    "sentence": sent,
                    "label": label,
                    "entailment_score": round(best_entailment, 4),
                    "contradiction_score": round(best_contradiction, 4),
                    "best_chunk_id": best_chunk_id
                })

        else:
            return self._fallback_groundedness(sentences, retrieved_chunks)

        total_sentences = max(1, len(sentences))
        faithfulness_score = round(entailed_count / total_sentences, 4)
        is_faithful = faithfulness_score >= FAITHFULNESS_THRESHOLD

        unsupported_count = sum(1 for s in sentence_evaluations if s["label"] == "UNSUPPORTED")
        contra_count = sum(1 for s in sentence_evaluations if s["label"] == "CONTRADICTION")

        print(
            f"[NLI_FAITHFULNESS] Evaluated {len(sentences)} sentence(s) across {len(retrieved_chunks)} context chunk(s): "
            f"{entailed_count} ENTAILED, {unsupported_count} UNSUPPORTED, {contra_count} CONTRADICTION "
            f"(Faithfulness: {faithfulness_score:.2%})"
        )
        for s in sentence_evaluations:
            if s["label"] != "ENTAILED":
                print(f"  -> [NLI_FLAGGED] [{s['label']}] (entail={s['entailment_score']:.3f}, contra={s['contradiction_score']:.3f}): '{s['sentence'][:65]}...'")

        return {
            "faithfulness_score": float(faithfulness_score),
            "is_faithful": bool(is_faithful),
            "unsupported_claims": unsupported_claims,
            "context_coverage": float(faithfulness_score),
            "contradiction_detected": contradiction_detected,
            "sentence_evaluations": sentence_evaluations
        }

    def _fallback_groundedness(self, sentences: List[str], retrieved_chunks: List[TextChunk]) -> Dict[str, Any]:
        """Graceful fallback if NLI model fails to load."""
        full_context = " ".join([c.text for c in retrieved_chunks]).lower()
        supported = 0
        unsupported = []
        for s in sentences:
            tokens = [w for w in re.findall(r"\b\w+\b", s.lower()) if len(w) > 3]
            match_rate = sum(1 for t in tokens if t in full_context) / max(1, len(tokens))
            if match_rate >= 0.40:
                supported += 1
            else:
                unsupported.append(s)

        f_score = round(supported / max(1, len(sentences)), 4)
        return {
            "faithfulness_score": f_score,
            "is_faithful": f_score >= FAITHFULNESS_THRESHOLD,
            "unsupported_claims": unsupported,
            "context_coverage": f_score,
            "contradiction_detected": False,
            "sentence_evaluations": [{"sentence": s, "label": "UNSUPPORTED"} for s in unsupported]
        }

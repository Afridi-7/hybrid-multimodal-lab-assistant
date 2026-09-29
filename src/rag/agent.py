"""
Clinical reasoning agent (LangChain ReAct).

This is the *agentic* alternative to ``ClinicalReasoner`` (single-shot).
It exposes a small set of clinically-relevant tools that the LLM can call
iteratively before producing the structured final report:

  1. ``query_knowledge_base(query)``       — pulls passages from ChromaDB.
  2. ``lookup_lab_reference_ranges(name)`` — adult CBC reference ranges.
  3. ``interpret_differential(diff_json)`` — flags abnormal WBC differentials.
  4. ``get_uncertainty_summary()``         — MC-Dropout summary for this case.
  5. ``get_detection_counts()``            — Stage-1 YOLOv8 cell counts.

The agent is intentionally constrained:

* It runs at most ``reasoning.agent.max_iterations`` ReAct steps.
* Its **final** assistant message MUST be a JSON object matching
  ``ClinicalReasoner``'s schema. We parse it and fall back to a plain-text
  interpretation only if parsing fails.
* All retrieval happens through the existing :class:`ClinicalRetriever`,
  so citations stay grounded in the same indexed corpus.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.config.config_loader import Config
from src.rag.retriever import ClinicalRetriever

logger = logging.getLogger(__name__)


# --- Adult reference ranges (educational; not for clinical use). -------------
# Sources: Williams Hematology (10th ed.); WHO laboratory manuals.
LAB_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "neutrophil": {"unit": "% of WBC", "low": 40, "high": 70,
                   "note": "↑ in bacterial infection / stress; ↓ in viral infection, sepsis, marrow failure."},
    "lymphocyte": {"unit": "% of WBC", "low": 20, "high": 40,
                   "note": "↑ in viral infection, CLL; ↓ in immunodeficiency, steroids."},
    "monocyte":   {"unit": "% of WBC", "low": 2,  "high": 8,
                   "note": "↑ in chronic inflammation, TB, CMML."},
    "eosinophil": {"unit": "% of WBC", "low": 1,  "high": 4,
                   "note": "↑ in allergy, parasites, hypereosinophilic syndromes."},
    "basophil":   {"unit": "% of WBC", "low": 0,  "high": 1,
                   "note": "Persistent basophilia raises concern for CML / MPN."},
    "ig":         {"unit": "% of WBC", "low": 0,  "high": 0.5,
                   "note": "Immature granulocytes >1% indicate left shift; consider infection or marrow process."},
    "wbc_count":   {"unit": "x10^9/L", "low": 4.0, "high": 11.0,
                    "note": "Microscope-field counts in this pipeline are NOT clinical concentrations."},
    "rbc_count":   {"unit": "x10^12/L", "low": 4.0, "high": 5.5,
                    "note": "Sex-dependent. Microscope-field counts are NOT clinical concentrations."},
    "platelet":    {"unit": "x10^9/L", "low": 150, "high": 450,
                    "note": "Microscope-field counts only — true thrombocytopenia requires CBC."},
}


SYSTEM_PROMPT = """You are a clinical hematology reasoning agent. Your role is to
produce an evidence-grounded, safety-aware interpretation of a peripheral blood
smear analysis.

You have access to tools. Use them deliberately:

  • `query_knowledge_base` — search the indexed hematology textbooks. Call it
    1-3 times with focused queries to ground every claim you make.
  • `lookup_lab_reference_ranges` — look up adult reference ranges before
    calling any value "elevated", "low", or "abnormal".
  • `interpret_differential` — pass the WBC differential JSON to flag
    abnormalities (left shift, lymphocytosis, etc.).
  • `get_uncertainty_summary` — review model uncertainty before issuing
    confident statements.
  • `get_detection_counts` — get Stage-1 YOLO counts for context.

CRITICAL SAFETY RULES:
  1. Cell counts here are PER-MICROSCOPE-FIELD, not clinical concentrations.
     Do NOT translate them into ×10⁹/L diagnoses.
  2. If MC-Dropout flags samples as HIGH uncertainty, set
     `requires_expert_review: true`.
  3. Cite tool evidence explicitly using "[Reference N]" where N is the
     reference id returned by `query_knowledge_base`.
  4. Provide DIFFERENTIAL diagnoses with rationale, never definitive ones.
  5. A differential is only meaningful when enough white cells were
     classified. If the number of classified WBCs is below the stated
     minimum, do NOT interpret the percentages as abnormal; say that the
     differential is not reliable and set `requires_expert_review: true`.
  6. If CBC laboratory findings are provided, reconcile or contrast them
     with the image-derived findings and note any discordance.

Once you have gathered enough evidence, produce your FINAL message as a
JSON object (no markdown, no preface) with EXACTLY these keys:

{
  "clinical_interpretation": "2-4 sentence overview",
  "key_findings": ["..."],
  "differential_diagnoses": ["Diagnosis [Reference N] — rationale", "..."],
  "recommendations": ["..."],
  "safety_flags": ["..."],
  "requires_expert_review": true|false,
  "confidence_assessment": "LOW"|"MEDIUM"|"HIGH"
}

The JSON object MUST be the entire content of your final assistant message.
"""


def _format_chunks(chunks: List[Dict[str, Any]], offset: int) -> str:
    """Render retrieved chunks as numbered references for the LLM."""
    if not chunks:
        return "[no passages retrieved]"
    parts = []
    for i, c in enumerate(chunks, start=offset + 1):
        src = c.get("source", "unknown")
        text = (c.get("text") or "").strip().replace("\n", " ")
        if len(text) > 600:
            text = text[:600] + "..."
        parts.append(f"[Reference {i}] ({src}) {text}")
    return "\n\n".join(parts)


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from an LLM final message."""
    if not text:
        return None
    # Strip common wrappers like ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} block.
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


class ClinicalReasoningAgent:
    """LangChain ReAct agent over the Stage-3 reasoning task."""

    def __init__(self, config: Config, retriever: ClinicalRetriever) -> None:
        self.config = config
        self.retriever = retriever

        self.model_name = config.get("llm.model_name", "gpt-4o")
        self.temperature = config.get("llm.temperature", 0.1)
        self.max_iterations = int(config.get("reasoning.agent.max_iterations", 6))
        self.return_steps = bool(config.get("reasoning.agent.return_intermediate_steps", True))

        api_key_env = config.get("llm.api_key_env_var", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found in environment variable: {api_key_env}"
            )

        self._llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
        )

        # Per-call mutable state. Guarded by a lock so the singleton agent stays
        # safe under serialised backend access.
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "detection_counts": {},
            "wbc_differential": {},
            "uncertainty_summary": {},
            "retrieved_references": [],  # accumulated across tool calls
        }

        self._tools = self._build_tools()
        self._agent = create_react_agent(self._llm, self._tools)

        logger.info(
            "ClinicalReasoningAgent initialised (model=%s, max_iter=%d, tools=%d)",
            self.model_name, self.max_iterations, len(self._tools),
        )

    # ------------------------------------------------------------------ tools
    def _build_tools(self):
        agent = self  # closure target

        @tool
        def query_knowledge_base(query: str) -> str:
            """Search the indexed hematology knowledge base for passages
            relevant to `query`. Returns numbered references the agent must
            cite as [Reference N]."""
            chunks = agent.retriever.retrieve(query)
            offset = len(agent._state["retrieved_references"])
            for i, c in enumerate(chunks, start=offset + 1):
                agent._state["retrieved_references"].append({
                    "reference_id": i,
                    "source": c.get("source", "unknown"),
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("score"),
                })
            return _format_chunks(chunks, offset)

        @tool
        def lookup_lab_reference_ranges(parameter: str) -> str:
            """Return the adult reference range for a CBC/differential parameter.
            Valid names: neutrophil, lymphocyte, monocyte, eosinophil, basophil,
            ig, wbc_count, rbc_count, platelet."""
            key = parameter.strip().lower().replace(" ", "_")
            entry = LAB_REFERENCE_RANGES.get(key)
            if not entry:
                return (f"No reference range available for '{parameter}'. "
                        f"Known: {sorted(LAB_REFERENCE_RANGES.keys())}.")
            return json.dumps({"parameter": key, **entry})

        @tool
        def interpret_differential(differential_json: str) -> str:
            """Given a JSON string of {wbc_subtype: percentage}, flag deviations
            from adult reference ranges (e.g. left shift, lymphocytosis)."""
            try:
                diff = json.loads(differential_json)
            except json.JSONDecodeError:
                return "Invalid JSON; expected an object like {\"neutrophil\": 65}."
            flags: List[str] = []
            for k, v in diff.items():
                key = k.lower()
                ref = LAB_REFERENCE_RANGES.get(key)
                if not ref:
                    flags.append(f"{k}: no reference available.")
                    continue
                if v < ref["low"]:
                    flags.append(f"{k} = {v}% (LOW; ref {ref['low']}–{ref['high']}%). {ref['note']}")
                elif v > ref["high"]:
                    flags.append(f"{k} = {v}% (HIGH; ref {ref['low']}–{ref['high']}%). {ref['note']}")
                else:
                    flags.append(f"{k} = {v}% (within reference {ref['low']}–{ref['high']}%).")
            if not flags:
                return "Differential is empty."
            return "\n".join(flags)

        @tool
        def get_uncertainty_summary() -> str:
            """Return MC-Dropout uncertainty statistics for this case."""
            return json.dumps(agent._state["uncertainty_summary"] or {})

        @tool
        def get_detection_counts() -> str:
            """Return Stage-1 YOLOv8 cell counts (WBC/RBC/Platelet) for this case."""
            return json.dumps(agent._state["detection_counts"] or {})

        return [
            query_knowledge_base,
            lookup_lab_reference_ranges,
            interpret_differential,
            get_uncertainty_summary,
            get_detection_counts,
        ]

    # --------------------------------------------------------------- public API
    def generate_reasoning(
        self,
        vision_summary: Dict[str, Any],
        retrieved_context: str,  # ignored — agent calls KB itself
        uncertainty_summary: Optional[Dict[str, Any]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,  # ignored
    ) -> Dict[str, Any]:
        """Drop-in replacement for ``ClinicalReasoner.generate_reasoning``."""
        del retrieved_context, retrieved_chunks  # Agent fetches its own evidence.

        with self._lock:
            self._state = {
                "detection_counts": vision_summary.get("cell_counts")
                                    or vision_summary.get("total_counts", {}),
                "wbc_differential": vision_summary.get("wbc_differential", {}),
                "uncertainty_summary": uncertainty_summary or {},
                "retrieved_references": [],
            }

            user_prompt = self._build_user_prompt(vision_summary, uncertainty_summary)
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            try:
                logger.info("Invoking ReAct agent (max_iter=%d)", self.max_iterations)
                final_state = self._agent.invoke(
                    {"messages": messages},
                    config={"recursion_limit": self.max_iterations * 2 + 4},
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent invocation failed: %s", exc)
                return self._fallback_response(str(exc))

            return self._postprocess(final_state)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _build_user_prompt(
        vision_summary: Dict[str, Any],
        uncertainty_summary: Optional[Dict[str, Any]],
    ) -> str:
        counts = vision_summary.get("cell_counts") or vision_summary.get("total_counts", {})
        diff = vision_summary.get("wbc_differential", {})
        flagged = (uncertainty_summary or {}).get("flagged_count", 0)
        total = (uncertainty_summary or {}).get("total_samples") \
            or (uncertainty_summary or {}).get("sample_count", 0)

        n_wbc = vision_summary.get("wbc_classified", total)
        min_wbc = vision_summary.get("min_wbc_for_differential")

        lines = [
            "A peripheral blood smear was analysed with the three-stage AI pipeline.\n",
            f"Stage-1 detection counts (per microscope field): {json.dumps(counts)}",
            f"Stage-2 WBC differential: {json.dumps(diff)}",
            f"Stage-2 MC-Dropout uncertainty: {flagged}/{total} crops flagged.",
        ]
        if min_wbc:
            enough = "sufficient" if n_wbc >= min_wbc else "NOT sufficient"
            lines.append(
                f"Number of white cells classified: {n_wbc} (minimum for a reliable "
                f"differential: {min_wbc}; this is {enough})."
            )

        cbc = vision_summary.get("cbc_report") or {}
        findings = cbc.get("findings") or []
        if findings:
            lines.append("")
            lines.append("CBC laboratory findings (second input modality):")
            if cbc.get("sex"):
                lines.append(f"  - Sex: {cbc['sex']}")
            for f in findings:
                ref = f.get("reference_range") or [None, None]
                lines.append(
                    f"  - {f.get('analyte')} = {f.get('value')} {f.get('unit', '')} "
                    f"(ref {ref[0]}-{ref[1]}): {f.get('label')} ({f.get('severity')})"
                )

        lines.append("")
        lines.append(
            "Use your tools to gather evidence, then return the structured JSON "
            "report described in the system instructions."
        )
        return "\n".join(lines)

    def _postprocess(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        messages = final_state.get("messages", [])
        final_text = ""
        if messages:
            last = messages[-1]
            final_text = getattr(last, "content", str(last)) or ""

        parsed = _extract_json_block(final_text) or {}
        report = {
            "clinical_interpretation": parsed.get("clinical_interpretation") or final_text[:500],
            "key_findings": parsed.get("key_findings", []),
            "differential_diagnoses": parsed.get("differential_diagnoses", []),
            "recommendations": parsed.get("recommendations", []),
            "safety_flags": list(parsed.get("safety_flags", [])),
            "confidence_assessment": parsed.get("confidence_assessment", "MEDIUM"),
            "requires_expert_review": bool(parsed.get("requires_expert_review", False)),
            "raw_response": final_text,
            "retrieved_references": list(self._state["retrieved_references"]),
            "reasoning_mode": "agent",
        }

        # Force-flag if MC-Dropout already flagged anything.
        if (self._state.get("uncertainty_summary") or {}).get("flagged_count", 0) > 0:
            report["requires_expert_review"] = True
            if "HIGH_UNCERTAINTY" not in report["safety_flags"]:
                report["safety_flags"].append("HIGH_UNCERTAINTY")

        if self.return_steps:
            report["agent_trace"] = self._extract_trace(messages)
        return report

    @staticmethod
    def _extract_trace(messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert LangChain message log into a compact ReAct trace for the UI."""
        trace: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.__class__.__name__.replace("Message", "").lower()
            content = getattr(msg, "content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    trace.append({
                        "type": "tool_call",
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?"),
                        "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                    })
            elif role == "tool":
                trace.append({
                    "type": "tool_result",
                    "name": getattr(msg, "name", "?"),
                    "content": (content[:600] + "...") if len(content) > 600 else content,
                })
            elif content and role in ("ai", "assistant"):
                trace.append({
                    "type": "thought",
                    "content": (content[:600] + "...") if len(content) > 600 else content,
                })
        return trace

    @staticmethod
    def _fallback_response(error: str) -> Dict[str, Any]:
        return {
            "clinical_interpretation":
                "Agentic reasoning could not be completed automatically.",
            "key_findings": [],
            "differential_diagnoses": [],
            "recommendations": [
                "Manual hematologist review required — agent execution failed."
            ],
            "safety_flags": ["AGENT_ERROR"],
            "confidence_assessment": "LOW",
            "requires_expert_review": True,
            "raw_response": "",
            "retrieved_references": [],
            "agent_trace": [],
            "reasoning_mode": "agent",
            "error": error,
        }

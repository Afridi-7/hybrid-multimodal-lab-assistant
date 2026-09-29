"""
LLM-based clinical reasoning module using OpenAI GPT-4o.

Provides evidence-grounded clinical reasoning with safety constraints and abstention logic.
"""

import logging
import json
import os
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import OpenAI

from src.config.config_loader import Config

logger = logging.getLogger(__name__)


class ClinicalReasoner:
    """LLM-based clinical reasoner with safety constraints."""
    
    def __init__(self, config: Config):
        """
        Initialize clinical reasoner.
        
        Args:
            config: Configuration object with LLM parameters
            
        Raises:
            ValueError: If OpenAI API key not found
        """
        self.config = config
        
        # Load configuration
        self.model_name = config.get('llm.model_name', 'gpt-4o')
        self.temperature = config.get('llm.temperature', 0.1)
        self.max_tokens = config.get('llm.max_tokens', 1500)
        
        # Safety settings
        self.enable_abstention = config.get('llm.safety.enable_abstention', True)
        self.propagate_uncertainty = config.get('llm.safety.propagate_vision_uncertainty', True)
        self.require_citations = config.get('llm.safety.require_citations', True)
        
        # Get API key
        api_key_env = config.get('llm.api_key_env_var', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found in environment variable: {api_key_env}\n"
                "Set one of the following:\n"
                "  Windows PowerShell: $env:OPENAI_API_KEY='your-key-here'\n"
                "  Windows CMD: set OPENAI_API_KEY=your-key-here\n"
                "  .env file in project root: OPENAI_API_KEY=your-key-here"
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        logger.info(f"ClinicalReasoner initialized (model={self.model_name})")
    
    def generate_reasoning(
        self,
        vision_summary: Dict[str, Any],
        retrieved_context: str,
        uncertainty_summary: Optional[Dict[str, Any]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate clinical reasoning from vision results and retrieved evidence.
        
        Args:
            vision_summary: Dictionary with detection and classification results
            retrieved_context: Formatted retrieved knowledge chunks with citations
            uncertainty_summary: Optional uncertainty metrics from classification
            
        Returns:
            Dictionary with clinical reasoning:
                {
                    'clinical_interpretation': str,
                    'key_findings': List[str],
                    'differential_diagnoses': List[str],
                    'recommendations': List[str],
                    'safety_flags': List[str],
                    'citations_used': List[int],
                    'confidence_assessment': str,
                    'requires_expert_review': bool,
                    'raw_response': str
                }
                
        Examples:
            >>> reasoner = ClinicalReasoner(config)
            >>> result = reasoner.generate_reasoning(
            ...     vision_summary={'cell_counts': {...}, 'wbc_differential': {...}},
            ...     retrieved_context=formatted_context,
            ...     uncertainty_summary={'flagged_count': 2}
            ... )
            >>> print(result['clinical_interpretation'])
        """
        # Build prompt
        prompt = self._build_prompt(vision_summary, retrieved_context, uncertainty_summary)
        
        # Call OpenAI API
        try:
            response = self._call_openai(prompt)
            
            # Parse response
            parsed = self._parse_response(response, uncertainty_summary, retrieved_chunks)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return self._create_fallback_response(str(e), uncertainty_summary)
    
    def _build_prompt(
        self,
        vision_summary: Dict[str, Any],
        retrieved_context: str,
        uncertainty_summary: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for clinical reasoning."""
        # Format vision results
        vision_text = self._format_vision_summary(vision_summary)
        
        # Add uncertainty information if available
        uncertainty_text = ""
        if uncertainty_summary and self.propagate_uncertainty:
            uncertainty_text = self._format_uncertainty_summary(uncertainty_summary)
        
        # Build system prompt
        system_prompt = """You are a clinical hematology expert assistant. Your role is to provide evidence-based interpretation of blood smear analysis results.

CRITICAL SAFETY REQUIREMENTS:
1. All claims MUST be supported by provided reference materials
2. Cite references explicitly using [Reference N] notation
3. If evidence is insufficient or ambiguous, clearly state this limitation
4. Do NOT make definitive diagnoses - provide differential diagnoses and recommend expert review
5. Highlight any uncertainty flags from the vision analysis
6. Be conservative and prioritize patient safety
7. CROSS-MODAL REASONING: when CBC laboratory findings are provided alongside the image-derived WBC differential, explicitly reconcile or contrast the two modalities (e.g. "image shows neutrophil predominance AND CBC reports leukocytosis → consistent with absolute neutrophilia"). Note any discordance.
8. CELL COUNT: if fewer white cells were classified than the stated minimum, do NOT interpret the differential percentages as abnormal; state that the differential is not reliable and set requires_expert_review to true.

OUTPUT FORMAT:
Provide your response as a JSON object with these fields:
{
    "clinical_interpretation": "Brief overview (2-3 sentences)",
    "key_findings": ["Finding 1", "Finding 2", ...],
    "differential_diagnoses": ["Diagnosis 1 [Reference N]", "Diagnosis 2 [Reference M]", ...],
    "recommendations": ["Recommendation 1", "Recommendation 2", ...],
    "safety_flags": ["Flag 1", "Flag 2", ...],
    "citations_used": [1, 2, 3, ...],
    "confidence_assessment": "LOW|MEDIUM|HIGH",
    "requires_expert_review": true/false
}"""
        
        # Build user prompt
        user_prompt = f"""**BLOOD SMEAR ANALYSIS RESULTS:**

{vision_text}

{uncertainty_text}

**RETRIEVED MEDICAL REFERENCES:**

{retrieved_context if retrieved_context else "[No relevant references found]"}

---

Please provide your clinical interpretation following the safety requirements and JSON format specified."""
        
        return f"{system_prompt}\n\n{user_prompt}"
    
    def _format_vision_summary(self, vision_summary: Dict[str, Any]) -> str:
        """Format vision summary for prompt."""
        lines = []
        
        # Cell counts
        if 'cell_counts' in vision_summary or 'total_counts' in vision_summary:
            counts = vision_summary.get('cell_counts') or vision_summary.get('total_counts', {})
            lines.append("**Cell Counts:**")
            for cell_type, count in counts.items():
                lines.append(f"  - {cell_type}: {count}")
            lines.append("")
        
        # WBC differential
        if 'wbc_differential' in vision_summary:
            diff = vision_summary['wbc_differential']
            lines.append("**WBC Differential:**")
            for subtype, percentage in diff.items():
                lines.append(f"  - {subtype}: {percentage}%")
            min_wbc = vision_summary.get('min_wbc_for_differential')
            if min_wbc:
                n_wbc = vision_summary.get('wbc_classified', 0)
                enough = "sufficient" if n_wbc >= min_wbc else "NOT sufficient"
                lines.append(
                    f"  - White cells classified: {n_wbc} (minimum for a reliable "
                    f"differential: {min_wbc}; {enough})"
                )
            lines.append("")
        
        # Batch statistics (if available)
        if 'cell_count_stats' in vision_summary:
            stats = vision_summary['cell_count_stats']
            lines.append("**Cell Count Statistics (Mean ± Variance):**")
            for cell_type in stats.get('mean', {}).keys():
                mean = stats['mean'].get(cell_type, 0)
                var = stats['variance'].get(cell_type, 0)
                lines.append(f"  - {cell_type}: {mean:.1f} ± {var:.1f}")
            lines.append("")

        # Multimodal: CBC tabular findings
        if 'cbc_report' in vision_summary and vision_summary['cbc_report']:
            cbc = vision_summary['cbc_report']
            findings = cbc.get('findings', [])
            if findings:
                lines.append("**CBC Laboratory Findings (tabular modality):**")
                if cbc.get('sex'):
                    lines.append(f"  - Sex: {cbc['sex']}")
                lines.append(
                    f"  - Abnormal analytes: {cbc.get('abnormal_count', 0)} of {len(findings)}"
                )
                for f in findings:
                    marker = "⚠️ " if f.get('direction') != 'normal' else "  "
                    ref = f.get('reference_range', [None, None])
                    lines.append(
                        f"{marker}{f.get('analyte')} = {f.get('value')} {f.get('unit', '')} "
                        f"(ref {ref[0]}–{ref[1]}) → {f.get('label')} ({f.get('severity')})"
                    )
                lines.append("")

        return "\n".join(lines)
    
    def _format_uncertainty_summary(self, uncertainty_summary: Dict[str, Any]) -> str:
        """Format uncertainty summary for prompt."""
        lines = ["**UNCERTAINTY ANALYSIS:**"]
        
        flagged = uncertainty_summary.get('flagged_count', 0)
        total = uncertainty_summary.get('total_samples', 0)
        
        if flagged:
            lines.append(f"  - Cells flagged for review: {flagged}/{total}")
        
        if 'uncertainty_distribution' in uncertainty_summary:
            dist = uncertainty_summary['uncertainty_distribution']
            lines.append(f"  - Uncertainty levels: LOW={dist.get('LOW', 0)}, MEDIUM={dist.get('MEDIUM', 0)}, HIGH={dist.get('HIGH', 0)}")
        
        if 'mean_confidence' in uncertainty_summary:
            lines.append(f"  - Mean classification confidence: {uncertainty_summary['mean_confidence']:.2%}")
        
        if flagged > 0:
            lines.append("\n⚠️ **HIGH UNCERTAINTY DETECTED** - Some cells have ambiguous morphology requiring expert review")
        
        lines.append("")
        return "\n".join(lines)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with error handling."""
        logger.info(f"Calling OpenAI API (model={self.model_name})")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            logger.info(f"Received response ({len(content)} chars)")
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_response(
        self,
        response: str,
        uncertainty_summary: Optional[Dict[str, Any]],
        retrieved_chunks: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        try:
            parsed = json.loads(response)
            
            # Validate required fields
            parsed.setdefault('clinical_interpretation', '')
            parsed.setdefault('key_findings', [])
            parsed.setdefault('differential_diagnoses', [])
            parsed.setdefault('recommendations', [])
            
            # Add safety flags if not present
            if 'safety_flags' not in parsed:
                parsed['safety_flags'] = []

            # Normalize list fields to prevent downstream schema drift.
            for list_field in ['key_findings', 'differential_diagnoses', 'recommendations', 'safety_flags']:
                if not isinstance(parsed.get(list_field), list):
                    parsed[list_field] = [str(parsed[list_field])]
            
            # Check for insufficient evidence
            if not parsed.get('differential_diagnoses') or len(response) < 100:
                parsed['safety_flags'].append('INSUFFICIENT_EVIDENCE')
            
            # Propagate uncertainty flags
            if uncertainty_summary and uncertainty_summary.get('flagged_count', 0) > 0:
                if 'HIGH_UNCERTAINTY' not in parsed['safety_flags']:
                    parsed['safety_flags'].append('HIGH_UNCERTAINTY')
                parsed['requires_expert_review'] = True

            # Keep citations strict and bounded by retrieval results.
            parsed['citations_used'] = self._normalize_citations(parsed, response, retrieved_chunks)

            if self.require_citations and retrieved_chunks and not parsed['citations_used']:
                if 'INSUFFICIENT_EVIDENCE' not in parsed['safety_flags']:
                    parsed['safety_flags'].append('INSUFFICIENT_EVIDENCE')
                parsed['requires_expert_review'] = True
            
            # Add metadata
            parsed['raw_response'] = response
            parsed['timestamp'] = datetime.now().isoformat()
            parsed['model'] = self.model_name
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            
            return {
                'clinical_interpretation': response[:500],
                'key_findings': [],
                'differential_diagnoses': [],
                'recommendations': ['Manual review required due to parsing error'],
                'safety_flags': ['NON_JSON_RESPONSE'],
                'citations_used': [],
                'confidence_assessment': 'LOW',
                'requires_expert_review': True,
                'raw_response': response,
                'error': str(e)
            }

    def _normalize_citations(
        self,
        parsed: Dict[str, Any],
        raw_response: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]]
    ) -> List[int]:
        """Extract and validate citation ids from structured output and text."""
        max_ref = len(retrieved_chunks or [])
        if max_ref <= 0:
            return []

        citations = parsed.get('citations_used', [])
        normalized: List[int] = []
        if isinstance(citations, list):
            for item in citations:
                try:
                    value = int(item)
                    if 1 <= value <= max_ref and value not in normalized:
                        normalized.append(value)
                except (TypeError, ValueError):
                    continue

        if not normalized:
            matches = re.findall(r'\[Reference\s+(\d+)\]', raw_response)
            for match in matches:
                value = int(match)
                if 1 <= value <= max_ref and value not in normalized:
                    normalized.append(value)

        return normalized
    
    def _create_fallback_response(
        self,
        error_message: str,
        uncertainty_summary: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create fallback response on error."""
        safety_flags = ['LLM_ERROR']

        flagged = 0
        if uncertainty_summary:
            flagged = uncertainty_summary.get('flagged_count',
                       uncertainty_summary.get('flagged_samples', 0)) or 0
        if flagged > 0:
            safety_flags.append('HIGH_UNCERTAINTY')
        
        return {
            'clinical_interpretation': 'Unable to generate automated interpretation due to system error.',
            'key_findings': [],
            'differential_diagnoses': [],
            'recommendations': [
                'Manual expert review required',
                'System encountered an error during automated analysis'
            ],
            'safety_flags': safety_flags,
            'citations_used': [],
            'confidence_assessment': 'LOW',
            'requires_expert_review': True,
            'raw_response': '',
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }

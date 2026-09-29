"""
Main pipeline orchestrator for Hybrid Multimodal Lab Assistant system.

Coordinates all three stages: Detection â†’ Classification â†’ RAG Reasoning
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import json
import numpy as np

from src.config.config_loader import Config, load_config
from src.detection.detector import CellDetector
from src.classification.classifier import WBCClassifier
from src.rag.retriever import ClinicalRetriever
from src.rag.llm_reasoner import ClinicalReasoner
from src.multimodal.cbc_analyzer import CBCInput, CBCReport, analyze_cbc
from src.utils.logging_config import setup_logging
from src.utils.pipeline_helpers import collect_wbc_crops
from src.utils.safety_rules import DEFAULT_MIN_WBC_FOR_DIFFERENTIAL, apply_minimum_cell_rule
from src.utils.reproducibility import set_global_seeds

logger = logging.getLogger(__name__)


class BloodSmearPipeline:
    """End-to-end pipeline for blood smear analysis with clinical reasoning."""
    
    def __init__(self, config: Optional[Config] = None, config_path: Optional[str] = None):
        """
        Initialize pipeline with all components.
        
        Args:
            config: Configuration object (if None, loads from config_path)
            config_path: Path to config.yaml (if config not provided)
            
        Examples:
            >>> pipeline = BloodSmearPipeline()
            >>> results = pipeline.analyze('blood_smear.jpg')
        """
        # Load configuration
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        # Setup logging
        log_level = config.get('logging.level', 'INFO')
        setup_logging(log_level=log_level)

        # Seed RNGs as early as possible for reproducible MC-Dropout / retrieval runs.
        set_global_seeds(
            seed=config.get('system.random_seed'),
            deterministic_torch=config.get('system.deterministic_torch', False),
        )

        logger.info("Initializing Blood Smear Pipeline...")
        
        # Initialize components
        self.enable_stage1 = config.get('pipeline.enable_stage1', True)
        self.enable_stage2 = config.get('pipeline.enable_stage2', True)
        self.enable_stage3 = config.get('pipeline.enable_stage3', True)
        self.continue_on_error = config.get('pipeline.continue_on_error', True)
        self.save_intermediate = config.get('pipeline.save_intermediate_results', True)
        self.enable_gradcam = config.get('classification.gradcam.enabled', False)
        self.reasoning_mode = config.get('reasoning.mode', 'linear')  # 'linear' or 'agent'
        self.min_wbc_for_differential = int(config.get(
            'llm.safety.min_wbc_for_differential', DEFAULT_MIN_WBC_FOR_DIFFERENTIAL))
        
        # Initialize detectors
        self.detector = CellDetector(config) if self.enable_stage1 else None
        self.classifier = WBCClassifier(config) if self.enable_stage2 else None
        
        # Initialize RAG components
        if self.enable_stage3:
            self.retriever = ClinicalRetriever(config)
            self.reasoner = ClinicalReasoner(config)

            # Build retrieval index
            logger.info("Building RAG retrieval index...")
            index_stats = self.retriever.build_index()
            logger.info(f"Index built: {index_stats['total_chunks']} chunks, type={index_stats['index_type']}")

            # Optional ReAct agent (lazy import â€” only if requested).
            self.agent = None
            if self.reasoning_mode == 'agent':
                from src.rag.agent import ClinicalReasoningAgent
                logger.info("Initializing LangChain ReAct reasoning agent...")
                self.agent = ClinicalReasoningAgent(config, self.retriever)
        else:
            self.retriever = None
            self.reasoner = None
            self.agent = None
        
        # Output directories â€” resolve relative paths against the repo root so
        # results/figures land in the canonical top-level folders regardless of
        # the process cwd (e.g. uvicorn started from backend/).
        _repo_root = Path(__file__).resolve().parent.parent
        self.results_dir = Path(config.get('pipeline.output.results_dir', 'results'))
        if not self.results_dir.is_absolute():
            self.results_dir = _repo_root / self.results_dir
        self.figures_dir = Path(config.get('pipeline.output.figures_dir', 'figures'))
        if not self.figures_dir.is_absolute():
            self.figures_dir = _repo_root / self.figures_dir
        self.results_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)
        
        logger.info("Pipeline initialized successfully")
    
    def analyze(
        self,
        image_input: Union[str, Path, List[Union[str, Path]]],
        save_results: bool = True,
        cbc: Optional[Union[CBCInput, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete analysis pipeline on blood smear image(s).
        
        Args:
            image_input: Single image path or list of image paths
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with complete analysis results:
                {
                    'stage1_detection': {...},
                    'stage2_classification': {...},
                    'stage3_reasoning': {...},
                    'metadata': {...}
                }
                
        Examples:
            >>> pipeline = BloodSmearPipeline()
            >>> results = pipeline.analyze('smear.jpg')
            >>> print(results['stage3_reasoning']['clinical_interpretation'])
        """
        logger.info(f"Starting pipeline analysis...")
        start_time = datetime.now()
        
        results = {
            'metadata': {
                'timestamp': start_time.isoformat(),
                'input': str(image_input),
                'pipeline_version': '1.0.0'
            }
        }

        # Multimodal: analyse CBC tabular input (if provided) before Stage 3.
        cbc_report: Optional[CBCReport] = None
        if cbc is not None:
            try:
                cbc_report = analyze_cbc(cbc)
                results['cbc_report'] = cbc_report.to_dict()
                results['metadata']['modalities'] = ['image', 'tabular_cbc']
                logger.info(
                    "CBC modality: %d findings (%d abnormal)",
                    len(cbc_report.findings),
                    cbc_report.abnormal_count,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("CBC analysis failed, continuing without it: %s", exc)
                cbc_report = None
        else:
            results['metadata']['modalities'] = ['image']
        
        try:
            # Stage 1: Cell Detection
            if self.enable_stage1:
                logger.info("=" * 60)
                logger.info("STAGE 1: Cell Detection (YOLOv8)")
                logger.info("=" * 60)
                
                detection_results = self.detector.detect(image_input)
                results['stage1_detection'] = detection_results
                
                logger.info(f"âœ“ Detected cells: {detection_results['total_counts']}")
                
                if self.save_intermediate:
                    self._save_json(detection_results, 'stage1_detection.json')
            
            # Stage 2: WBC Classification with Uncertainty
            if self.enable_stage2 and self.enable_stage1:
                logger.info("=" * 60)
                logger.info("STAGE 2: WBC Classification + Uncertainty")
                logger.info("=" * 60)
                
                if detection_results['per_image']:
                    all_wbc_crops = collect_wbc_crops(self.detector, detection_results['per_image'])

                    if all_wbc_crops:
                        wbc_images = [crop['image'] for crop in all_wbc_crops]
                        classification_results = self.classifier.classify_batch(
                            wbc_images, compute_gradcam=self.enable_gradcam
                        )

                        # Attach crop provenance so downstream stages can trace predictions to source images.
                        for pred, crop in zip(classification_results['predictions'], all_wbc_crops):
                            pred['source_image_path'] = crop['source_image_path']
                            pred['source_image_index'] = crop['source_image_index']
                            pred['source_detection_index'] = crop['index']

                        uncertainty_summary = self.classifier.create_uncertainty_summary(
                            classification_results['predictions']
                        )

                        results['stage2_classification'] = {
                            'predictions': classification_results['predictions'],
                            'summary': classification_results['summary'],
                            'uncertainty_summary': uncertainty_summary,
                            'total_wbc_crops': len(all_wbc_crops),
                            'images_processed': len(detection_results['per_image'])
                        }

                        logger.info(f"âœ“ Classified {len(all_wbc_crops)} WBCs across {len(detection_results['per_image'])} image(s)")
                        logger.info(f"  Flagged for review: {uncertainty_summary.get('flagged_count', 0)}")

                        if self.save_intermediate:
                            self._save_json(results['stage2_classification'], 'stage2_classification.json')
                    else:
                        logger.warning("No WBC crops found for classification")
                        results['stage2_classification'] = {
                            'error': 'No WBCs detected',
                            'predictions': [],
                            'summary': {'sample_count': 0, 'flagged_count': 0, 'requires_expert_review': False},
                            'uncertainty_summary': {
                                'total_samples': 0,
                                'uncertainty_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
                                'flagged_count': 0,
                                'flagged_percentage': 0.0,
                                'mean_confidence': 0.0,
                                'mean_entropy': 0.0,
                                'mean_variance': 0.0
                            },
                            'total_wbc_crops': 0,
                            'images_processed': len(detection_results['per_image'])
                        }
            
            # Stage 3: RAG-based Clinical Reasoning
            if self.enable_stage3:
                logger.info("=" * 60)
                logger.info("STAGE 3: Clinical Reasoning (RAG + LLM)")
                logger.info("=" * 60)
                
                # Build vision summary (image-derived findings + optional CBC)
                vision_summary = self._build_vision_summary(results)
                if cbc_report is not None:
                    vision_summary['cbc_report'] = cbc_report.to_dict()
                # Number of WBCs behind the differential, so the reasoner can
                # judge whether percentages are meaningful.
                n_wbc = int(results.get('stage2_classification', {}).get('total_wbc_crops', 0) or 0)
                vision_summary['wbc_classified'] = n_wbc
                vision_summary['min_wbc_for_differential'] = self.min_wbc_for_differential
                
                # Build query from vision results
                query = self._build_rag_query(vision_summary)
                logger.info(f"Query: {query}")
                
                # Retrieve relevant knowledge
                retrieved_chunks = self.retriever.retrieve(query)
                retrieval_quality = self.retriever.check_retrieval_quality(retrieved_chunks)
                
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks "
                           f"(sufficient={'yes' if retrieval_quality['sufficient_evidence'] else 'no'})")
                
                # Format context
                context = self.retriever.format_retrieved_context(retrieved_chunks)
                
                # Get uncertainty summary if available
                uncertainty_summary = results.get('stage2_classification', {}).get('uncertainty_summary')
                
                # Generate clinical reasoning (linear LLM or ReAct agent)
                reasoner = self.agent if self.reasoning_mode == 'agent' and self.agent else self.reasoner
                logger.info("Reasoning mode: %s", 'agent' if reasoner is self.agent else 'linear')
                reasoning_results = reasoner.generate_reasoning(
                    vision_summary=vision_summary,
                    retrieved_context=context,
                    uncertainty_summary=uncertainty_summary,
                    retrieved_chunks=retrieved_chunks
                )
                
                # Add retrieval quality info. In agent mode the agent already
                # produced its own retrieved_references via tool calls; keep
                # those rather than overwriting with the pre-fetched list.
                reasoning_results['retrieval_quality'] = retrieval_quality
                if not reasoning_results.get('retrieved_references'):
                    reasoning_results['retrieved_references'] = [
                        {
                            'reference_id': idx,
                            'source': chunk.get('source', 'unknown'),
                            'chunk_id': chunk.get('chunk_id'),
                            'score': chunk.get('score')
                        }
                        for idx, chunk in enumerate(retrieved_chunks, 1)
                    ]
                # Deterministic safety rule: too few cells -> expert review.
                reasoning_results = apply_minimum_cell_rule(
                    reasoning_results, n_wbc, self.min_wbc_for_differential)
                results['stage3_reasoning'] = reasoning_results
                
                logger.info(f"âœ“ Generated clinical reasoning")
                logger.info(f"  Safety flags: {reasoning_results.get('safety_flags', [])}")
                logger.info(f"  Expert review required: {reasoning_results.get('requires_expert_review', False)}")
                
                if self.save_intermediate:
                    self._save_json(reasoning_results, 'stage3_reasoning.json')
            
            # Save complete results
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_json(results, f'complete_analysis_{timestamp}.json')
            
            # Add execution time
            end_time = datetime.now()
            results['metadata']['execution_time_seconds'] = (end_time - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("âœ“ Pipeline completed successfully")
            logger.info(f"  Total time: {results['metadata']['execution_time_seconds']:.2f}s")
            logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            if not self.continue_on_error:
                raise
            
            results['error'] = str(e)
            results['metadata']['status'] = 'failed'
            return results
    
    def _build_vision_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Build unified vision summary from stage 1 and 2 results."""
        summary = {}
        
        # Add detection results
        if 'stage1_detection' in results:
            detection = results['stage1_detection']
            summary['cell_counts'] = detection.get('total_counts', {})
            summary['cell_count_stats'] = detection.get('cell_count_stats', {})
        
        # Add classification results
        if 'stage2_classification' in results:
            classification = results['stage2_classification']
            
            # Build WBC differential (percentage breakdown)
            if 'summary' in classification:
                class_dist = classification['summary'].get('class_distribution', {})
                total = sum(class_dist.values())
                
                if total > 0:
                    wbc_differential = {
                        cls: round(100.0 * count / total, 1)
                        for cls, count in class_dist.items()
                    }
                    summary['wbc_differential'] = wbc_differential
            
            # Add uncertainty summary
            if 'uncertainty_summary' in classification:
                summary['uncertainty_summary'] = classification['uncertainty_summary']
        
        return summary
    
    def _build_rag_query(self, vision_summary: Dict[str, Any]) -> str:
        """Build a natural-language semantic query from vision results.

        Important: detection counts here are *per-image cell counts in a
        microscope field*, NOT clinical concentrations (cells/uL). Therefore
        we describe what was *observed in the smear* and let the WBC
        differential (relative percentages) drive the clinical phrasing,
        instead of guessing leukocytosis / thrombocytopenia from raw counts.
        """
        observations: List[str] = []

        if 'wbc_differential' in vision_summary:
            diff = vision_summary['wbc_differential']
            ordered = sorted(diff.items(), key=lambda kv: kv[1], reverse=True)
            top_terms = [f"{pct:.0f}% {name}" for name, pct in ordered if pct > 0]
            if top_terms:
                observations.append(
                    "WBC differential observed in the smear: " + ", ".join(top_terms[:6])
                )

            # Relative-abundance morphology cues (still observation-level, not
            # diagnostic claims about the patient).
            morphology_cues = []
            if diff.get('neutrophil', 0) > 70:
                morphology_cues.append("neutrophil predominance")
            if diff.get('lymphocyte', 0) > 40:
                morphology_cues.append("lymphocyte predominance")
            if diff.get('monocyte', 0) > 10:
                morphology_cues.append("increased monocytes")
            if diff.get('eosinophil', 0) > 5:
                morphology_cues.append("increased eosinophils")
            if diff.get('basophil', 0) > 2:
                morphology_cues.append("increased basophils")
            if diff.get('ig', 0) > 2:
                morphology_cues.append("immature granulocytes present")
            if diff.get('erythroblast', 0) > 1:
                morphology_cues.append("nucleated red blood cells present")
            if morphology_cues:
                observations.append(
                    "Notable morphology cues: " + ", ".join(morphology_cues)
                )

        if 'cell_counts' in vision_summary:
            counts = vision_summary['cell_counts']
            count_terms = [f"{cls}={n}" for cls, n in counts.items() if n]
            if count_terms:
                observations.append(
                    "Per-image detection counts (microscope field, not clinical concentration): "
                    + ", ".join(count_terms)
                )

        if 'uncertainty_summary' in vision_summary:
            unc = vision_summary['uncertainty_summary']
            flagged = unc.get('flagged_count', 0)
            total = unc.get('total_samples', 0)
            if flagged and total:
                observations.append(
                    f"{flagged} of {total} WBC classifications flagged with high model uncertainty"
                )

        # Multimodal: include any abnormal CBC labels in the retrieval query so
        # we pull textbook context for *both* the image findings and the lab.
        if 'cbc_report' in vision_summary:
            cbc = vision_summary['cbc_report'] or {}
            abnormal_labels = [
                f.get('label')
                for f in cbc.get('findings', [])
                if f.get('direction') != 'normal' and f.get('label')
            ]
            if abnormal_labels:
                observations.append(
                    "Concurrent CBC abnormalities: " + ", ".join(sorted(set(abnormal_labels)))
                )

        if not observations:
            observations.append(
                "Peripheral blood smear with no flagged abnormalities; "
                "summarize normal WBC differential reference ranges and morphology."
            )

        query = (
            "Provide hematology reference information relevant to the following "
            "peripheral blood smear observations. "
            + " ".join(observations)
        )
        return query

    def _save_json(self, data: Dict[str, Any], filename: str) -> None:
        """Save data as JSON file."""
        filepath = self.results_dir / filename
        serializable = self._to_json_safe(data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved: {filepath}")

    def _to_json_safe(self, value: Any, parent_key: Optional[str] = None) -> Any:
        """Recursively convert values into JSON-safe structures."""
        if isinstance(value, dict):
            return {k: self._to_json_safe(v, parent_key=k) for k, v in value.items()}

        if isinstance(value, list):
            return [self._to_json_safe(item, parent_key=parent_key) for item in value]

        if isinstance(value, tuple):
            return [self._to_json_safe(item, parent_key=parent_key) for item in value]

        # Avoid dumping huge raw image arrays into JSON artifacts.
        if isinstance(value, np.ndarray):
            return {
                'omitted': True,
                'dtype': str(value.dtype),
                'shape': list(value.shape),
                'reason': 'non-serializable ndarray omitted from JSON output'
            }

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()

        if isinstance(value, Path):
            return str(value)

        return value


def create_pipeline(config_path: Optional[str] = None) -> BloodSmearPipeline:
    """
    Factory function to create pipeline instance.
    
    Args:
        config_path: Path to config.yaml (None to use default search)
        
    Returns:
        Initialized BloodSmearPipeline
        
    Examples:
        >>> pipeline = create_pipeline('config.yaml')
        >>> results = pipeline.analyze('smear.jpg')
    """
    return BloodSmearPipeline(config_path=config_path)

"""
WBC classification module with uncertainty quantification using Monte Carlo Dropout.

Provides EfficientNet-based classification with Bayesian uncertainty estimates.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm

from src.config.config_loader import Config, get_model_path
from src.utils.metrics import classify_uncertainty_level, create_summary_statistics

logger = logging.getLogger(__name__)


class WBCClassifier:
    """EfficientNet-B0 based WBC classifier with MC Dropout uncertainty."""
    
    def __init__(self, config: Config):
        """
        Initialize WBC classifier with configuration.
        
        Args:
            config: Configuration object with classification parameters
            
        Raises:
            FileNotFoundError: If model file not found
        """
        self.config = config
        
        # Load configuration parameters
        self.mc_passes = config.get('classification.mc_dropout_passes', 20)
        self.wbc_classes = config.get('classification.wbc_classes', [
            'basophil', 'eosinophil', 'erythroblast', 'ig',
            'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
        ])
        self.num_classes = len(self.wbc_classes)
        
        # Uncertainty thresholds
        self.low_conf = config.get('classification.uncertainty.low.min_confidence', 0.85)
        self.low_entropy = config.get('classification.uncertainty.low.max_entropy', 0.3)
        self.med_conf = config.get('classification.uncertainty.medium.min_confidence', 0.65)
        self.med_entropy = config.get('classification.uncertainty.medium.max_entropy', 0.6)
        
        # Image preprocessing
        image_size = config.get('classification.image_size', 224)
        mean = config.get('classification.normalization.mean', [0.485, 0.456, 0.406])
        std = config.get('classification.normalization.std', [0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        configured_model_path = config.get('models.efficientnet_classification')
        if not configured_model_path:
            raise ValueError("Model path not specified in config: models.efficientnet_classification")

        model_path = get_model_path(config, 'efficientnet_classification')
        
        logger.info(f"Loading EfficientNet model from: {model_path}")
        self.model = self._load_model(model_path)
        
        logger.info(f"WBCClassifier initialized (device={self.device}, mc_passes={self.mc_passes})")
    
    def _load_model(self, model_path: Path) -> nn.Module:
        """
        Load EfficientNet model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint (.pt file)
            
        Returns:
            Loaded model in eval mode
        """
        # Create base model
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=self.num_classes)

        # Load checkpoint. Prefer weights_only=True (PyTorch >= 2.0) to mitigate
        # arbitrary-code-execution risk from untrusted .pt files. Fall back only
        # if the checkpoint truly cannot be loaded that way (e.g. older format).
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except (TypeError, RuntimeError, Exception) as load_exc:  # noqa: BLE001
            logger.warning(
                "weights_only=True load failed (%s). Falling back to legacy load. "
                "Only do this for checkpoints you trust.",
                load_exc,
            )
            checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume checkpoint is the state dict itself
            state_dict = checkpoint

        # Some training runs use a multi-layer classifier head (classifier.0/3/6).
        if 'classifier.0.weight' in state_dict:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),  # must match the training notebook (0.3)
                nn.Linear(512, self.num_classes),
            )

        model.load_state_dict(state_dict)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def classify_with_uncertainty(
        self,
        image: Union[np.ndarray, Image.Image],
        n_passes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Classify WBC image with uncertainty quantification using MC Dropout.
        
        Args:
            image: Input image as numpy array (H×W×3) or PIL Image
            n_passes: Number of MC Dropout passes (None to use config default)
            
        Returns:
            Dictionary with prediction and uncertainty:
                {
                    'predicted_class': str,
                    'confidence': float,              # Mean probability of predicted class
                    'variance': float,                # Mean variance across classes
                    'entropy': float,                 # Shannon entropy
                    'uncertainty_level': 'LOW'|'MEDIUM'|'HIGH',
                    'flagged': bool,                  # True if high uncertainty
                    'class_probabilities': dict       # All class probabilities
                }
                
        Examples:
            >>> classifier = WBCClassifier(config)
            >>> result = classifier.classify_with_uncertainty(wbc_crop)
            >>> print(f"{result['predicted_class']}: {result['confidence']:.2%}")
            >>> if result['flagged']:
            ...     print("⚠️ High uncertainty - requires expert review")
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Number of MC passes
        n = n_passes if n_passes is not None else self.mc_passes
        
        # Enable dropout for MC sampling
        self._enable_dropout(self.model)
        
        # Collect predictions from multiple stochastic forward passes
        predictions = []
        with torch.no_grad():
            for _ in range(n):
                logits = self.model(image_tensor)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy()[0])
        
        # Stack predictions: shape (n_passes, n_classes)
        predictions = np.array(predictions)
        
        # Calculate mean probabilities
        mean_probs = np.mean(predictions, axis=0)
        
        # Predicted class
        predicted_idx = np.argmax(mean_probs)
        predicted_class = self.wbc_classes[predicted_idx]
        confidence = float(mean_probs[predicted_idx])
        
        # Calculate uncertainty metrics
        variance = float(np.mean(np.var(predictions, axis=0)))
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-8)))
        
        # Classify uncertainty level
        uncertainty_level, flagged = classify_uncertainty_level(
            confidence=confidence,
            entropy=entropy,
            low_conf_threshold=self.low_conf,
            low_entropy_threshold=self.low_entropy,
            med_conf_threshold=self.med_conf,
            med_entropy_threshold=self.med_entropy
        )
        
        # Create class probabilities dictionary
        class_probs = {cls: float(prob) for cls, prob in zip(self.wbc_classes, mean_probs)}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'variance': variance,
            'entropy': entropy,
            'uncertainty_level': uncertainty_level,
            'flagged': flagged,
            'class_probabilities': class_probs
        }
    
    def classify_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        n_passes: Optional[int] = None,
        compute_gradcam: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify batch of WBC images with uncertainty quantification.

        Args:
            images: List of images as numpy arrays or PIL Images
            n_passes: Number of MC Dropout passes per image
            compute_gradcam: If True, attach a Grad-CAM heatmap (PNG data URL)
                to each prediction under the ``gradcam_base64`` key.
        """
        logger.info(f"Classifying batch of {len(images)} images")

        predictions = []
        for idx, image in enumerate(images):
            try:
                result = self.classify_with_uncertainty(image, n_passes)
                result['index'] = idx
                if compute_gradcam:
                    try:
                        result['gradcam_base64'] = self.compute_gradcam(
                            image, predicted_class=result['predicted_class']
                        )
                    except Exception as cam_exc:  # noqa: BLE001
                        logger.warning("Grad-CAM failed for crop %d: %s", idx, cam_exc)
                predictions.append(result)
            except Exception as e:
                logger.error(f"Failed to classify image {idx}: {e}")
                # Add placeholder result
                predictions.append({
                    'index': idx,
                    'predicted_class': 'unknown',
                    'confidence': 0.0,
                    'variance': 0.0,
                    'entropy': 0.0,
                    'uncertainty_level': 'HIGH',
                    'flagged': True,
                    'error': str(e)
                })

        # Create summary statistics
        summary = create_summary_statistics(predictions)

        return {
            'predictions': predictions,
            'summary': summary
        }

    def compute_gradcam(
        self,
        image: Union[np.ndarray, Image.Image],
        predicted_class: Optional[str] = None,
    ) -> str:
        """
        Compute a Grad-CAM heatmap for a single WBC crop.

        Returns a ``data:image/png;base64,...`` URL ready to embed in JSON / HTML.
        """
        from src.classification.gradcam import GradCAM

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image

        # Determine class index. If not given, use the argmax of a single deterministic forward pass.
        if predicted_class is not None and predicted_class in self.wbc_classes:
            class_idx = self.wbc_classes.index(predicted_class)
        else:
            self.model.eval()
            with torch.no_grad():
                input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                class_idx = int(torch.softmax(self.model(input_tensor), dim=1).argmax(dim=1).item())

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with GradCAM(self.model) as cam:
            heatmap = cam.compute(input_tensor, class_idx=class_idx)

        return GradCAM.render_overlay(pil_image, heatmap, alpha=0.45)


    def _enable_dropout(self, model: nn.Module) -> None:
        """
        Enable dropout layers at inference time for MC Dropout.
        
        Args:
            model: PyTorch model with dropout layers
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def create_uncertainty_summary(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create detailed uncertainty summary from predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with uncertainty analysis:
                {
                    'total_samples': int,
                    'uncertainty_distribution': {'LOW': int, 'MEDIUM': int, 'HIGH': int},
                    'flagged_count': int,
                    'flagged_percentage': float,
                    'mean_confidence': float,
                    'mean_entropy': float,
                    'mean_variance': float
                }
        """
        if not predictions:
            return {}
        
        # Count uncertainty levels
        uncertainty_dist = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for pred in predictions:
            level = pred.get('uncertainty_level', 'HIGH')
            uncertainty_dist[level] = uncertainty_dist.get(level, 0) + 1
        
        # Calculate averages
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]
        entropies = [p['entropy'] for p in predictions if 'entropy' in p]
        variances = [p['variance'] for p in predictions if 'variance' in p]
        flagged_count = sum(p.get('flagged', False) for p in predictions)
        
        return {
            'total_samples': len(predictions),
            'sample_count': len(predictions),  # alias for downstream consumers
            'uncertainty_distribution': uncertainty_dist,
            'flagged_count': flagged_count,
            'flagged_samples': flagged_count,  # alias for schema consistency
            'flagged_percentage': 100.0 * flagged_count / len(predictions) if predictions else 0.0,
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'mean_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'mean_variance': float(np.mean(variances)) if variances else 0.0
        }

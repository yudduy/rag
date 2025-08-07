"""
Multimodal embedding module for CLIP integration and cross-modal retrieval.

This module provides lightweight multimodal capabilities for the RAG system,
enabling text-image shared embedding space and cross-modal search functionality.
"""

import os
import logging
import time
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not available. Install for advanced image processing")

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available. Install clip-by-openai for multimodal features")

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR not available. Install pytesseract for image text extraction")

from llama_index.core.schema import BaseNode, ImageNode, TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class MultimodalEmbedding(BaseEmbedding):
    """
    CLIP-based multimodal embedding model for text and images.
    
    Provides unified embedding space for text and images, enabling cross-modal
    retrieval and semantic search across different modalities.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the multimodal embedding model.
        
        Args:
            model_name: CLIP model name (ViT-B/32, ViT-B/16, ViT-L/14)
            device: Device to run the model on (cuda/cpu/auto)
            cache_dir: Directory to cache the model
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available. Install clip-by-openai package")
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir or os.getenv("CLIP_MODEL_CACHE_DIR", "./models/clip")
        
        # Create cache directory
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load CLIP model
        self._load_model()
        
        # Configuration
        self.max_image_size_mb = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
        self.supported_formats = set(os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,bmp,tiff").lower().split(","))
        self.min_quality_score = float(os.getenv("MIN_IMAGE_QUALITY_SCORE", "0.5"))
        
        logger.info(f"MultimodalEmbedding initialized with {model_name} on {self.device}")
    
    def _load_model(self):
        """Load the CLIP model with caching."""
        try:
            start_time = time.time()
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            
            # Get embedding dimension
            with torch.no_grad():
                dummy_text = clip.tokenize(["test"]).to(self.device)
                dummy_features = self.model.encode_text(dummy_text)
                self._embed_dim = dummy_features.shape[1]
            
            load_time = time.time() - start_time
            logger.info(f"CLIP model loaded in {load_time:.2f}s, embedding dim: {self._embed_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"CLIP model loading failed: {e}")
    
    @property
    def embed_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embed_dim
    
    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding using CLIP.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            with torch.no_grad():
                # Tokenize and encode text
                tokens = clip.tokenize([text], truncate=True).to(self.device)
                features = self.model.encode_text(tokens)
                
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                
                return features[0].cpu().numpy().tolist()
                
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return [0.0] * self.embed_dim
    
    def get_image_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """
        Get image embedding using CLIP.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of embedding values
        """
        try:
            # Validate image
            if not self._validate_image(image_path):
                logger.warning(f"Image validation failed: {image_path}")
                return [0.0] * self.embed_dim
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode image
                features = self.model.encode_image(image_tensor)
                
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
                
                return features[0].cpu().numpy().tolist()
                
        except Exception as e:
            logger.error(f"Image embedding failed for {image_path}: {e}")
            return [0.0] * self.embed_dim
    
    def get_batch_embeddings(self, inputs: List[Union[str, Path]]) -> List[List[float]]:
        """
        Get embeddings for a batch of text or images.
        
        Args:
            inputs: List of texts or image paths
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = int(os.getenv("IMAGE_BATCH_SIZE", "8"))
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_embeddings = []
            
            for item in batch:
                if isinstance(item, (str, Path)) and Path(item).exists():
                    # Assume it's an image path
                    embedding = self.get_image_embedding(item)
                else:
                    # Assume it's text
                    embedding = self.get_text_embedding(str(item))
                
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            # Ensure result is in [0, 1] range
            return float(max(0, min(1, (similarity + 1) / 2)))
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Validate image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid
        """
        try:
            path = Path(image_path)
            
            # Check existence
            if not path.exists():
                return False
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_image_size_mb:
                logger.warning(f"Image too large: {size_mb:.1f}MB > {self.max_image_size_mb}MB")
                return False
            
            # Check format
            suffix = path.suffix.lower().lstrip('.')
            if suffix not in self.supported_formats:
                logger.warning(f"Unsupported image format: {suffix}")
                return False
            
            # Try to open image
            with Image.open(path) as img:
                # Basic quality check (could be enhanced)
                if img.width < 32 or img.height < 32:
                    logger.warning(f"Image too small: {img.width}x{img.height}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    # Required by BaseEmbedding interface
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        return self.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        return self.get_text_embedding(text)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get embedding for query text."""
        return self.get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding for text."""
        return self.get_text_embedding(text)


class ImageTextExtractor:
    """
    Extract text from images using OCR for enhanced indexing.
    """
    
    def __init__(self, language: str = "eng"):
        """
        Initialize the OCR extractor.
        
        Args:
            language: OCR language (default: 'eng')
        """
        self.language = language
        self.ocr_available = OCR_AVAILABLE
        
        if not self.ocr_available:
            logger.warning("OCR not available. Text extraction from images disabled")
    
    def extract_text(self, image_path: Union[str, Path]) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        if not self.ocr_available:
            return ""
        
        try:
            # Check if cv2 is available
            if not CV2_AVAILABLE:
                # Fallback to PIL for basic OCR
                from PIL import Image
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, lang=self.language)
                return ' '.join(text.split()).strip()
            
            # Read image with cv2
            image = cv2.imread(str(image_path))
            if image is None:
                return ""
            
            # Preprocess for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get better results
            _, threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text
            text = pytesseract.image_to_string(threshold_img, lang=self.language)
            
            # Clean up text
            text = ' '.join(text.split())  # Remove extra whitespace
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return ""


class MultimodalNodeCreator:
    """
    Create multimodal nodes combining text and image information.
    """
    
    def __init__(self, embedding_model: MultimodalEmbedding, ocr_enabled: bool = True):
        """
        Initialize the node creator.
        
        Args:
            embedding_model: Multimodal embedding model
            ocr_enabled: Whether to enable OCR text extraction
        """
        self.embedding_model = embedding_model
        self.text_extractor = ImageTextExtractor() if ocr_enabled else None
        
    def create_image_node(
        self, 
        image_path: Union[str, Path], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ImageNode]:
        """
        Create an image node with multimodal embedding.
        
        Args:
            image_path: Path to image file
            metadata: Additional metadata
            
        Returns:
            ImageNode or None if creation fails
        """
        try:
            path = Path(image_path)
            
            # Get image embedding
            embedding = self.embedding_model.get_image_embedding(path)
            
            # Extract text if OCR is enabled
            extracted_text = ""
            if self.text_extractor:
                extracted_text = self.text_extractor.extract_text(path)
            
            # Create metadata
            node_metadata = metadata or {}
            node_metadata.update({
                "image_path": str(path.absolute()),
                "image_name": path.name,
                "image_size_bytes": path.stat().st_size,
                "extracted_text": extracted_text,
                "modality": "image",
                "embedding_dim": len(embedding)
            })
            
            # Create ImageNode
            node = ImageNode(
                image_path=str(path),
                text=extracted_text,  # Use extracted text as node text
                metadata=node_metadata,
                embedding=embedding
            )
            
            logger.debug(f"Created image node for {path.name}")
            return node
            
        except Exception as e:
            logger.error(f"Failed to create image node for {image_path}: {e}")
            return None
    
    def create_text_node_with_multimodal_embedding(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> TextNode:
        """
        Create a text node with multimodal (CLIP) embedding.
        
        Args:
            text: Text content
            metadata: Additional metadata
            
        Returns:
            TextNode with CLIP embedding
        """
        try:
            # Get text embedding using CLIP
            embedding = self.embedding_model.get_text_embedding(text)
            
            # Create metadata
            node_metadata = metadata or {}
            node_metadata.update({
                "modality": "text",
                "embedding_model": "clip",
                "embedding_dim": len(embedding)
            })
            
            # Create TextNode with multimodal embedding
            node = TextNode(
                text=text,
                metadata=node_metadata,
                embedding=embedding
            )
            
            logger.debug(f"Created text node with multimodal embedding (length: {len(text)})")
            return node
            
        except Exception as e:
            logger.error(f"Failed to create multimodal text node: {e}")
            # Fallback to node without embedding
            return TextNode(text=text, metadata=metadata or {})


def get_multimodal_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> Optional[MultimodalEmbedding]:
    """
    Factory function to create multimodal embedding model.
    
    Args:
        model_name: CLIP model name
        device: Device to run on
        
    Returns:
        MultimodalEmbedding instance or None if disabled/unavailable
    """
    try:
        # Check if multimodal is enabled
        if not os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true":
            logger.info("Multimodal embedding disabled via configuration")
            return None
        
        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available, multimodal embedding disabled")
            return None
        
        model_name = model_name or os.getenv("CLIP_MODEL_NAME", "ViT-B/32")
        
        return MultimodalEmbedding(
            model_name=model_name,
            device=device
        )
        
    except Exception as e:
        logger.error(f"Failed to create multimodal embedding model: {e}")
        return None


def is_multimodal_enabled() -> bool:
    """Check if multimodal features are enabled and available."""
    return (
        os.getenv("MULTIMODAL_ENABLED", "false").lower() == "true" and
        CLIP_AVAILABLE
    )


def is_image_indexing_enabled() -> bool:
    """Check if image indexing is enabled."""
    return (
        is_multimodal_enabled() and
        os.getenv("IMAGE_INDEXING_ENABLED", "false").lower() == "true"
    )


def is_cross_modal_search_enabled() -> bool:
    """Check if cross-modal search is enabled."""
    return (
        is_multimodal_enabled() and
        os.getenv("CROSS_MODAL_SEARCH_ENABLED", "false").lower() == "true"
    )
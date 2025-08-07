import logging
import os
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv

# Import multimodal components
from src.multimodal import (
    get_multimodal_embedding_model,
    MultimodalNodeCreator,
    is_multimodal_enabled,
    is_image_indexing_enabled
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def generate_index():
    """
    Index the documents in the data directory using sentence-window parsing
    with optional multimodal support for images.
    """
    from src.index import STORAGE_DIR, get_multimodal_index, persist_multimodal_index
    from src.settings import init_settings
    from llama_index.core.indices import VectorStoreIndex
    from llama_index.core.readers import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceWindowNodeParser

    try:
        load_dotenv()
        init_settings()

        # Configuration for sentence windowing
        window_size = int(os.environ.get("SENTENCE_WINDOW_SIZE", "3"))
        data_dir = os.environ.get("DATA_DIR", "ui/data")
        
        logger.info(f"Creating new index with sentence window size: {window_size}")
        logger.info(f"Reading documents from: {data_dir}")
        
        # Check multimodal configuration
        multimodal_enabled = is_multimodal_enabled()
        image_indexing_enabled = is_image_indexing_enabled()
        
        if multimodal_enabled:
            logger.info("Multimodal processing enabled")
            if image_indexing_enabled:
                logger.info("Image indexing enabled - will process images alongside text documents")
        
        # Validate data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Load documents (text files)
        reader = SimpleDirectoryReader(
            data_dir,
            recursive=True,
            exclude_hidden=True,
            file_extractor=None  # Use default extractors for now
        )
        documents = reader.load_data()
        
        logger.info(f"Loaded {len(documents)} text documents")
        
        # Parse documents with sentence windowing
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        
        text_nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(text_nodes)} sentence-window text nodes")
        
        # Process images if multimodal indexing is enabled
        image_nodes = []
        if image_indexing_enabled:
            try:
                image_nodes = _process_images(data_dir)
                logger.info(f"Created {len(image_nodes)} image nodes")
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                logger.info("Continuing with text-only indexing")
        
        # Combine all nodes
        all_nodes = text_nodes + image_nodes
        
        if not all_nodes:
            raise ValueError("No nodes created from documents")
        
        logger.info(f"Total nodes for indexing: {len(all_nodes)} ({len(text_nodes)} text, {len(image_nodes)} image)")
        
        # Create index with multimodal support if enabled
        if multimodal_enabled:
            index = get_multimodal_index(all_nodes, show_progress=True)
            if not index:
                logger.warning("Multimodal index creation failed, falling back to standard index")
                index = VectorStoreIndex(text_nodes, show_progress=True)
        else:
            index = VectorStoreIndex(text_nodes, show_progress=True)
        
        # Persist the index
        if persist_multimodal_index(index):
            logger.info(f"Successfully created and stored index in {STORAGE_DIR}")
            
            # Log final statistics
            from src.index import get_index_info
            info = get_index_info(index)
            logger.info(f"Index statistics: {info}")
        else:
            raise RuntimeError("Failed to persist index")
        
    except Exception as e:
        logger.error(f"Failed to generate index: {str(e)}")
        raise


def _process_images(data_dir: str) -> List:
    """
    Process images in the data directory for multimodal indexing.
    
    Args:
        data_dir: Directory containing images
        
    Returns:
        List of image nodes
    """
    image_nodes = []
    
    try:
        # Get multimodal embedding model
        embedding_model = get_multimodal_embedding_model()
        if not embedding_model:
            logger.warning("Multimodal embedding model not available")
            return []
        
        # Create node creator
        node_creator = MultimodalNodeCreator(
            embedding_model=embedding_model,
            ocr_enabled=os.getenv("OCR_ENABLED", "true").lower() == "true"
        )
        
        # Find image files
        data_path = Path(data_dir)
        supported_formats = set(os.getenv("SUPPORTED_IMAGE_FORMATS", "jpg,jpeg,png,bmp,tiff").lower().split(","))
        
        image_files = []
        for ext in supported_formats:
            image_files.extend(data_path.rglob(f"*.{ext}"))
            image_files.extend(data_path.rglob(f"*.{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Process each image
        for image_path in image_files:
            try:
                # Create image node
                node = node_creator.create_image_node(
                    image_path=image_path,
                    metadata={
                        "source": str(image_path.relative_to(data_path)),
                        "file_type": image_path.suffix.lower(),
                        "parent_dir": data_dir
                    }
                )
                
                if node:
                    image_nodes.append(node)
                    logger.debug(f"Processed image: {image_path.name}")
                else:
                    logger.warning(f"Failed to create node for: {image_path}")
                    
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(image_nodes)}/{len(image_files)} images")
        return image_nodes
        
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return []


def _extract_images_from_documents(data_dir: str) -> List:
    """
    Extract images from PDF and other document types.
    
    Args:
        data_dir: Directory containing documents
        
    Returns:
        List of extracted image nodes
    """
    extracted_image_nodes = []
    
    try:
        # This is a placeholder for more advanced document image extraction
        # Could be implemented with pdf2image, python-docx image extraction, etc.
        
        from pathlib import Path
        import tempfile
        
        data_path = Path(data_dir)
        pdf_files = list(data_path.rglob("*.pdf")) + list(data_path.rglob("*.PDF"))
        
        if not pdf_files:
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files for image extraction")
        
        try:
            from pdf2image import convert_from_path
            
            embedding_model = get_multimodal_embedding_model()
            if not embedding_model:
                return []
                
            node_creator = MultimodalNodeCreator(embedding_model=embedding_model)
            
            for pdf_path in pdf_files:
                try:
                    # Convert PDF pages to images
                    with tempfile.TemporaryDirectory() as temp_dir:
                        images = convert_from_path(
                            pdf_path,
                            output_folder=temp_dir,
                            first_page=1,
                            last_page=min(5, 20),  # Limit to first 5 pages for performance
                            dpi=150  # Lower DPI for faster processing
                        )
                        
                        for i, image in enumerate(images):
                            image_path = Path(temp_dir) / f"{pdf_path.stem}_page_{i+1}.png"
                            image.save(image_path, "PNG")
                            
                            # Create node for extracted image
                            node = node_creator.create_image_node(
                                image_path=image_path,
                                metadata={
                                    "source_document": str(pdf_path.name),
                                    "page_number": i + 1,
                                    "extracted_from_pdf": True,
                                    "parent_dir": data_dir
                                }
                            )
                            
                            if node:
                                extracted_image_nodes.append(node)
                        
                        logger.debug(f"Extracted {len(images)} images from {pdf_path.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to extract images from {pdf_path}: {e}")
                    continue
            
            logger.info(f"Extracted {len(extracted_image_nodes)} images from PDF documents")
            
        except ImportError:
            logger.info("pdf2image not available - skipping PDF image extraction")
            
    except Exception as e:
        logger.error(f"Document image extraction failed: {e}")
        
    return extracted_image_nodes


def generate_multimodal_index():
    """
    Generate index with full multimodal support including document image extraction.
    This is an enhanced version that extracts images from documents.
    """
    logger.info("Starting multimodal index generation with document image extraction")
    
    # First run standard generation
    generate_index()
    
    # Then extract and index images from documents if enabled
    if is_image_indexing_enabled():
        try:
            from src.index import get_index, STORAGE_DIR
            
            data_dir = os.environ.get("DATA_DIR", "ui/data")
            
            # Extract images from documents
            extracted_nodes = _extract_images_from_documents(data_dir)
            
            if extracted_nodes:
                logger.info(f"Adding {len(extracted_nodes)} extracted images to index")
                
                # Load existing index
                existing_index = get_index()
                if existing_index:
                    # Add new nodes to existing index
                    for node in extracted_nodes:
                        existing_index.insert(node)
                    
                    # Persist updated index
                    existing_index.storage_context.persist(STORAGE_DIR)
                    logger.info("Successfully added extracted images to existing index")
                    
        except Exception as e:
            logger.error(f"Failed to add extracted images to index: {e}")
            logger.info("Basic multimodal index generation completed successfully")

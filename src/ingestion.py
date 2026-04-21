import os
import logging
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document(file_path, original_filename):
    """Load and split a document into chunks with metadata."""
    logger.info(f"Processing document: {original_filename}")
    ext = os.path.splitext(file_path)[-1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        docs = loader.load()
        if not docs:
            raise ValueError(f"No content extracted from {original_filename}. The file might be empty or corrupted.")
            
        page_count = len(docs)
        
        # Ensure source metadata is the filename, not the temp path
        for doc in docs:
            doc.metadata["source"] = original_filename
            
        # Standardized text splitter configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, 
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        logger.info(f"Successfully processed {original_filename}: {len(splits)} chunks.")
        return splits, page_count
    except Exception as e:
        logger.error(f"Error processing {original_filename}: {str(e)}")
        raise

def process_multiple_documents(temp_files, original_filenames):
    """Process multiple files and return combined splits and statistics."""
    all_splits = []
    total_pages = 0
    
    if not temp_files:
        return [], {"file_count": 0, "total_pages": 0, "total_chunks": 0}

    for path, name in zip(temp_files, original_filenames):
        splits, pages = process_document(path, name)
        all_splits.extend(splits)
        total_pages += pages
    
    stats = {
        "file_count": len(temp_files),
        "total_pages": total_pages,
        "total_chunks": len(all_splits)
    }
    return all_splits, stats

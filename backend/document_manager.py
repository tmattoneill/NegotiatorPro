import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes
from datetime import datetime

# Document processing imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    Docx2txtLoader
)
try:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document uploads and processing for RAG system"""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF Document',
        '.txt': 'Text Document', 
        '.docx': 'Word Document (DOCX)'
    }
    
    # Add .doc support only if unstructured is available
    if HAS_UNSTRUCTURED:
        SUPPORTED_EXTENSIONS['.doc'] = 'Word Document (DOC)'
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, sources_dir: str = "sources", upload_dir: str = "uploads"):
        self.sources_dir = Path(sources_dir)
        self.upload_dir = Path(upload_dir)
        self.sources_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        
    def get_supported_extensions(self) -> Dict[str, str]:
        """Get supported file extensions and descriptions"""
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def validate_file(self, file_path: str, max_size: Optional[int] = None) -> Dict[str, Any]:
        """Validate uploaded file"""
        file_path = Path(file_path)
        max_size = max_size or self.MAX_FILE_SIZE
        
        result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check if file exists
        if not file_path.exists():
            result["errors"].append("File does not exist")
            return result
        
        # Check file size
        file_size = file_path.stat().st_size
        result["info"]["size"] = file_size
        result["info"]["size_mb"] = round(file_size / (1024 * 1024), 2)
        
        if file_size > max_size:
            result["errors"].append(f"File size ({result['info']['size_mb']}MB) exceeds maximum ({max_size / (1024 * 1024)}MB)")
            return result
        
        if file_size == 0:
            result["errors"].append("File is empty")
            return result
        
        # Check file extension
        ext = file_path.suffix.lower()
        result["info"]["extension"] = ext
        
        if not self.is_supported_file(file_path.name):
            result["errors"].append(f"File type '{ext}' not supported. Supported types: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}")
            return result
        
        result["info"]["type"] = self.SUPPORTED_EXTENSIONS[ext]
        
        # Try to detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        result["info"]["mime_type"] = mime_type
        
        # Basic file content validation
        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Read first 1KB
                    if not content.strip():
                        result["warnings"].append("Text file appears to be empty or whitespace only")
        except UnicodeDecodeError:
            result["warnings"].append("Text file encoding may not be UTF-8")
        except Exception as e:
            result["warnings"].append(f"Could not validate file content: {str(e)}")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def save_uploaded_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Save uploaded file to sources directory"""
        try:
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation["valid"]:
                return {
                    "success": False,
                    "message": "File validation failed",
                    "errors": validation["errors"],
                    "warnings": validation["warnings"]
                }
            
            # Generate unique filename if needed
            source_path = self.sources_dir / filename
            original_filename = filename
            counter = 1
            
            while source_path.exists():
                name_part = Path(original_filename).stem
                ext_part = Path(original_filename).suffix
                filename = f"{name_part}_{counter}{ext_part}"
                source_path = self.sources_dir / filename
                counter += 1
            
            # Copy file to sources directory
            shutil.copy2(file_path, source_path)
            
            # Log the upload
            logger.info(f"File uploaded successfully: {filename} ({validation['info']['size_mb']}MB)")
            
            return {
                "success": True,
                "message": f"File '{filename}' uploaded successfully",
                "filename": filename,
                "path": str(source_path),
                "info": validation["info"],
                "warnings": validation["warnings"]
            }
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            return {
                "success": False,
                "message": f"Error saving file: {str(e)}",
                "errors": [str(e)]
            }
    
    def list_source_documents(self) -> List[Dict[str, Any]]:
        """List all documents in sources directory"""
        documents = []
        
        for file_path in self.sources_dir.iterdir():
            if file_path.is_file() and self.is_supported_file(file_path.name):
                try:
                    stat = file_path.stat()
                    documents.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": file_path.suffix.lower(),
                        "type": self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown")
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {file_path.name}: {e}")
        
        # Sort by modification time (newest first)
        documents.sort(key=lambda x: x["modified"], reverse=True)
        return documents
    
    def delete_document(self, filename: str) -> Dict[str, Any]:
        """Delete a document from sources directory"""
        try:
            file_path = self.sources_dir / filename
            
            if not file_path.exists():
                return {
                    "success": False,
                    "message": f"File '{filename}' not found"
                }
            
            if not file_path.is_file():
                return {
                    "success": False,
                    "message": f"'{filename}' is not a file"
                }
            
            file_path.unlink()
            logger.info(f"Document deleted: {filename}")
            
            return {
                "success": True,
                "message": f"Document '{filename}' deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            return {
                "success": False,
                "message": f"Error deleting document: {str(e)}"
            }
    
    def get_document_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a document"""
        file_path = self.sources_dir / filename
        
        if not file_path.exists():
            return None
        
        try:
            stat = file_path.stat()
            info = {
                "filename": filename,
                "path": str(file_path),
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower(),
                "type": self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "Unknown"),
                "pages": None,
                "word_count": None
            }
            
            # Try to get additional metadata
            try:
                if info["extension"] == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                    info["pages"] = len(documents)
                    info["word_count"] = sum(len(doc.page_content.split()) for doc in documents)
                elif info["extension"] == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        info["word_count"] = len(content.split())
                elif info["extension"] == '.docx':
                    loader = Docx2txtLoader(str(file_path))
                    documents = loader.load()
                    info["word_count"] = sum(len(doc.page_content.split()) for doc in documents)
                elif info["extension"] == '.doc' and HAS_UNSTRUCTURED:
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                    documents = loader.load()
                    info["word_count"] = sum(len(doc.page_content.split()) for doc in documents)
            except Exception as e:
                logger.warning(f"Could not get additional metadata for {filename}: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting document info for {filename}: {e}")
            return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        file_count = 0
        type_stats = {}
        
        for file_path in self.sources_dir.iterdir():
            if file_path.is_file() and self.is_supported_file(file_path.name):
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
                
                ext = file_path.suffix.lower()
                if ext not in type_stats:
                    type_stats[ext] = {"count": 0, "size": 0}
                type_stats[ext]["count"] += 1
                type_stats[ext]["size"] += size
        
        return {
            "total_files": file_count,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "type_breakdown": type_stats,
            "sources_dir": str(self.sources_dir)
        }
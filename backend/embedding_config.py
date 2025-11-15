import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingConfig:
    """Manages embedding model configuration and metadata"""
    
    # Available embedding models with their specifications
    EMBEDDING_MODELS = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "Most sophisticated - Highest quality embeddings",
            "cost_per_1k": 0.00013,
            "recommended": True,
            "max_input": 8191
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "Good balance - Better than ada-002",
            "cost_per_1k": 0.00002,
            "recommended": False,
            "max_input": 8191
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "description": "Legacy model - Lower quality",
            "cost_per_1k": 0.00010,
            "recommended": False,
            "max_input": 8191
        }
    }
    
    def __init__(self, config_file: str = "embedding_config.json"):
        self.config_file = Path(config_file)
        self.vectorstore_dir = Path("vectorstore")
        self.load_config()
    
    def load_config(self):
        """Load embedding configuration from file or create default"""
        default_config = {
            "current_model": "text-embedding-3-large",
            "vectorstore_model": None,  # Model used to build current vectorstore
            "vectorstore_created": None,
            "auto_detect": True,  # Automatically detect model from vectorstore metadata
            "fallback_model": "text-embedding-ada-002"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # Ensure all keys exist
                for key, value in default_config.items():
                    if key not in self.config:
                        self.config[key] = value
                logger.info(f"Loaded embedding config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading embedding config: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
            logger.info("Created new embedding config with defaults")
        
        # Auto-detect model from vectorstore if enabled
        if self.config["auto_detect"]:
            self.detect_vectorstore_model()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved embedding config to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving embedding config: {e}")
    
    def detect_vectorstore_model(self) -> Optional[str]:
        """Try to detect which model was used to build the current vectorstore"""
        metadata_file = self.vectorstore_dir / "metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                detected_model = metadata.get("embedding_model")
                
                if detected_model and detected_model in self.EMBEDDING_MODELS:
                    self.config["vectorstore_model"] = detected_model
                    logger.info(f"Detected vectorstore model: {detected_model}")
                    self.save_config()
                    return detected_model
            except Exception as e:
                logger.warning(f"Could not read vectorstore metadata: {e}")
        
        # Fallback: Try to infer from vectorstore dimensions
        return self.infer_model_from_dimensions()
    
    def infer_model_from_dimensions(self) -> Optional[str]:
        """Infer embedding model from vectorstore dimensions"""
        try:
            import faiss
            index_file = self.vectorstore_dir / "index.faiss"
            
            if index_file.exists():
                index = faiss.read_index(str(index_file))
                dimensions = index.d
                
                # Find models with matching dimensions
                matching_models = [
                    model for model, specs in self.EMBEDDING_MODELS.items()
                    if specs["dimensions"] == dimensions
                ]
                
                if matching_models:
                    # Prefer the first match (usually the most common)
                    inferred_model = matching_models[0]
                    self.config["vectorstore_model"] = inferred_model
                    logger.info(f"Inferred vectorstore model from dimensions ({dimensions}D): {inferred_model}")
                    self.save_config()
                    return inferred_model
                else:
                    logger.warning(f"Unknown vectorstore dimensions: {dimensions}")
            
        except Exception as e:
            logger.warning(f"Could not infer model from vectorstore: {e}")
        
        return None
    
    def get_current_model(self) -> str:
        """Get the model to use for queries (should match vectorstore)"""
        # Use vectorstore model if detected, otherwise current model
        return self.config.get("vectorstore_model") or self.config["current_model"]
    
    def get_model_specs(self, model_name: str) -> Dict[str, Any]:
        """Get specifications for a given model"""
        return self.EMBEDDING_MODELS.get(model_name, {})
    
    def set_vectorstore_model(self, model_name: str):
        """Set the model used for the vectorstore (called after rebuild)"""
        if model_name not in self.EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {model_name}")
        
        self.config["vectorstore_model"] = model_name
        self.config["vectorstore_created"] = None  # Will be set during vectorstore creation
        self.save_config()
        logger.info(f"Set vectorstore model to: {model_name}")
    
    def save_vectorstore_metadata(self, model_name: str, chunks_count: int):
        """Save metadata about the vectorstore build"""
        from datetime import datetime
        
        metadata = {
            "embedding_model": model_name,
            "model_specs": self.EMBEDDING_MODELS[model_name],
            "chunks_count": chunks_count,
            "created_at": datetime.now().isoformat(),
            "created_by": "negotiation_advisor"
        }
        
        metadata_file = self.vectorstore_dir / "metadata.json"
        try:
            # Ensure vectorstore directory exists
            self.vectorstore_dir.mkdir(exist_ok=True)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update our config
            self.config["vectorstore_model"] = model_name
            self.config["vectorstore_created"] = metadata["created_at"]
            self.save_config()
            
            logger.info(f"Saved vectorstore metadata: {model_name}")
        except Exception as e:
            logger.error(f"Error saving vectorstore metadata: {e}")
    
    def validate_compatibility(self) -> Dict[str, Any]:
        """Check if current configuration is compatible"""
        current_model = self.get_current_model()
        vectorstore_model = self.config.get("vectorstore_model")
        
        result = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "current_model": current_model,
            "vectorstore_model": vectorstore_model
        }
        
        if not vectorstore_model:
            result["warnings"].append("Vectorstore model not detected - using fallback")
        elif current_model != vectorstore_model:
            current_dims = self.EMBEDDING_MODELS.get(current_model, {}).get("dimensions")
            vectorstore_dims = self.EMBEDDING_MODELS.get(vectorstore_model, {}).get("dimensions")
            
            if current_dims != vectorstore_dims:
                result["compatible"] = False
                result["errors"].append(
                    f"Dimension mismatch: current model ({current_model}) has {current_dims}D, "
                    f"but vectorstore was built with {vectorstore_dims}D ({vectorstore_model})"
                )
        
        return result
    
    def get_embedding_kwargs(self, model_name: str = None) -> Dict[str, Any]:
        """Get appropriate kwargs for OpenAIEmbeddings"""
        model_name = model_name or self.get_current_model()
        
        if model_name not in self.EMBEDDING_MODELS:
            logger.warning(f"Unknown model {model_name}, using fallback")
            model_name = self.config["fallback_model"]
        
        return {"model": model_name}
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models with their specs"""
        return self.EMBEDDING_MODELS.copy()
    
    def get_status_report(self) -> str:
        """Get a human-readable status report"""
        compatibility = self.validate_compatibility()
        current_model = self.get_current_model()
        specs = self.get_model_specs(current_model)
        
        report = [
            "🔧 Embedding Configuration Status",
            "=" * 40,
            f"Current Model: {current_model}",
            f"Dimensions: {specs.get('dimensions', 'Unknown')}",
            f"Vectorstore Model: {compatibility['vectorstore_model'] or 'Not detected'}",
            f"Compatible: {'✅ Yes' if compatibility['compatible'] else '❌ No'}",
        ]
        
        if compatibility["warnings"]:
            report.append("\n⚠️ Warnings:")
            for warning in compatibility["warnings"]:
                report.append(f"  • {warning}")
        
        if compatibility["errors"]:
            report.append("\n❌ Errors:")
            for error in compatibility["errors"]:
                report.append(f"  • {error}")
        
        return "\n".join(report)
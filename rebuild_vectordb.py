#!/usr/bin/env python3
"""
VectorDB Rebuild Script for Negotiation Advisor
Rebuilds the RAG vector database from all documents in ./sources
"""

import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from embedding_config import EmbeddingConfig

# Load environment variables
load_dotenv()

class VectorDBRebuilder:
    """Handles rebuilding the vector database from scratch"""
    
    def __init__(self, sources_dir: str = "sources", vectorstore_dir: str = "vectorstore"):
        self.sources_dir = Path(sources_dir)
        self.vectorstore_dir = Path(vectorstore_dir)
        self.supported_extensions = {'.pdf', '.txt', '.docx'}
        self.embedding_config = EmbeddingConfig()
        
        # Stats tracking
        self.stats = {
            "start_time": None,
            "total_files": 0,
            "processed_files": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "embedding_tokens": 0,
            "estimated_cost": 0.0,
            "errors": []
        }
    
    def print_header(self):
        """Print script header"""
        print("=" * 80)
        print("🔄 NEGOTIATION ADVISOR - VECTOR DATABASE REBUILD")
        print("=" * 80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Sources: {self.sources_dir.absolute()}")
        print(f"🎯 Target: {self.vectorstore_dir.absolute()}")
        print()
    
    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        print("🔍 Checking environment...")
        
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY not found in environment")
            print("   Please set it in .env file or environment variables")
            return False
        print("✅ OpenAI API key found")
        
        # Check sources directory
        if not self.sources_dir.exists():
            print(f"❌ Sources directory not found: {self.sources_dir}")
            return False
        print(f"✅ Sources directory found: {self.sources_dir}")
        
        return True
    
    def select_embedding_model(self) -> str:
        """Let user select embedding model"""
        print("\n🎯 SELECT EMBEDDING MODEL")
        print("-" * 40)
        
        models = self.embedding_config.list_available_models()
        model_names = list(models.keys())
        
        for i, (model_name, info) in enumerate(models.items(), 1):
            status = "⭐ RECOMMENDED" if info["recommended"] else ""
            print(f"{i}. {model_name} {status}")
            print(f"   📏 Dimensions: {info['dimensions']}")
            print(f"   📝 {info['description']}")
            print(f"   💰 Cost: ${info['cost_per_1k']:.5f} per 1K tokens")
            print()
        
        while True:
            try:
                choice = input(f"Select model (1-{len(model_names)}) [1 for recommended]: ").strip()
                if not choice:
                    choice = "1"  # Default to recommended
                
                idx = int(choice) - 1
                if 0 <= idx < len(model_names):
                    selected_model = model_names[idx]
                    print(f"✅ Selected: {selected_model}")
                    return selected_model
                else:
                    print(f"❌ Please enter a number between 1 and {len(model_names)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def scan_documents(self) -> List[Dict[str, Any]]:
        """Scan sources directory for supported documents"""
        print("\n📚 Scanning documents...")
        documents = []
        
        for file_path in self.sources_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    stat = file_path.stat()
                    doc_info = {
                        "path": file_path,
                        "name": file_path.name,
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "extension": file_path.suffix.lower(),
                        "modified": datetime.fromtimestamp(stat.st_mtime)
                    }
                    documents.append(doc_info)
                except Exception as e:
                    self.stats["errors"].append(f"Error scanning {file_path.name}: {e}")
        
        documents.sort(key=lambda x: x["name"])
        
        print(f"📊 Found {len(documents)} documents:")
        total_size = sum(doc["size"] for doc in documents)
        print(f"   📁 Total size: {total_size / (1024*1024):.1f} MB")
        
        # Group by type
        by_type = {}
        for doc in documents:
            ext = doc["extension"]
            if ext not in by_type:
                by_type[ext] = {"count": 0, "size": 0}
            by_type[ext]["count"] += 1
            by_type[ext]["size"] += doc["size"]
        
        for ext, info in by_type.items():
            print(f"   {ext.upper()}: {info['count']} files ({info['size']/(1024*1024):.1f} MB)")
        
        self.stats["total_files"] = len(documents)
        return documents
    
    def load_document(self, doc_info: Dict[str, Any]) -> List[Any]:
        """Load a single document"""
        file_path = doc_info["path"]
        ext = doc_info["extension"]
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif ext == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif ext == '.docx':
                loader = Docx2txtLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': doc_info["name"],
                    'file_type': ext[1:].upper(),
                    'file_size': doc_info["size"],
                    'modified_date': doc_info["modified"].isoformat()
                })
            
            return documents
            
        except Exception as e:
            error_msg = f"Error loading {doc_info['name']}: {e}"
            self.stats["errors"].append(error_msg)
            print(f"   ❌ {error_msg}")
            return []
    
    def process_documents(self, document_list: List[Dict[str, Any]]) -> List[Any]:
        """Process all documents and return chunks"""
        print(f"\n📖 Processing {len(document_list)} documents...")
        
        all_documents = []
        
        for i, doc_info in enumerate(document_list, 1):
            print(f"   [{i}/{len(document_list)}] {doc_info['name']} ({doc_info['size_mb']} MB)")
            
            start_time = time.time()
            documents = self.load_document(doc_info)
            
            if documents:
                all_documents.extend(documents)
                self.stats["processed_files"] += 1
                self.stats["total_pages"] += len(documents)
                
                elapsed = time.time() - start_time
                print(f"       ✅ Loaded {len(documents)} pages in {elapsed:.2f}s")
            else:
                print(f"       ❌ Failed to load")
        
        print(f"\n📄 Total pages loaded: {self.stats['total_pages']}")
        
        # Create text chunks
        print("🔪 Creating text chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        start_time = time.time()
        chunks = text_splitter.split_documents(all_documents)
        elapsed = time.time() - start_time
        
        self.stats["total_chunks"] = len(chunks)
        print(f"✅ Created {len(chunks)} chunks in {elapsed:.2f}s")
        
        return chunks
    
    def estimate_cost(self, chunks: List[Any], model_name: str) -> float:
        """Estimate embedding cost"""
        total_tokens = sum(len(chunk.page_content.split()) for chunk in chunks)
        # Rough estimate: 1 token ≈ 0.75 words
        estimated_tokens = int(total_tokens / 0.75)
        
        model_specs = self.embedding_config.get_model_specs(model_name)
        cost_per_1k = model_specs.get("cost_per_1k", 0.0001)  # Fallback cost
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k
        
        self.stats["embedding_tokens"] = estimated_tokens
        self.stats["estimated_cost"] = estimated_cost
        
        print(f"\n💰 Cost Estimation:")
        print(f"   📊 Estimated tokens: {estimated_tokens:,}")
        print(f"   💵 Estimated cost: ${estimated_cost:.4f}")
        
        return estimated_cost
    
    def confirm_rebuild(self) -> bool:
        """Ask user to confirm rebuild"""
        print(f"\n⚠️  REBUILD CONFIRMATION")
        print("-" * 30)
        
        if self.vectorstore_dir.exists():
            print(f"🗑️  This will DELETE existing vector database at: {self.vectorstore_dir}")
        
        print(f"📁 Will process {self.stats['total_files']} files")
        print(f"📄 {self.stats['total_pages']} pages → {self.stats['total_chunks']} chunks")
        print(f"💰 Estimated cost: ${self.stats['estimated_cost']:.4f}")
        
        if self.stats["errors"]:
            print(f"\n⚠️  {len(self.stats['errors'])} errors occurred during scanning:")
            for error in self.stats["errors"][:3]:  # Show first 3 errors
                print(f"   • {error}")
            if len(self.stats["errors"]) > 3:
                print(f"   • ... and {len(self.stats['errors']) - 3} more")
        
        while True:
            response = input(f"\n🤔 Proceed with rebuild? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("❌ Please enter 'y' or 'n'")
    
    def backup_existing(self):
        """Backup existing vectorstore if it exists"""
        if self.vectorstore_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"vectorstore_backup_{timestamp}")
            
            print(f"💾 Backing up existing vectorstore to: {backup_dir}")
            shutil.copytree(self.vectorstore_dir, backup_dir)
            shutil.rmtree(self.vectorstore_dir)
            print("✅ Backup completed")
    
    def create_vectorstore(self, chunks: List[Any], model_name: str):
        """Create the vector database"""
        print(f"\n🏗️  Creating vector database with {model_name}...")
        
        # Initialize embeddings with selected model
        embeddings = OpenAIEmbeddings(model=model_name)
        logger.info(f"Using embedding model: {model_name}")
        
        start_time = time.time()
        
        try:
            # Create FAISS vectorstore
            print("⚙️  Generating embeddings...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Save to disk
            print("💾 Saving vectorstore...")
            vectorstore.save_local(str(self.vectorstore_dir))
            
            # Save metadata about the build
            print("📝 Saving vectorstore metadata...")
            self.embedding_config.save_vectorstore_metadata(model_name, len(chunks))
            
            elapsed = time.time() - start_time
            print(f"✅ Vector database created successfully in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"❌ Error creating vectorstore: {e}")
            raise
    
    def print_summary(self):
        """Print final summary"""
        total_time = time.time() - self.stats["start_time"]
        
        print("\n" + "=" * 80)
        print("📊 REBUILD SUMMARY")
        print("=" * 80)
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📁 Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
        print(f"📄 Pages processed: {self.stats['total_pages']}")
        print(f"🧩 Chunks created: {self.stats['total_chunks']}")
        print(f"🏗️  Vector database: {self.vectorstore_dir}")
        print(f"💰 Estimated cost: ${self.stats['estimated_cost']:.4f}")
        
        if self.stats["errors"]:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                print(f"   • {error}")
        
        print(f"\n✅ Rebuild completed successfully!")
        print("🚀 You can now restart the negotiation advisor to use the new database.")
    
    def run(self):
        """Main execution flow"""
        self.stats["start_time"] = time.time()
        
        try:
            self.print_header()
            
            # Environment checks
            if not self.check_environment():
                sys.exit(1)
            
            # Scan documents
            documents = self.scan_documents()
            if not documents:
                print("❌ No supported documents found in sources directory")
                sys.exit(1)
            
            # Model selection
            model_name = self.select_embedding_model()
            
            # Process documents
            chunks = self.process_documents(documents)
            if not chunks:
                print("❌ No content extracted from documents")
                sys.exit(1)
            
            # Cost estimation
            self.estimate_cost(chunks, model_name)
            
            # Confirmation
            if not self.confirm_rebuild():
                print("🛑 Rebuild cancelled by user")
                sys.exit(0)
            
            # Backup existing
            self.backup_existing()
            
            # Create new vectorstore
            self.create_vectorstore(chunks, model_name)
            
            # Summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n🛑 Rebuild interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    rebuilder = VectorDBRebuilder()
    rebuilder.run()
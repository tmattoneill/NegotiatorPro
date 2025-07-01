import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
import glob

# Load environment variables
load_dotenv()

def test_pdf_loading():
    print("Testing PDF loading...")
    pdf_files = glob.glob("sources/*.pdf")
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    if pdf_files:
        print(f"\nTesting load of first PDF: {pdf_files[0]}")
        loader = PyPDFLoader(pdf_files[0])
        documents = loader.load()
        print(f"Loaded {len(documents)} pages")
        if documents:
            print(f"First page preview: {documents[0].page_content[:200]}...")

def test_embeddings():
    print("\nTesting OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    test_text = "This is a test sentence for embedding."
    try:
        embedding = embeddings.embed_query(test_text)
        print(f"Embedding successful, dimension: {len(embedding)}")
    except Exception as e:
        print(f"Embedding failed: {e}")

if __name__ == "__main__":
    test_pdf_loading()
    test_embeddings()
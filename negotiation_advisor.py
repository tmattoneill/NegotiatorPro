import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob
import pickle
import logging
import time

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

class ModelConfig:
    """Model configuration middleware to handle different model parameters"""
    
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": None
        },
        "o3-mini": {
            "model": "o3-mini",
            # o3 models don't support temperature parameter
        },
        "gpt-4": {
            "model": "gpt-4",
            "temperature": 0.3,
            "max_tokens": None
        },
        "gpt-3.5-turbo": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.3,
            "max_tokens": None
        }
    }
    
    @staticmethod
    def get_model_kwargs(model_name):
        """Get appropriate kwargs for a specific model"""
        if model_name not in ModelConfig.MODEL_CONFIGS:
            logger.warning(f"Unknown model {model_name}, using default config")
            return {"model": model_name, "temperature": 0.3}
        
        config = ModelConfig.MODEL_CONFIGS[model_name].copy()
        # Filter out None values to avoid passing them to ChatOpenAI
        config = {k: v for k, v in config.items() if v is not None}
        logger.info(f"Using config for {model_name}: {config}")
        return config

class NegotiationRAG:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.vectorstore_path = "negotiation_vectorstore.pkl"
        
    def load_documents(self):
        """Load and process PDF documents"""
        logger.info("Starting PDF document loading...")
        docs = []
        pdf_files = glob.glob("sources/*.pdf")
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing {pdf_file} ({i+1}/{len(pdf_files)})")
            start_time = time.time()
            
            try:
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                
                # Add source metadata
                for doc in documents:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)
                
                docs.extend(documents)
                end_time = time.time()
                logger.info(f"Loaded {len(documents)} pages from {pdf_file} in {end_time-start_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
            
        logger.info(f"Total documents loaded: {len(docs)} pages")
        return docs
    
    def create_chunks(self, documents):
        """Split documents into chunks"""
        logger.info("Starting text chunking...")
        start_time = time.time()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        end_time = time.time()
        logger.info(f"Created {len(chunks)} text chunks in {end_time-start_time:.2f}s")
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create FAISS vectorstore from document chunks"""
        logger.info("Starting vector embedding creation...")
        start_time = time.time()
        
        try:
            logger.info("Initializing OpenAI embeddings...")
            embeddings = OpenAIEmbeddings()
            
            logger.info(f"Creating FAISS vectorstore from {len(chunks)} chunks...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            end_time = time.time()
            logger.info(f"Vector store created successfully in {end_time-start_time:.2f}s")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
    
    def save_vectorstore(self):
        """Save vectorstore to disk"""
        if self.vectorstore:
            try:
                logger.info("Saving vectorstore to disk...")
                self.vectorstore.save_local("vectorstore")
                logger.info("Vector store saved successfully")
            except Exception as e:
                logger.error(f"Error saving vectorstore: {e}")
    
    def load_vectorstore(self):
        """Load vectorstore from disk"""
        try:
            logger.info("Attempting to load existing vectorstore...")
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully from disk")
            return True
        except Exception as e:
            logger.info(f"No existing vector store found: {e}")
            return False
    
    def setup_system(self):
        """Initialize the RAG system"""
        logger.info("Starting RAG system setup...")
        setup_start = time.time()
        
        # Try to load existing vectorstore first
        if not self.load_vectorstore():
            logger.info("Creating new vectorstore...")
            # Create new vectorstore
            documents = self.load_documents()
            chunks = self.create_chunks(documents)
            self.vectorstore = self.create_vectorstore(chunks)
            self.save_vectorstore()
        
        logger.info("Setting up QA chain...")
        # Create custom prompt for negotiation guidance with system instructions
        prompt_template = """You are a skilled sales negotiator and expert advisor, leveraging principles from mainstream negotiation books like 'Getting Past No', 'The Upward Spiral', 'Getting to Yes', 'Never Split the Difference', and 'How to Win Friends and Influence People.' You carefully analyze client communications to craft empathetic yet assertive responses.

Your goal is to find win-win solutions while ensuring the best outcomes for the user. You prioritize understanding the client's needs and concerns, and help apply negotiation strategies like building rapport, uncovering underlying interests, and leveraging tactical empathy. You must avoid being overly aggressive or dismissive of client positions, always focusing on maintaining positive relationships while negotiating favorable terms.

Context from negotiation materials:
{context}

Negotiation Question: {question}

Each response MUST include:
• A detailed breakdown of the negotiation so far and piece-by-piece analysis of the client's communication
• A fully composed draft response to the client when appropriate
• A bullet list of calibrated questions to use in the negotiation
• Potential client responses and suggested actions for each scenario
• PLEASE Framework self-assessment (Polite, Logical, Empathetic, Assertive, Strategic, Engaging - score each /5)

Your responses must be POLITE, LOGICAL, EMPATHETIC, ASSERTIVE, STRATEGIC, and ENGAGING. The tone should remain professional but non-formal to foster ease and approachability.

Provide comprehensive negotiation guidance:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chains for both models with model-specific parameters
        logger.info("Initializing ChatOpenAI and retrieval chains...")
        
        # Use ModelConfig middleware to get appropriate parameters
        default_config = ModelConfig.get_model_kwargs("gpt-4o-mini")
        premium_config = ModelConfig.get_model_kwargs("o3-mini")
        
        self.default_llm = ChatOpenAI(**default_config)
        self.premium_llm = ChatOpenAI(**premium_config)
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        chain_kwargs = {"prompt": PROMPT}
        
        self.default_qa_chain = RetrievalQA.from_chain_type(
            llm=self.default_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_kwargs
        )
        
        self.premium_qa_chain = RetrievalQA.from_chain_type(
            llm=self.premium_llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_kwargs
        )
        
        setup_end = time.time()
        logger.info(f"RAG system setup complete in {setup_end-setup_start:.2f}s!")
    
    def get_advice(self, question, use_premium_model=False):
        """Get negotiation advice based on the question"""
        if not hasattr(self, 'default_qa_chain') or not hasattr(self, 'premium_qa_chain'):
            return "System not initialized properly."
        
        try:
            # Select appropriate QA chain based on model choice
            if use_premium_model:
                qa_chain = self.premium_qa_chain
                logger.info("Using o3-mini model for this query")
            else:
                qa_chain = self.default_qa_chain
                logger.info("Using gpt-4o-mini model for this query")
            
            response = qa_chain.run(question)
            return response
        except Exception as e:
            logger.error(f"Error getting advice: {e}")
            return f"Error getting advice: {str(e)}"

def create_gradio_interface(rag_system):
    """Create and return Gradio interface"""
    
    def negotiate_advisor(question, partner_context="", use_premium=False):
        """Main function for the Gradio interface"""
        if partner_context.strip():
            enhanced_question = f"Context about my negotiation partner: {partner_context}\n\nMy question: {question}"
        else:
            enhanced_question = question
        
        if not question.strip():
            return "Please enter a negotiation question.", "Ready: Please enter a question"
        
        # Update model status
        model_name = "o3-mini" if use_premium else "gpt-4o-mini"
        status = f"Processing with {model_name}..."
        
        advice = rag_system.get_advice(enhanced_question, use_premium_model=use_premium)
        
        # Update final status
        final_status = f"Completed with {model_name}"
        
        return advice, final_status
    
    # Create Gradio interface
    with gr.Blocks(title="Negotiation Tactics Advisor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤝 Negotiation Tactics Advisor")
        gr.Markdown("Get expert negotiation guidance based on proven strategies from leading negotiation books including 'Getting to Yes', 'Never Split the Difference', and more.")
        
        with gr.Row():
            with gr.Column(scale=1):
                partner_info = gr.Textbox(
                    label="About Your Negotiation Partner (Optional)",
                    placeholder="e.g., Experienced executive, tends to be aggressive, budget-conscious, deadline pressure...",
                    lines=3
                )
                question = gr.Textbox(
                    label="Your Negotiation Question",
                    placeholder="e.g., How should I handle a lowball offer? What's the best way to anchor the price?",
                    lines=4
                )
                
                with gr.Row():
                    use_premium_model = gr.Checkbox(
                        label="🚀 Use o3-mini (Premium Model)",
                        value=False,
                        info="Default: gpt-4o-mini (faster, cost-effective) | Premium: o3-mini (more advanced reasoning, no temperature control)"
                    )
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Ready: gpt-4o-mini (default)",
                    interactive=False,
                    max_lines=1
                )
                
                submit_btn = gr.Button("Get Negotiation Advice", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                advice_output = gr.Textbox(
                    label="Negotiation Advice",
                    lines=20,
                    interactive=False
                )
        
        submit_btn.click(
            fn=negotiate_advisor,
            inputs=[question, partner_info, use_premium_model],
            outputs=[advice_output, model_status]
        )
        
        # Example questions section
        gr.Markdown("### Example Questions You Can Ask:")
        
        example_questions = [
            "How do I respond to 'That's my final offer'?",
            "What's the best way to make the first offer?",
            "How can I build rapport with a difficult negotiator?",
            "They're using high-pressure tactics. What should I do?",
            "How do I negotiate when I have less leverage?",
            "What questions should I ask to understand their interests?",
            "How do I handle emotional manipulation in negotiations?",
            "What's the best way to counter anchoring tactics?",
            "How do I create win-win solutions?",
            "What should I do when they walk away from the table?"
        ]
        
        with gr.Row():
            for i in range(0, len(example_questions), 2):
                with gr.Column():
                    if i < len(example_questions):
                        def create_example_handler(example_text):
                            def handler():
                                return example_text, "", False, "Ready: gpt-4o-mini (default)"
                            return handler
                        
                        gr.Button(example_questions[i], size="sm").click(
                            create_example_handler(example_questions[i]),
                            outputs=[question, partner_info, use_premium_model, model_status]
                        )
                    if i+1 < len(example_questions):
                        def create_example_handler2(example_text):
                            def handler():
                                return example_text, "", False, "Ready: gpt-4o-mini (default)"
                            return handler
                        
                        gr.Button(example_questions[i+1], size="sm").click(
                            create_example_handler2(example_questions[i+1]),
                            outputs=[question, partner_info, use_premium_model, model_status]
                        )
        
        gr.Markdown("---")
        gr.Markdown("*This advisor draws from negotiation expertise in your PDF library. Always adapt advice to your specific situation and legal/ethical constraints.*")
    
    return demo

if __name__ == "__main__":
    logger.info("=== Starting Negotiation RAG System ===")
    
    try:
        logger.info("Initializing RAG system...")
        rag_system = NegotiationRAG()
        rag_system.setup_system()
        
        logger.info("Creating Gradio interface...")
        demo = create_gradio_interface(rag_system)
        
        logger.info("Launching Gradio application...")
        demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob

# Load environment variables
load_dotenv()

class NegotiationRAG:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.setup_system()
    
    def load_documents(self):
        """Load and process PDF documents"""
        docs = []
        pdf_files = glob.glob("sources/*.pdf")
        
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            docs.extend(documents)
        
        return docs
    
    def create_chunks(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def create_vectorstore(self, chunks):
        """Create FAISS vectorstore from document chunks"""
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    
    def setup_system(self):
        """Initialize the RAG system"""
        print("Loading PDF documents...")
        documents = self.load_documents()
        print(f"Loaded {len(documents)} document pages")
        
        print("Creating text chunks...")
        chunks = self.create_chunks(documents)
        print(f"Created {len(chunks)} text chunks")
        
        print("Creating vector embeddings...")
        self.vectorstore = self.create_vectorstore(chunks)
        print("Vector store created successfully")
        
        # Create custom prompt for negotiation guidance
        prompt_template = """You are an expert negotiation advisor. Use the following context from negotiation books and materials to provide specific, actionable guidance.

Context: {context}

Question: {question}

Based on the negotiation principles and tactics from the provided materials, give detailed advice that includes:
1. Specific strategies or techniques to use
2. What to say or how to phrase things
3. What to watch out for or avoid
4. How to prepare for this situation

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0.3),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("RAG system setup complete!")
    
    def get_advice(self, question):
        """Get negotiation advice based on the question"""
        if not self.qa_chain:
            return "System not initialized properly."
        
        try:
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            return f"Error getting advice: {str(e)}"

# Initialize the RAG system
rag_system = NegotiationRAG()

def negotiate_advisor(question, partner_context=""):
    """Main function for the Gradio interface"""
    if partner_context:
        enhanced_question = f"My negotiation partner: {partner_context}\n\nMy question: {question}"
    else:
        enhanced_question = question
    
    advice = rag_system.get_advice(enhanced_question)
    return advice

# Create Gradio interface
with gr.Blocks(title="Negotiation Tactics Advisor") as demo:
    gr.Markdown("# 🤝 Negotiation Tactics Advisor")
    gr.Markdown("Get expert negotiation guidance based on proven strategies from leading negotiation books.")
    
    with gr.Row():
        with gr.Column():
            partner_info = gr.Textbox(
                label="About Your Negotiation Partner (Optional)",
                placeholder="e.g., Experienced executive, tends to be aggressive, budget-conscious...",
                lines=3
            )
            question = gr.Textbox(
                label="Your Negotiation Question",
                placeholder="e.g., How should I handle a lowball offer? What's the best way to anchor the price?",
                lines=4
            )
            submit_btn = gr.Button("Get Negotiation Advice", variant="primary")
        
        with gr.Column():
            advice_output = gr.Textbox(
                label="Negotiation Advice",
                lines=15,
                interactive=False
            )
    
    submit_btn.click(
        fn=negotiate_advisor,
        inputs=[question, partner_info],
        outputs=advice_output
    )
    
    # Example questions
    gr.Markdown("### Example Questions:")
    examples = [
        "How do I respond to 'That's my final offer'?",
        "What's the best way to make the first offer?",
        "How can I build rapport with a difficult negotiator?",
        "They're using high-pressure tactics. What should I do?",
        "How do I negotiate when I have less leverage?",
        "What questions should I ask to understand their interests?"
    ]
    
    for example in examples:
        gr.Button(example, size="sm").click(
            lambda x=example: x,
            outputs=question
        )

if __name__ == "__main__":
    demo.launch(share=True)
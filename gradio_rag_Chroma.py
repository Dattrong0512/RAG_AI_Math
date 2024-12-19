import os
import warnings
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from pdf2image import convert_from_path
import pytesseract
import subprocess


class MultiFileChatbot:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.loaded_documents = []
        self.custom_system_prompt = "You are a helpful AI assistant specialized in analyzing documents."
        self.libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"  # Đường dẫn đầy đủ tới soffice.exe

        # Cấu hình đường dẫn tới Tesseract OCR
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text_from_image_pdf(self, file_path: str) -> str:
        """Extract text from image-based PDF using OCR."""
        text = ""
        poppler_path = r"C:\Program Files\poppler\poppler-24.08.0\Library\bin"  # Đường dẫn Poppler
        try:
            # Convert PDF pages to images
            print(f"Converting PDF to images for OCR: {file_path}")
            images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
            for i, image in enumerate(images):
                print(f"Processing page {i + 1} with OCR...")
                text += pytesseract.image_to_string(image, lang="eng") + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF with images: {e}")
        return text

    def load_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files (PDF, TXT, DOCX, DOC) and return their documents."""
        all_documents = []

        try:
            for file_path in file_paths:
                if file_path.endswith(".pdf"):
                    print(f"Processing PDF file: {file_path}")
                    text = self.extract_text_from_image_pdf(file_path)
                    if text.strip():
                        all_documents.append(Document(page_content=text, metadata={"source": file_path}))
                    else:
                        print(f"No text extracted from {file_path}")
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    all_documents.extend(loader.load())
                elif file_path.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    all_documents.extend(loader.load())
                elif file_path.endswith(".doc"):
                    if not self.check_libreoffice():
                        print(f"LibreOffice is not installed. Cannot process file: {file_path}")
                        continue

                    docx_path = self.convert_doc_to_docx(file_path)
                    if docx_path:
                        loader = Docx2txtLoader(docx_path)
                        all_documents.extend(loader.load())
                    else:
                        print(f"Failed to convert .doc to .docx: {file_path}")
                        continue
                else:
                    print(f"Unsupported file type: {file_path}")
                    continue

                print(f"Loaded file: {file_path}")

            return all_documents
        except Exception as e:
            print(f"Error loading files: {e}")
            return []

    def split_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> Optional[Chroma]:
        """Create a vector store from documents."""
        try:
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
            print(f"Vector store created successfully with {len(documents)} documents.")
            return self.vector_store
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return None

    def create_qa_chain(self, system_prompt: Optional[str] = None) -> Optional[RetrievalQA]:
        """Create a question-answering chain with optional custom system prompt."""
        if not self.vector_store:
            return None

        try:
            effective_system_prompt = system_prompt or self.custom_system_prompt

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                temperature=0.1,
                convert_system_message_to_human=True,
                system_prompt=effective_system_prompt,
                model_kwargs={"max_output_tokens": 8192, "top_k": 10, "top_p": 0.95}
            )

            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
            )
            return self.qa_chain
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            return None

    def process_query(self, query: str) -> str:
        """Process query against loaded documents."""
        if not self.qa_chain:
            return "Error: QA chain not initialized. Please load documents first."

        try:
            response = self.qa_chain.invoke({"query": query})
            top_chunks = "\n\n".join([
                f"Chunk (Page {doc.metadata.get('page', 'N/A')}):\n{doc.page_content[:500]}"
                for doc in response['source_documents']
            ])

            return f"Top Relevant Chunks:\n{top_chunks}\n\nFinal Answer:\n{response['result']}"
        except Exception as e:
            return f"Error processing query: {e}"

    def initialize_files(self, file_paths, custom_system_prompt: Optional[str] = None):
        """Initialize vector store and QA chain from multiple files."""
        self.loaded_documents = []

        documents = self.load_files(file_paths)
        if not documents:
            return "Error: Could not load files"

        split_docs = self.split_documents(documents)
        self.loaded_documents = split_docs

        if not self.create_vector_store(split_docs):
            return "Error: Could not create vector store"

        if custom_system_prompt:
            self.custom_system_prompt = custom_system_prompt

        if not self.create_qa_chain(custom_system_prompt):
            return "Error: Could not create QA chain"

        file_names = [os.path.basename(f) for f in file_paths]
        return f"Files loaded and processed successfully!\nLoaded files: {', '.join(file_names)}\n" \
               f"Total document chunks: {len(split_docs)}"


def launch_gradio():
    """Launch Gradio interface for Multi-File Q&A."""
    chatbot = MultiFileChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Multi-File Q&A Chatbot")

        with gr.Row():
            with gr.Column():
                file_inputs = gr.File(
                    label="Upload Files",
                    file_types=['.pdf', '.txt', '.docx', '.doc'],
                    file_count="multiple"
                )
                system_prompt_input = gr.Textbox(
                    label="Custom System Prompt (Optional)",
                    placeholder="Enter a custom instruction for the AI...",
                    lines=3
                )
                load_btn = gr.Button("Load Files")
                status_output = gr.Textbox(label="Status")

            with gr.Column():
                query_input = gr.Textbox(label="Ask a Question")
                submit_btn = gr.Button("Ask Question")
                output = gr.Textbox(label="Response", lines=10)

        load_btn.click(
            fn=chatbot.initialize_files,
            inputs=[file_inputs, system_prompt_input],
            outputs=[status_output]
        )

        submit_btn.click(
            fn=chatbot.process_query,
            inputs=[query_input],
            outputs=[output]
        )

    demo.launch()


def main():
    launch_gradio()


if __name__ == "__main__":
    main()
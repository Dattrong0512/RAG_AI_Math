import os
import warnings
from typing import List
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

import gradio as gr
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from pdf2image import convert_from_path
import pytesseract
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import subprocess
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_core.documents import Document


class MilvusHandler:
    def __init__(self, collection_name: str, embedding_dim: int = 768):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None

    def connect(self):
        connections.connect(host="localhost", port="19530")
        print("Connected to Milvus")

    def create_collection(self):
        # Connect to Milvus
        self.connect()
        try:
            # Kiểm tra nếu collection đã tồn tại bằng cách lấy schema của collection
            self.collection = Collection(name=self.collection_name)
            if self.collection.schema is not None:
                print(f"Collection '{self.collection_name}' already exists. Using existing collection.")
                return
        except Exception:
            # Nếu có lỗi khi lấy schema, nghĩa là collection chưa tồn tại, tiến hành tạo mới
            id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000)
            vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)

            schema = CollectionSchema(fields=[id_field, text_field, vector_field], description="Chatbot Collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created successfully.")


    def insert_data(self, texts: List[str], vectors: List[List[float]]):
        if not self.collection:
            print("Collection not initialized.")
            return
        result = self.collection.insert([texts, vectors])
        print(f"Inserted {len(texts)} records into '{self.collection_name}'.")

        # Tạo chỉ mục cho trường vector
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        self.collection.create_index(field_name="vector", index_params=index_params)
        print("Index created for the 'vector' field.")

        # Tải bộ sưu tập vào bộ nhớ
        self.collection.load()
        print("Collection loaded into memory.")

        # Truy vấn và in dữ liệu
        results = self.collection.query(expr="id >= 0", output_fields=["id", "text", "vector"])
        for record in results:
            print(f"ID: {record['id']}, Text: {record['text'][:50]}..., Vector: {record['vector'][:5]}...")



class FileProcessor:
    def __init__(self):
        self.milvus_handler = MilvusHandler(collection_name="ChatbotCollection")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.libreoffice_path = r"C:\\Program Files\\LibreOffice\\program\\soffice.exe"
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from image-based PDF using OCR."""
        text = ""
        poppler_path = r"C:\\Program Files\\poppler\\poppler-24.08.0\\Library\\bin"  # Đường dẫn Poppler
        try:
            print(f"Converting PDF to images for OCR: {file_path}")
            images = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)
            for i, image in enumerate(images):
                print(f"Processing page {i + 1} with OCR...")
                text += pytesseract.image_to_string(image, lang="eng") + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF with images: {e}")
        return text

    def convert_doc_to_docx(self, file_path: str) -> str:
        output_dir = os.path.dirname(file_path)
        try:
            subprocess.run([
                self.libreoffice_path, "--headless", "--convert-to", "docx", file_path, "--outdir", output_dir
            ], check=True)
            return file_path.replace(".doc", ".docx")
        except Exception as e:
            print(f"Error converting DOC to DOCX: {e}")
            return ""

    def load_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files (PDF, TXT, DOCX, DOC) and return their documents."""
        all_documents = []

        try:
            for file_path in file_paths:
                if file_path.endswith(".pdf"):
                    print(f"Processing PDF file: {file_path}")
                    text = self.extract_text_from_pdf(file_path)
                    if text.strip():
                        all_documents.append(Document(page_content=text, metadata={"source": file_path}))
                elif file_path.endswith(".txt"):
                    loader = TextLoader(file_path, encoding="utf-8")
                    all_documents.extend(loader.load())
                elif file_path.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                    all_documents.extend(loader.load())
                elif file_path.endswith(".doc"):
                    print(f"Converting DOC to DOCX: {file_path}")
                    docx_path = self.convert_doc_to_docx(file_path)
                    if docx_path:
                        loader = Docx2txtLoader(docx_path)
                        all_documents.extend(loader.load())
                    else:
                        print(f"Failed to convert {file_path}")
                else:
                    print(f"Unsupported file type: {file_path}")
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

    def process_files(self, file_paths: List[str]):
        all_documents = self.load_files(file_paths)
        split_docs = self.split_documents(all_documents)

        texts = [doc.page_content for doc in split_docs]
        vectors = [self.embeddings.embed_query(text) for text in texts]

        self.milvus_handler.connect()
        self.milvus_handler.create_collection()
        self.milvus_handler.insert_data(texts, vectors)
        return "Files processed, vectors created, and data saved to Milvus."


def upload_and_process(file_paths):
    processor = FileProcessor()
    return processor.process_files(file_paths)


with gr.Blocks() as demo:
    gr.Markdown("## Upload Files to Save into Milvus")

    file_input = gr.File(file_types=[".pdf", ".txt", ".docx", ".doc"], file_count="multiple", label="Upload Files")
    output = gr.Textbox(label="Status")
    upload_button = gr.Button("Process and Save")
    upload_button.click(upload_and_process, inputs=file_input, outputs=output)

demo.launch()

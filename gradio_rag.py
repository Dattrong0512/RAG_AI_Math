import os
import warnings
from typing import List
from dotenv import load_dotenv
from pymilvus import connections, Collection
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import gradio as gr

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')


class MilvusChatbot:
    def __init__(self, collection_name="ChatbotCollection"):
        self.collection_name = collection_name
        self.collection = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.system_prompt = "You are a helpful AI assistant specialized in answering questions based on database content."
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.1,
            convert_system_message_to_human=True,
            system_prompt=self.system_prompt,
            model_kwargs={"max_output_tokens": 8192, "top_k": 10, "top_p": 0.95}
        )
        self.connect_to_milvus()

    def connect_to_milvus(self):
        """Connect to Milvus and load the collection."""
        connections.connect(host="localhost", port="19530")
        print("Connected to Milvus")
        try:
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
            print(f"Collection '{self.collection_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading collection '{self.collection_name}': {e}")

    def query_milvus(self, query_text: str) -> str:
        """Embed the query and search Milvus for relevant results."""
        if not self.collection:
            return "Collection not initialized. Please check the Milvus connection."

        try:
            # Embed the query
            query_vector = self.embeddings.embed_query(query_text)

            print(f"Query vector created for text: {query_text}")

            # Search in Milvus
            search_params = {"metric_type": "L2","offset": 0,"ignore_growing":
                             False, "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=50,
                expr=None,
                output_fields=["text"]
            )

            # Check if results exist
            if not results or len(results[0]) == 0:
                return "No relevant documents found in the database."

            context = []
            # Process results
            for idx,hit in enumerate(results[0]):
                score = hit.distance
                description = hit.entity.text
                print(f"Result {idx+1}: Score: {score}, Description: {description}")
                context_fragment = f"Document {idx + 1}:\nScore: {score:.4f}\nContent: {description[:200]}..."
                context.append(context_fragment)

            # Create a prompt for Gemini with context
            gemini_prompt = f"Context:\n{context}\n\nQuery: {query_text}\nAnswer:"
            gemini_response = self.llm.invoke(gemini_prompt)

            # Truy cập nội dung từ thuộc tính 'content' (thay vì ['result'])
            response_content = gemini_response.content if hasattr(gemini_response, 'content') else str(gemini_response)

            # Trả về phản hồi cùng với context
            return f"Gemini Response:\n{response_content}\n\nContext:\n{context}"

        except Exception as e:
            return f"Error querying Milvus: {e}"


def launch_chat_interface():
    """Launch a simple chat interface."""
    chatbot = MilvusChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("## 🌟 Milvus-Powered Chatbot with Gemini AI")

        with gr.Row():
            query_input = gr.Textbox(label="Ask your question:", placeholder="Type your query here...")
            submit_button = gr.Button("Submit")
            response_output = gr.Textbox(label="Response", lines=10)

        submit_button.click(
            fn=chatbot.query_milvus,
            inputs=[query_input],
            outputs=[response_output]
        )

    demo.launch()


if __name__ == "__main__":
    launch_chat_interface()

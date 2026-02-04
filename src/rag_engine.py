from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class HealthcareRAG:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vector_store = None
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0) # Low temperature for factual accuracy

    def ingest_documents(self, documents: List[Document]):
        """Splits documents and creates/updates the vector store."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        print(f"Ingested {len(splits)} chunks into vector store.")

    def get_answer(self, query: str) -> Dict[str, Any]:
        """Retrieves context and generates an answer with citations."""
        if not self.vector_store:
             # Try determining if one exists on disk
            if os.path.exists(self.persist_directory):
                 self.vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_function)
            else:
                return {"result": "Vector store not initialized. Please ingest documents first.", "source_documents": []}

        # Custom Prompt for Safety and Citations
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        
        IMPORTANT SAFETY GUIDELINES:
        - You are a helpful healthcare assistant, but you are NOT a doctor.
        - Do not provide medical diagnoses or prescribe treatments.
        - If the user describes severe symptoms, recommend seeking urgent professional care.
        - Always cite your sources from the context provided.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer (include citations in format [Source: Page X]):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa_chain.invoke({"query": query})
        return result

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class RAGSystem:
    def __init__(self):
        # Implementación correcta del modelo text-embedding-3-small
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Creación del vector store en memoria
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def process_documents(self, doc_paths):
        docs = []
        for path in doc_paths:
            # Cargamos los documentos en markdown
            loader = TextLoader(path, encoding='utf-8')
            docs.extend(loader.load())
        
        # Dividimos el texto para vectorizarlo adecuadamente
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # Almacenamos los embeddings en InMemoryVectorStore
        self.vector_store.add_documents(splits)

    def get_retriever(self):
        # Configurado para recuperar los fragmentos más relevantes
        return self.vector_store.as_retriever(search_kwargs={"k": 3})
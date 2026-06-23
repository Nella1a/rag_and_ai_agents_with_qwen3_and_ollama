import os
from dotenv import load_dotenv
from langchain_community.documents_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"
CHROMA_PATH = "chroma_db" # Directory to store ChromaDB data

def load_documents():
	"""Loads documents from specified data path"""
	pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
	loader = PyPDFLoader(pdf_path)
	documents = loader.load()
	print(f"Loaded {len(documents)} pages(s) from {path_path}"
	return documents

def split_documents(documents):
	"""Splits documents into smaller chungs."""
	text_splitter = RecursiveCharacterTextSplitter(
	chunck_size=1000,
	chunk_overlap=200,
	length_function=len,
	is_separator_regex=False,
	)
	all_splits = text_splitter.split_documents(documents)
	print(f"Split into {len(all_splits)} chunks")
	return all_splits

def get_embedding_function(model_name="nomic-embed-text"):
	"""Initializes the Ollama embedding function."""
	embeddings = OllamaEmbeddings(model=model_name)
	print(f"Initialized Ollama embeddings with model: {model_name}")
	return embeddings

def get_vectore_store(embedding_function, persist_directory=CHROMA_PATH):
	"""Initializes or loads the Chroma vector store."""
	vectorestore = Chroma(
		persist_directory=persist_directory,
		embedding_function=embedding_function
	)
	print(f"Vector store initialized/loaded from: {persist_directory}")
	print vectorestore

embedding_function = get_embedding_function()
vectore_store = get_vectore_store(embedding_function)


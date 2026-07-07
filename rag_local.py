import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()  # loads env variables from .env file

DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"
CHROMA_PATH = "chroma_db"  # Directory to store ChromaDB data

def load_documents():
	"""Loads documents from specified data path"""
	pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
	loader = PyPDFLoader(pdf_path)
	documents = loader.load()
	print(f"Loaded {len(documents)} pages(s) from {pdf_path}")
	return documents

def split_documents(documents):
	"""Splits documents into smaller chunks."""
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
	# Ensure Ollama server is running (via ollama serve)
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
	return vectorestore

def index_documents(chunks, embedding_function, persist_directory=CHROMA_PATH):
	""" Indexes document chunks into the Chroma vector store."""
	print(f"Indexing {len(chunks)} chunks ...")
	# Use from_documents for initial creation.
	# This will overwrite existing data if the directory exist but isn't a valid Chroma DB.
	# For incremental updates, initialize Chroma first and use vectorstore.add_documents()

	vectore_store = Chroma.from_documents(
		documents=chunks,
		embedding=embedding_function,
		persist_directory=persist_directory
	)

	vectore_store.persist() # Ensure data is saved
	print(f"/Indexing complete. Data saved to: {persist_directory}")
	return vectore_store

loaded_docs = load_documents()
chunks = split_documents(loaded_docs)

embedding_function = get_embedding_function()
vectore_store = get_vectore_store(embedding_function)
vectore_store = index_documents(chunks, embedding_function)


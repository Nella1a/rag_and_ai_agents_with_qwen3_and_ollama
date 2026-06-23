import os
from dotenv import load_dotenv
from langchain_community.documents_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"

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

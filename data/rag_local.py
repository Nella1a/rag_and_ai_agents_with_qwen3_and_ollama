import os
from dotenv import load_dotenv
from langchain_community.documents_loaders import PyPDFLoader

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

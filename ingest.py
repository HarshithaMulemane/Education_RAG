import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()


DATA_PATH = "data/course_notes.txt"
VECTORSTORE_PATH = "vectorstore/faiss_index"


def ingest_documents():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    print("Loading documents...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20
    )
    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings and FAISS vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)

    print("Vector store saved successfully")


if __name__ == "__main__":
    ingest_documents()
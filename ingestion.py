import pinecone
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ReadTheDocsLoader
import os
from dotenv import load_dotenv
load_dotenv()


pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT_REGION"))


def ingest_docs() -> None:
    """loader = ReadTheDocsLoader(
        path="/Users/sam/Desktop/readdocs/documentation_reader/documentation-helper/langchain-docs/api.python.langchain.com/en/latest/document_loaders")
    raw_documents = loader.load()
    print(f"Loaded: {len(raw_documents)} documents")
    """
    loader = TextLoader(
        "/Users/sam/Desktop/readdocs/documentation_reader/documentation-helper/GSARs.txt")
    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    """
    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace(
            "/Users/sam/Desktop/readdocs/documentation_reader/documentation-helper/langchain-docs/", "https://")
        doc.metadata.update({"source": new_url})
    """

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents=documents,
                            embedding=embeddings, index_name=os.getenv("INDEX_NAME"))

    print("****Added to Pinecone Vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()

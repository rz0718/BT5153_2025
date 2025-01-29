import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb
from langchain_openai import ChatOpenAI


# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def load_pdf_documents(pdf_directory: str) -> List:
    """Load PDF documents from a directory."""
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def process_documents(documents: List) -> List:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(documents: List) -> LanceDB:
    """Create LanceDB vector store."""
    # Initialize LanceDB
    db = lancedb.connect("lancedb")

    # Extract data
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Get embeddings
    embeddings_list = embeddings.embed_documents(texts)

    # Create data
    data = [
        {"id": str(i), "vector": emb, "text": text, "metadata": metadata}
        for i, (emb, text, metadata) in enumerate(
            zip(embeddings_list, texts, metadatas)
        )
    ]

    # Create table
    table = db.create_table("pdf_vectors", data=data, mode="overwrite")

    # Create vector store
    vector_store = LanceDB(
        connection=db,
        table_name="pdf_vectors",
        embedding=embeddings,
    )

    return vector_store


def create_rag_chain(llm, vector_store: LanceDB):
    """Create RAG chain."""
    # Initialize LLM

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # Create prompt template
    template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know.

    Context: {context}
    Question: {question}

    Answer:"""

    prompt = PromptTemplate.from_template(template)

    # Create the RAG chain
    def format_docs(docs):
        return "\n\n".join(
            f"{doc.page_content}\n(Source: {doc.metadata['source']}, Page: {doc.metadata['page']})"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )  # here, it is not chatmodel but the legacy llm model. So the output would be string directly
    # StrOutputParser() is used to parse the output of the llm to a string.
    # Here, it is the ChatModel. So it extracts the .content attribute of the message, ensuring that the final output is in string format.
    return rag_chain


def main():
    # Load documents
    print("Loading PDF documents...")
    documents = load_pdf_documents("offline_doc")

    # Process documents
    print("Processing documents...")
    processed_docs = process_documents(documents)

    # Create vector store
    print("Creating vector store...")
    vector_store = create_vectorstore(processed_docs)

    # Create RAG chain
    print("Creating RAG chain...")
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    rag_chain = create_rag_chain(llm, vector_store)

    # Example usage
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print("\nGenerating answer...")
        response = rag_chain.invoke(question)
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()

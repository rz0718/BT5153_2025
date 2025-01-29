import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import LanceDB
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import lancedb
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


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


def create_llm(model_id: str):
    """Create LLM using HuggingFacePipeline."""
    print("\nInitializing model and tokenizer...")
    print(
        "This might take a few minutes on first run as the model needs to be downloaded."
    )

    print("\nDownloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("\nDownloading model (this might take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="model_cache",  # Cache the model for future use
    )

    print("\nCreating pipeline...")
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
    )

    print("Model initialization complete!")

    # Create LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def create_rag_chain(model_id: str, vector_store: LanceDB):
    """Create RAG chain."""
    # Initialize LLM
    llm = create_llm(model_id)

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
        | llm.bind(
            skip_prompt=True
        )  # here, it is not chatmodel but the legacy llm model. So the output would be string directly
    )

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
    vector_store = create_vectorstoreLangChain_Basic(processed_docs)

    # Create RAG chain
    print("Creating RAG chain...")
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    rag_chain = create_rag_chain(model_id, vector_store)

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

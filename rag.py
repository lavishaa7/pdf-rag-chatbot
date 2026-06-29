from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


def load_and_split_pdf(file_path):
    """Load a PDF and split it into chunks. Raises ValueError if no
    extractable text is found (e.g. scanned/image-only PDFs)."""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("Could not read this PDF. The file may be corrupted.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It might be a scanned/image-only document."
        )

    return chunks


def get_vectorstore(chunks, collection_name):
    """Build an in-memory Chroma vectorstore for this session only.

    Using an in-memory store (no persist_directory) with a unique
    collection_name per upload avoids mixing chunks from previous
    PDFs/sessions into the search results.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vectorstore


def get_conversational_chain(vectorstore, stream_handler=None):
    """Build the conversational RAG chain.

    A separate, non-streaming LLM is used for the internal
    "condense question" step (rewriting follow-ups using chat
    history), so only the final answer streams to the UI.

    `num_predict` caps the max tokens generated per answer, which
    helps keep responses snappy with a small model like llama3.2:1b.
    """
    qa_llm = ChatOllama(
        model="llama3.2:1b",
        streaming=True,
        callbacks=[stream_handler] if stream_handler else None,
    )
    condense_llm = ChatOllama(model="llama3.2:1b")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=qa_llm,
        condense_question_llm=condense_llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    return chain


def get_response(chain, question):
    """Returns a tuple of (answer_text, source_documents)."""
    response = chain.invoke({"question": question})
    return response["answer"], response.get("source_documents", [])
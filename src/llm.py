import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_llm():
    """Returns a cached ChatOllama instance."""
    try:
        return ChatOllama(model="llama3")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Ollama LLM. Ensure Ollama is running and 'llama3' model is pulled. Error: {str(e)}")

def get_rag_chain(vectorstore):
    """Create the RAG chain using local Ollama and the llama3 model."""
    llm = get_llm()
    
    system_prompt = (
        "You are an expert AI document assistant. "
        "Your goal is to answer questions strictly using the provided context. "
        "\n\n"
        "GUIDELINES:\n"
        "1. Start with a direct answer to the question.\n"
        "2. Follow with a '### Key Insights' section using bullet points if multiple points are relevant.\n"
        "3. If the answer is not in the context, say: 'I'm sorry, but I don't have enough information in the provided documents to answer this.'\n"
        "4. Be professional and concise.\n"
        "\n"
        "CONTEXT:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # LCEL Chain Construction
    # This pipeline retrieves docs, formats them, and passes them to the prompt/LLM
    rag_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # We return the chain and the retriever separately so app.py can retrieve docs for validation
    return rag_chain, retriever

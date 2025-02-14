import os
import sys
import io
import base64
import tiktoken

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import streamlit as st
from config import OPENAI_MODEL_NAME, MODEL_TEMPERATURE, VECTORSTORE_PATH, OPENAI_API_KEY, PROMPT_TEMPLATE, TOKEN_LIMIT, RETRIEVED_DOCS_COUNT
from langsmith import traceable


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME, 
    openai_api_key=OPENAI_API_KEY,
    temperature=MODEL_TEMPERATURE
)

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=PROMPT_TEMPLATE,
)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    """Loads the FAISS vectorstore from disk."""
    try:
        vs = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vs
    except MemoryError as me:
        st.error(
            "MemoryError: The FAISS vectorstore is too large to load into memory."
        )
        raise me

@traceable(name="search_documents")
def retrieve_documents(query: str, k: int = 5):
    """Retrieves documents from the vectorstore based on a query."""
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return []
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

def count_tokens(text: str) -> int:
    """Counts tokens in a text string using OpenAI's tokenizer."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

@traceable(name="build_context")
def build_context(docs) -> Tuple:
    """
    Build a context string by concatenating document summaries and texts.
    Truncates if total tokens exceed TOKEN_LIMIT.
    Returns: (context_str, truncated)
    """
    context_parts = []
    total_tokens = 0

    for i, doc in enumerate(docs, start=1):
        doc_text = f"Article {i}: {doc.page_content}"
        doc_tokens = count_tokens(doc_text)

        if total_tokens + doc_tokens > TOKEN_LIMIT:
            break 

        context_parts.append(doc_text)
        total_tokens += doc_tokens

    return "\n\n".join(context_parts)

@traceable(name="generate_response")
def generate_response(query: str, context: str) -> str:
    """
    Generate a response using the LLM.
    
    Args:
        query: User's question
        context: Context from retrieved documents
        
    Returns:
        Generated response
    """
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(context=context, query=query)
    return response

@traceable(name="process_query")
def process_query(query: str) -> Tuple[str, List[str]]:
    """
    Process a user query through the RAG pipeline.
    
    Args:
        query: User's question
        
    Returns:
        Tuple of (response text, list of sources)
    """

    docs = retrieve_documents(query, k=RETRIEVED_DOCS_COUNT)
    if not docs:
        return "I couldn't find any relevant information to answer your question.", []
    
    context = build_context(docs)
    response = generate_response(query, context)
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    return response, sources

st.set_page_config(page_title="The Batch RAG", layout="wide")
st.title("The Batch RAG")

st.markdown(
    """
    Enter a query below to retrieve documents, answer your question based on the retrieved content
    and display any associated media.
    """
)

with st.form("query_form"):
    query = st.text_input("Enter your query:")
    k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Search")

if submitted and query:
    with st.spinner("Retrieving documents..."):
        docs = retrieve_documents(query, k=k)
    
    if docs:
        st.success(f"Retrieved {len(docs)} document(s).")
        
        context = build_context(docs)
        
        response, sources = process_query(query)
        
        st.markdown("## Answer")
        st.markdown(response)
        st.markdown("---")
        
        st.markdown("## Related resources")
        for i, doc in enumerate(docs, start=1):
            doc_type = doc.metadata.get("type", "text").lower()
            if doc_type == "image":
                st.markdown(f"**Summary:** {doc.page_content}")
                encoded_str = doc.metadata.get("encoded_image", "")
                try:
                    image_bytes = base64.b16decode(encoded_str)
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            else:
                st.markdown(f"**Title:** {doc.metadata.get('title', '')}")
                st.markdown(f"**Source:** {doc.metadata.get('source', '')}")
                st.markdown(f"**Summary:** {doc.metadata.get('summary', '')}")
                # st.markdown(f"**Description:** {doc.metadata.get('description', '')}")
                # st.markdown("**Full text:**")
                # st.text_area(" ", doc.page_content, height=500, key=f"doc_text_{i}")
            st.markdown("---")
    else:
        st.warning("No documents found for your query.")

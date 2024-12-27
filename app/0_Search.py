import langchain_chroma as Chroma
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="amazon_reviews",
    embedding_function=embeddings,
    persist_directory="data/chroma",
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)

results = vector_store.similarity_search_with_score("I like the controller", k=5)
st.write(results)

results = retriever.invoke("I like the controller")
st.write(results)


@st.cache_data
def search(query):
    st.session_state["search_results"] = retriever.invoke(query)


# Search
query = st.text_input("Search", "I love this product")
search_results = st.button("Search", on_click=search(query))

st.write(st.session_state["search_results"])

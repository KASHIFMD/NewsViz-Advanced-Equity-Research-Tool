from dotenv import load_dotenv
import openai
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.title("NewsViz: Advanced Equity Research Tool 📈")

# Input the  securely
on = st.toggle("Do you have your own API key?", disabled=False)
api_key = ""
if on:
    api_key = st.text_input(
        "Enter API Key", value="Enter Valid API key", type="password")
    os.environ['OPENAI_API_KEY'] = api_key
else:
    # take environment variables from .env (especially openai api key)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

openai.api_key = api_key


def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True


n = st.sidebar.text_input(f"Number of articles")


def f(n):
    try:
        n = int(n)
    except ValueError:
        st.sidebar.write("Entered integer is not valid, let n=3")
        return 3
    return n


isValid = is_api_key_valid()
llm = OpenAI(temperature=0.9, max_tokens=500)

st.sidebar.title("News Article URLs")
urls = []
for i in range(f(n)):
    url = st.sidebar.text_input(f"URL {i+1}")
    # st.write(f"url_{i} = {url}")
    urls.append(url)
    # st.write("i=", i)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()


if process_url_clicked and isValid:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    st.write("number of chunks created = ", len(docs))

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query and isValid:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm, retriever=vectorstore.as_retriever())
            # st.write("chain", chain)
            # langchain.debug = True
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            if isValid:
                sources = result.get("sources", "")
            if isValid and sources:
                st.subheader("Sources:")
                # Split the sources by newline
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)


elif isValid == False:
    st.write('Please enter valid api key')
for i in range(0, 10):
    st.text("")
st.title("Python code:")
# Load your Python code
with open("main.py", "r") as f:
    code = f.read()

# Display the Python code in a code block
st.code(code)

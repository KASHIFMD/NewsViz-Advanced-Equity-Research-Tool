import streamlit as st
import openai
from dotenv import load_dotenv
import os
import pickle
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

#  Title
st.title("NewsViz: Advanced Equity Research Tool 📈")

# Input the  securely
on = st.toggle("Do you have your own API key?")
api_key = ""

# check if user has his own api key or not.
if on:
    api_key = st.text_input(
        "Enter API Key", value="Enter Valid API key", type="password")
    os.environ['OPENAI_API_KEY'] = api_key
else:
    # take environment variables from .env (especially openai api key)
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')


# A function to check whether api key is valid or not.
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


isValid = is_api_key_valid()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Decide number of articles to be taken as input.
n = st.sidebar.text_input(f"Number of articles")


def f(n):
    try:
        n = int(n)
    except ValueError:
        st.sidebar.write("Entered integer is not valid, let n=3")
        return 3
    return n


# Take article links as input from the user through streamlit webapp
st.sidebar.title("News Article URLs")
urls = []
for i in range(f(n)):
    url = st.sidebar.text_input(f"URL {i+1}")
    # st.write(f"url_{i} = {url}")
    urls.append(url)
    # st.write("i=", i)
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Processing the webpage content of given URLs.
main_placeholder = st.empty()
if process_url_clicked and isValid:
    # 1 Loading the data
    # This code creates an UnstructuredURLLoader object to load the
    # text data from the given URL. The UnstructuredURLLoader object
    # is then used to load the text data into memory.
    main_placeholder.text("Data Loading...Started...✅✅✅")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # 2 Split the data using some special conditions
    # This code creates a RecursiveCharacterTextSplitter object to
    # split the text data into tokens. The RecursiveCharacterTextSplitter
    # object is a text splitter that can be used to split text into tokens
    # at multiple levels of granularity. For example,
    # it can split text into words, sentences, and even paragraphs.
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    st.write("Number of chunks created = ", len(docs))

    # 3 Create embeddings of all chunks and save it to "FAISS index", here we have used OpenAI embeddings
    # This code creates an OpenAIEmbeddings object to generate
    # dense vector representations of the text tokens.
    # The OpenAIEmbeddings object is trained on a massive dataset of
    # text and code, and it can be used to generate vector
    # representations for sentences, paragraphs, and even entire documents.
    embeddings = OpenAIEmbeddings()
    # This code creates a FAISS vector store to store the vector representations
    # of the text tokens. The FAISS vector store is a
    # fast and efficient way to store and search for dense vectors.
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

    # 4 Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Take input query and process it.
query = st.text_input("Question: ")
if st.button("Submit") and isValid:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # This code creates a RetrievalQAWithSourcesChain object to
            # perform retrieval-based question answering with sources.
            # The RetrievalQAWithSourcesChain object takes a question
            # and a list of documents as input, and it returns the answer
            # to the question along with the sources that the answer is based on.
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
            # query = ''
elif not isValid:
    st.write('Please enter valid api key')


for i in range(0, 10):
    st.write("\n\n\n\n")
st.title("Python code:")
# Load your Python code
with open("main.py", "r") as f:
    code = f.read()

# Display the Python code in a code block
st.code(code)

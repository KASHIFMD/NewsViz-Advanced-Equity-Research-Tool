# NewsViz-Advanced-Equity-Research-Tool
Using LLM, NLP techniques and vector databases, this model help to generate insights from online articles. Insights are generated based on user prompts.
It is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.

![](AERT.jpg)

## Features

- Load URLs or upload text files containing URLs to fetch article content.
- Process article content through LangChain's UnstructuredURL Loader
- Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of relevant information
- Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.


## Installation

1.Clone this repository to your local machine using:

```bash
  git clone https://github.com/KASHIFMD/NewsViz-Advanced-Equity-Research-Tool.git
```
2.Navigate to the project directory:

```bash
  cd folder_path
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Set up your OpenAI API key by creating a .env file in the project root and adding your API

```bash
  OPENAI_API_KEY='key'
```
## Usage/Examples

1. Run the Streamlit app by executing:
```bash
streamlit run main.py
```

2.The web app will open in your browser.

- On the sidebar, put n = number of links.

- you can input URLs directly.

- Initiate the data loading and processing by clicking "Process URLs."

- Observe the system as it performs text splitting, generates embedding vectors, and efficiently indexes them using FAISS.

- The embeddings will be stored and indexed using FAISS, enhancing retrieval speed.

- The FAISS index will be saved in a local file path in pickle format for future use.
- One can now ask a question and get the answer based on those news articles
- URLs for sample use case
  ```bash
  - https://www.moneycontrol.com/news/business/markets/market-corrects-post-rbi-ups-inflation-forecast-icrr-bet-on-these-top-10-rate-sensitive-stocks-ideas-11142611.html
``` 
```bash - https://www.moneycontrol.com/news/business/banks/hdfc-bank-re-appoints-sanmoy-chakrabarti-as-chief-risk-officer-11259771.html
```
```bash
Who is Sanmoy Chakrabarti?
```
## Project Structure

- main.py: The main Streamlit application script.
- requirements.txt: A list of required Python packages for the project.
- faiss_store_openai.pkl: A pickle file to store the FAISS index.
- .env: Configuration file for storing your OpenAI API key.

## Proposed work:
- Use of NLP technque to summarize the give selected chunks based on the query from users.
- 
![](AERT.png)

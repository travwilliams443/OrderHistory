from flask import Flask, render_template, request
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import openai
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import hashlib

app = Flask(__name__)

# Use OpenAI embeddings for vectorization
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize your vector store
vector_store = Chroma(
    collection_name='amazon_products',
    persist_directory='./vectordb',
    embedding_function=openai_embedding
)

def query_orders(text, num_results=1):
    try:
        results = vector_store.similarity_search(
            text,
            k=num_results,
        )
    except Exception as e:
        # Log the exception if necessary
        results = []

    data = []

    for res in results:
        data.append({
            "Product Description": res.page_content,
            "Order Date": res.metadata.get('Order Date'),
            "Product Name": res.metadata.get('Product Name'),
            "Unit Price": res.metadata.get('Unit Price'),
            "Category": res.metadata.get('Category'),
            "ASIN": res.metadata.get('ASIN'),
        })

    df_results = pd.DataFrame(data)
    return df_results


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_text = request.form['query']
        num_results = int(request.form.get('num_results', 3))
        results_df = query_orders(query_text, num_results)
        return render_template('index.html', query=query_text, results=results_df, num_results=num_results)
    else:
        return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)

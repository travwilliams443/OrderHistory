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
from datetime import datetime

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
        # Use similarity_search_with_score to get documents and their scores
        results_with_scores = vector_store.similarity_search_with_score(
            text,
            k=num_results,
        )
    except Exception as e:
        # Log the exception if necessary
        results_with_scores = []

    data = []

    for res, score in results_with_scores:
        # Process Order Date
        order_date_str = res.metadata.get('Order Date')
        if order_date_str:
            try:
                # Convert to datetime object
                order_date = datetime.fromisoformat(order_date_str)
                # Format date for display
                order_date_formatted = order_date.strftime('%Y-%m-%d')
            except Exception:
                order_date = None
                order_date_formatted = order_date_str
        else:
            order_date = None
            order_date_formatted = None

        data.append({
            "Product Description": res.page_content,
            "Order Date": order_date_formatted,
            "Order Date Datetime": order_date,
            "Product Name": res.metadata.get('Product Name'),
            "Unit Price": res.metadata.get('Unit Price'),
            "Category": res.metadata.get('Category'),
            "ASIN": res.metadata.get('ASIN'),
            "Similarity Score": score
        })

    df_results = pd.DataFrame(data)

    return df_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_text = request.form['query']
        num_results = int(request.form.get('num_results', 3))
        sort_by = request.form.get('sort_by', 'similarity')
        try:
            results_df = query_orders(query_text, num_results, sort_by=sort_by)
            return render_template(
                'index.html',
                query=query_text,
                results=results_df,
                num_results=num_results,
                sort_by=sort_by
            )
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template(
                'index.html',
                query=query_text,
                error_message=error_message,
                num_results=num_results,
                sort_by=sort_by
            )
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

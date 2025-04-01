import json
# import openai
import psycopg2
import os 
from dotenv import load_dotenv
load_dotenv()
import requests

# Configure Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY_GPT4')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME = ""

headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}


# Function to get embeddings using Azure OpenAI Service
def get_embedding(text):
    url = f"{AZURE_OPENAI_ENDPOINT}"
    data = {
        "model": "text-embedding-ada-002",
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    # print(response_data)
    
    return response_data['data'][0]['embedding']

def search_similar_query(nl_query, top_k=4):
    conn = get_db_connection()
    cur = conn.cursor()
    # Generate Embedding for the Input Query
    query_embedding = get_embedding(nl_query)

    # Convert to PostgreSQL Vector Format
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    # Connect to PostgreSQL
    # conn = psycopg2.connect(**DB_CONFIG)
    # cur = conn.cursor()

    # Perform Similarity Search Using `pgvector`
    cur.execute(f"""
        SELECT nl_query, sql_query, 1 - (embedding <=> '{embedding_str}') AS similarity
        FROM embeddings
        ORDER BY similarity DESC
        LIMIT {top_k};
    """)

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


def search_tables(nl_query,top_k=15):
    # Generate Embedding for the Input Query
    conn = get_db_connection()
    cur = conn.cursor()
    query_embedding = get_embedding(nl_query)

    # Convert to PostgreSQL Vector Format
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    # Connect to PostgreSQL
    # conn = psycopg2.connect(**DB_CONFIG)
    # cur = conn.cursor()

    # Perform Similarity Search Using `pgvector`
    cur.execute(f"""
        SELECT table_name, columns_description, 1 - (embedding_vector <=> '{embedding_str}') AS similarity
        FROM schema_embeddings
        ORDER BY similarity DESC
        LIMIT {top_k};
    """)

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results



# Connect to PostgreSQL
from RAG.db_conf import get_db_connection
conn = get_db_connection()
cur = conn.cursor()

if __name__=='__main__':

        # Extract and Process Data
    with open("table_schemas.json", "r") as f:
        datas = json.load(f)
    # print(datas[0])
    for data in datas:
        # print(data['table_name'],data['schema'],"\n\n\n")
        table_name=data['table_name']
        columns_text=data['schema']
        
        
        # Generate Embedding
        
        embedding_vector = get_embedding(columns_text)

        # Convert to PostgreSQL format
        embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"

        # Store in PostgreSQL
        
        cur.execute(
            "INSERT INTO schema_embeddings (table_name, columns_description, embedding_vector) VALUES (%s, %s, %s)",
            (table_name, columns_text, embedding_str)
        )
        conn.commit()
    # cur.close()
    # conn.close()

    # Create Table (Choose JSONB or pgvector)
   


    # Load JSON Data
    with open("train_generate_task.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Insert Data into PostgreSQL
    for item in data:
        nl_query = item["NL"]
        sql_query = item["Query"]
        embedding = get_embedding(nl_query)

        # Insert into PostgreSQL
        cur.execute(
            "INSERT INTO embeddings (nl_query, embedding, sql_query) VALUES (%s, %s, %s)",
            (nl_query, json.dumps(embedding), sql_query)
        )

    conn.commit()
    cur.close()
    conn.close()

    print("Embeddings stored successfully!")

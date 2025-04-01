import psycopg2

# Connect to PostgreSQL
from db_conf import get_db_connection
conn = get_db_connection()
cur = conn.cursor()

# Enable pgvector extension (run once)
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create table to store embeddings, queries, and results
cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        nl_query TEXT NOT NULL,
        embedding VECTOR(1536),
        sql_query TEXT NOT NULL,
        query_result JSONB
    );
""")
cur.execute("""CREATE TABLE IF NOT EXISTS schema_embeddings (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(255) NOT NULL,
    columns_description TEXT NOT NULL,
    embedding_vector VECTOR(1536) -- Adjust based on embedding dimensions
);""")


conn.commit()
cur.close()
conn.close()

print("Database table created successfully!")

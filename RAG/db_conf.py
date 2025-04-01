import psycopg2
from dotenv import load_dotenv
load_dotenv()
import os
PG_PASSWORD=os.getenv('PG_PASSWORD',"")
# Database Connection Function
def get_db_connection():
    return psycopg2.connect(
        dbname="postgres",
        user="raghu",
        password=PG_PASSWORD,
        host="postgredatabas.postgres.database.azure.com",
        port=5432
    )

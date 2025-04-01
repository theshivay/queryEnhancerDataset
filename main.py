PATH_TO_SQL_QUERY_TESTING_FILE = "/Users/shivammishra/Desktop/test/queryEnhancerDataset/other/test.json"
# PATH_TO_SQL_QUERY_TESTING_FILE = "/Users/shivammishra/Desktop/Adobe/train_generate_task.json"
import os
import json
import time
import threading
import single_pipeline
import RAG.embedding_creator as search_api

def start_single_pipeline(nl_query):
    """
    Dummy function to process the NL query.
    Should return the corresponding SQL query as a string.
    """
    print(f"Processing query: {nl_query}")

    start_time = time.time()

    # Getting Relevant Tables [RAG]
    relevant_tables = search_api.search_tables(nl_query)
    # Getting Relevant Tables [LLM]
    relevant_tables, table_agent_token_count = single_pipeline.table_agent(nl_query, relevant_tables)
    # Getting Relevant Columns [LLM]
    relevant_schema, column_agent_token_count = single_pipeline.prune_agent(nl_query, relevant_tables)
    # Getting Relevant Sample Queries [RAG]
    relevant_sample_queries = search_api.search_similar_query(nl_query)
    # Getting SQL Query [LLM]
    sql_query, sql_agent_token_count = single_pipeline.final_sql_query_generator(nl_query, relevant_schema, relevant_sample_queries)

    print("-----------")
    print("Table Creation Token Count:", table_agent_token_count)
    print("Column Pruning Token Count:", column_agent_token_count)
    print("SQL Generation Token Count:", sql_agent_token_count)
    print("-----------")

    file_path = "result.json"

    # Read existing data
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Append new entry
    data.append({"NL": nl_query, "Query": sql_query})

    # Write back to file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    end_time = time.time()
    print("-----------Performance Metrics-----------")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Total Tokens: {table_agent_token_count + column_agent_token_count + sql_agent_token_count}")
    print(f"SQL Query: {sql_query}")
    print("----------------------------")
    exit()
    return sql_query  # Replace this with actual query generation logic

def normalize_string(s):
    """Convert to lowercase and strip extra spaces."""
    return s.lower().strip()

def check_query_match(expected_query, generated_query):
    """Compare normalized queries."""
    return normalize_string(expected_query) == normalize_string(generated_query)

def process_single_json_object(json_obj):
    """Process a single JSON object and return whether it matches."""
    nl_query = json_obj.get("NL")
    expected_query = json_obj.get("Query")
    generated_query = start_single_pipeline(nl_query)
    print("-"*30)
    print(f"Expected: {expected_query}")
    print(f"Generated: {generated_query}")
    print("-"*30)
    return check_query_match(expected_query, generated_query)

def process_json_objects(file_path):
    """Read JSON file and return the list of objects."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)  

def create_filtered_tables_dataset(input_file_path, output_file_path="filtered_tables_dataset.json"):
    """
    Create a dataset with user queries and filtered tables.
    
    Args:
        input_file_path: Path to the training data file
        output_file_path: Path to save the resulting JSON file
    """
    # Read the training data
    training_data = process_json_objects(input_file_path)
    results = []
    
    print(f"Processing {len(training_data)} queries for filtered tables dataset...")
    
    for idx, item in enumerate(training_data):
        query = item.get("NL")
        if not query:
            continue
        
        print(f"[{idx+1}/{len(training_data)}] Processing: {query[:50]}...")
        
        # Get filtered tables from RAG
        rag_filtered_tables = search_api.search_tables(query)
        
        # Get filtered tables from LLM
        llm_filtered_tables, _ = single_pipeline.table_agent(query, rag_filtered_tables)
        
        # Add to results
        results.append({
            "user_raw_query": query,
            "rag_filtered_tables": rag_filtered_tables,
            "llm_filtered_tables": llm_filtered_tables
        })
        
        # Save intermediate results every 10 items
        if idx % 10 == 0 and idx > 0:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4)
    
    # Save final results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Dataset created with {len(results)} entries and saved to {output_file_path}")
    return results

def process_queries_multithreaded(file_path, max_threads=16):
    """Process queries using multithreading."""
    json_objects = process_json_objects(file_path)
    
    threads = []
    successful_matches = 0
    unsuccessful_matches = 0
    lock = threading.Lock()
    
    def worker(json_obj):
        nonlocal successful_matches, unsuccessful_matches
        if process_single_json_object(json_obj):
            with lock:
                successful_matches += 1
        else:
            with lock:
                unsuccessful_matches += 1
    
    for json_obj in json_objects:
        thread = threading.Thread(target=worker, args=(json_obj,))
        threads.append(thread)
        thread.start()
        
        if len(threads) >= max_threads:
            for t in threads:
                t.join()
            threads = []
    
    for t in threads:
        t.join()
    
    print("\n=== Multithreaded Processing ===")
    print(f"Successful Matches: {successful_matches}")
    print(f"Unsuccessful Matches: {unsuccessful_matches}")

def process_queries_linear(file_path):
    """Process queries one by one in a normal linear way."""
    json_objects = process_json_objects(file_path)
    
    successful_matches = 0
    unsuccessful_matches = 0
    
    for json_obj in json_objects:
        if process_single_json_object(json_obj):
            successful_matches += 1
        else:
            unsuccessful_matches += 1
        
        # Print live updates
        print(f"Processed: {successful_matches + unsuccessful_matches}, "
              f"Successful: {successful_matches}, Unsuccessful: {unsuccessful_matches}")
    
    print("\n=== Linear Processing ===")
    print(f"Successful Matches: {successful_matches}")
    print(f"Unsuccessful Matches: {unsuccessful_matches}")

if __name__ == "__main__":
    sql_file_path = 'other/sample_submission_generate_task.json'  # Ensure proper escaping
    
    # Create filtered tables dataset
    # Uncomment the below line to create the dataset
    create_filtered_tables_dataset(PATH_TO_SQL_QUERY_TESTING_FILE, "filtered_tables_dataset.json")
    
    # print("\nRunning Multithreaded Processing...")
    # process_queries_multithreaded(sql_file_path)

    # print("\nRunning Linear Processing...")
    # process_queries_linear(sql_file_path)
import time
from app import build_index_from_files, build_term_document_matrix, search_with_matrix, evaluate_query, parse_query, read_documents_from_directory, inverted_index

# Define your document directory path
directory_path = '../wiki_text'

# Load documents
start_time = time.time()
documents_content_store = read_documents_from_directory(directory_path)
documents_loading_time = time.time() - start_time
print(f"Documents loaded in {documents_loading_time} seconds.")

# Initialize inverted index
start_time = time.time()
build_index_from_files(directory_path)
inverted_index_init_time = time.time() - start_time
print(f"Inverted Index initialized in {inverted_index_init_time} seconds.")

# Initialize term-document matrix
start_time = time.time()
build_term_document_matrix(documents_content_store)
term_document_matrix_init_time = time.time() - start_time
print(f"Term-Document Matrix initialized in {term_document_matrix_init_time} seconds.")

# Define a set of test queries
queries = ["war and not dog and not cat or crumbles", "dog or cat and not man", "war and not cat and fumble"]

# Function to measure query processing times
def measure_query_processing_times(queries):
    # Inverted Index Query Processing
    start_time = time.time()
    for query in queries:
        parsed_query = parse_query(query)
        evaluate_query(parsed_query, inverted_index)
    inverted_index_query_time = time.time() - start_time
    print(f"Inverted Index: Processed {len(queries)} queries in {inverted_index_query_time} seconds.")

    # Term-Document Matrix Query Processing
    start_time = time.time()
    for query in queries:
        search_with_matrix(query)
    term_document_matrix_query_time = time.time() - start_time
    print(f"Term-Document Matrix: Processed {len(queries)} queries in {term_document_matrix_query_time} seconds.")

# Call the function to perform the measurements
measure_query_processing_times(queries)
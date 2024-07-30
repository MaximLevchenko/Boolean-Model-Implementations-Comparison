import os
import numpy as np
import time
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from flask_cors import CORS

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)


# Global variables to store the term-document matrix, document names, and term index
term_document_matrix = None
doc_names = []
term_index = {}  # Global dictionary for term to index mapping


def build_term_document_matrix(documents):
    """
       Builds the term-document matrix for the provided documents.
       Each document's terms are processed (stemmed, stopwords removed),
       and a matrix of term frequencies is created.
       """
    global term_document_matrix, doc_names, term_index
    # Reset or initialize doc_names and term_index for each build to prevent stale data
    doc_names = sorted(documents.keys())
    # Extract all unique terms from the documents after processing (stemming, removing stopwords)
    all_terms = set(term for doc in documents.values() for term in process_text(doc))
    # Create a global term index mapping terms to their indices in the matrix
    term_index = {term: i for i, term in enumerate(sorted(all_terms))}

    # Initialize the term-document matrix with zeros
    matrix = np.zeros((len(term_index), len(doc_names)), dtype=int)

    # Populate the matrix with term frequencies
    for doc_id, content in documents.items():
        for term in process_text(content):
            if term in term_index:  # Check if the term exists in our global term index
                matrix[term_index[term], doc_names.index(doc_id)] += 1

    term_document_matrix = matrix


def search_with_matrix(query):
    """
        Searches documents using the term-document matrix.
        The query is expected to be in postfix notation (RPN).
        Logical operations AND, OR, NOT are supported.
        """
    global term_index, term_document_matrix, doc_names

    # Convert query to postfix notation if needed (this example assumes it's already in postfix)
    parsed_query = parse_query(query)  # Assume this returns query in postfix notation as list of terms and operators

    stack = []

    for token in parsed_query:
        if token.upper() in ('AND', 'OR', 'NOT'):
            if token.upper() == 'AND':
                set2 = stack.pop()
                set1 = stack.pop()
                result_set = set1.intersection(set2)
            elif token.upper() == 'OR':
                set2 = stack.pop()
                set1 = stack.pop()
                result_set = set1.union(set2)
            elif token.upper() == 'NOT':
                set1 = stack.pop()
                all_docs_set = set(range(len(doc_names)))
                result_set = all_docs_set - set1
            stack.append(result_set)
        else:
            # Process the term
            if token.lower() in term_index:
                term_idx = term_index[token.lower()]
                # Get document indices where term appears
                doc_indices = np.where(term_document_matrix[term_idx, :] > 0)[0]
                stack.append(set(doc_indices))
            else:
                stack.append(set())  # Term not found in any document

    # Assuming there's only one item left in stack which is the final result set of document indices
    if stack:
        final_doc_indices = stack.pop()
        matching_doc_ids = [doc_names[idx] for idx in final_doc_indices]
        results = fetch_documents(matching_doc_ids)
        return results
    else:
        return []


def is_valid_query(query):
    """
        Validates the search query to ensure correct syntax.
        Checks for matching parentheses and valid placement of logical operators.
        """
    # Check for matching parentheses
    parentheses_stack = []
    for char in query:
        if char == '(':
            parentheses_stack.append(char)
        elif char == ')':
            if not parentheses_stack or parentheses_stack[-1] != '(':
                return False
            parentheses_stack.pop()

    # Check if stack is empty, meaning all parentheses were matched
    if parentheses_stack:
        return False

    # Check for presence of input after logical operators
    if re.search(r'\b(AND|OR|NOT|and|or|not)\b\s*$', query.strip()):
        return False

    return True


def read_documents_from_directory(directory_path):
    """
    Reads all wiki_text files in the specified directory,
    returning a dictionary mapping each file name (without extension) to its content.
    """
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Assuming wiki_text files
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                print("Reading")
                # Use filename without extension as document ID
                doc_id = os.path.splitext(filename)[0]
                documents[doc_id] = file.read()
    return documents


def process_text(text):
    """
        Processes the input text by tokenizing, removing stopwords, and stemming words.
        Returns a list of processed terms.
        """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

    return stemmed_tokens


inverted_index = {}


def parse_query(query):
    """
        Parses the input query and converts it into postfix notation using the shunting yard algorithm.
        Returns a list of tokens in postfix notation.
        """
    tokens = query.replace('(', ' ( ').replace(')', ' ) ').split()
    postfix_tokens = shunting_yard(tokens)
    return postfix_tokens


def shunting_yard(tokens):
    """
        Converts infix notation tokens to postfix notation using the shunting yard algorithm.
        Handles the precedence and associativity of logical operators AND, OR, NOT.
        """
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output_queue = []
    operator_stack = []

    for token in tokens:
        if token.upper() not in ('AND', 'OR', 'NOT', '(', ')'):
            output_queue.append(token)
        elif token.upper() in precedence:
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence.get(operator_stack[-1], 0) >= precedence[token.upper()]):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token.upper())
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            top_token = operator_stack.pop()
            while top_token != '(':
                output_queue.append(top_token)
                top_token = operator_stack.pop()

    while operator_stack:
        output_queue.append(operator_stack.pop())

    return output_queue

# Global variable to store the content of each document, mapping doc_id to content
documents_content_store = {}


def fetch_documents(doc_ids):
    """
        Fetches documents' content based on the provided document IDs.
        Returns a list of dictionaries containing document details including URL.
        """
    documents = []
    for doc_id in doc_ids:
        if doc_id in documents_content_store:
            file_url = f"http://localhost:5000/files/{doc_id}.txt"  # Construct file URL
            documents.append({
                "id": doc_id,
                "title": f"Document {doc_id}",
                "content": documents_content_store[doc_id],
                "url": file_url  # Include the URL in the response
            })
    return documents


def evaluate_query(parsed_query, inverted_index):
    """
        Evaluates the parsed query using the inverted index.
        Supports logical operations AND, OR, NOT.
        """
    stack = []

    for token in parsed_query:
        if token in ('AND', 'OR', 'NOT'):
            if token == 'AND':
                if len(stack) < 2:  # Check if there are at least two operands on the stack
                    raise ValueError("Invalid query: AND operation requires two operands.")
                set2 = stack.pop()
                set1 = stack.pop()
                result_set = set1.intersection(set2)
                stack.append(result_set)
            elif token == 'OR':
                if len(stack) < 2:  # Check if there are at least two operands on the stack
                    raise ValueError("Invalid query: OR operation requires two operands.")
                set2 = stack.pop()
                set1 = stack.pop()
                result_set = set1.union(set2)
                stack.append(result_set)
            elif token == 'NOT':
                if not stack:  # Check if there is at least one operand on the stack
                    raise ValueError("Invalid query: NOT operation requires an operand.")
                set1 = stack.pop()
                all_docs = set(documents_content_store.keys())  # Adjust based on actual doc IDs
                result_set = all_docs - set1
                stack.append(result_set)
        else:
            token_set = set(inverted_index.get(token.lower(), []))
            stack.append(token_set)

    if len(stack) != 1:  # After processing all tokens, only one result set should remain
        raise ValueError("Invalid query: unbalanced operands and operations.")

    return stack.pop()  # Return the remaining set, which is the result of the query


def build_index_from_files(directory_path):
    """
    Reads documents from files in the given directory,
    processes each document, and builds the inverted index.
    Additionally, populates documents_content_store with document contents.
    """
    global documents_content_store  # Ensure we're modifying the global variable
    documents_content_store.clear()  # Clear previous contents, if any

    documents = read_documents_from_directory(directory_path)
    documents_content_store.update(documents)  # Store document contents

    for doc_id, content in documents.items():
        processed_text = process_text(content)
        for term in processed_text:
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)

@app.route('/files/<path:filename>')
def serve_file(filename):
    """
        Serves files from the specified directory based on the filename.
        """
    return send_from_directory('../wiki_text', filename)


@app.route('/', methods=['POST'])
def search_documents():
    """
        Handles search requests.
        Validates the query and chooses the search method (inverted index or term-document matrix).
        Returns the search results in JSON format.
        """
    query = request.json.get('query', '')
    search_method = request.json.get('method', 'inverted_index')  # New field to choose the search method

    if not is_valid_query(query):
        return jsonify({"error": "Invalid query. Please check your syntax."}), 400

    try:
        if search_method == 'term_document_matrix':
            # Make sure the term-document matrix is built
            if term_document_matrix is None:
                build_term_document_matrix(documents_content_store)
            results = search_with_matrix(query)
        else:
            parsed_query = parse_query(query)
            result_doc_ids = evaluate_query(parsed_query, inverted_index)
            results = fetch_documents(result_doc_ids)

        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    directory_path = '../wiki_text'  # Update this path
    start_time = time.time()
    documents_content_store = read_documents_from_directory(directory_path)
    build_term_document_matrix(documents_content_store)  # Also build the term-document matrix
    term_document_matrix_time = time.time() - start_time
    start_time = time.time()
    build_index_from_files(directory_path)  # Build the index from files
    inverted_index_time = time.time() - start_time
    print(f"Time to build inverted index: {inverted_index_time} seconds.")
    print(f"Time to build term-document matrix: {term_document_matrix_time} seconds.")
    app.run(debug=False)
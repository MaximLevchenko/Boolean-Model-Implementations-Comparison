# Inverted Index vs Term-Document Matrix

This project consists of a web-based search application coupled with a Python backend. It provides functionalities for document processing, indexing, and searching through documents via a web interface. The backend is built using Flask, and the frontend is created using React. The purpose of this project is also to **compare the efficiency** and performance of two different methods for handling search operations: the **inverted index** and the **term-document matrix**.

For a detailed analysis and performance comparison, please refer to the accompanying [report](report.pdf) titled "Performance Comparison of Inverted Index and Term-Document Matrix in a Boolean Model Search System,". The report outlines the methodologies used, the results obtained, and discusses the implications of using these two different data structures in search systems.
## Project Structure

### Backend

- **app.py**: Main Python server file that includes Flask routes for document searching and file serving.
- **test_performance.py**: Contains performance testing scripts for the document indexing and search functionalities.

### Frontend

- **App.js**, **App.css**: Core React component and its styling.
- **components/Search.js**, **components/Search.css**: React component and styling for the search functionality.

### Wiki Articles Folder
- wiki_* 

Folder containing sample Wikipedia articles used for testing and demonstration purposes.
## Features

- **Document Indexing**: Process documents to create an inverted index and term-document matrix.
- **Search Interface**: User interface for querying the indexed documents.
- **Performance Testing**: Scripts to measure the performance of document loading, indexing, and querying.

## Performance Overview

- **Initialization**: The time required to build both the inverted index and the term-document matrix from scratch.
- **Query Processing**: Demonstrated average times for processing various Boolean queries. The inverted index was generally faster, especially for complex queries involving multiple Boolean operators due to less computationally intensive set operations.

## Setup

### Requirements

- Node.js and npm
- Python 3.x
- Flask and NLTK for Python
- React for frontend

### Installation

#### Python Dependencies

Install the necessary Python packages using:

```bash
pip install flask numpy nltk
```
### React Setup

Navigate to the `search-app` directory and install dependencies:

```bash
cd frontend
npm install
```

## Running the Application

### Start the Backend Server
Navigate to the `src` directory and start a backend server:

```bash
cd backend
python3 app.py
```


### Run the React Application
```bash
cd frontend
npm start
```


## Running Performance Tests

To measure the performance of document loading, indexing, and querying, you can run the `test_performance.py` script. This script benchmarks the initialization times for the inverted index and term-document matrix, as well as the time taken to process different types of queries.

### Running the Test Script

Execute the performance test script using the following command:

```bash
cd backend
python test_performance.py
```

### Modifying Queries

The test_performance.py script contains a predefined list of queries to test the search functionalities. You can modify these queries to check the performance for different search scenarios. For example, you can edit the queries list in the script:
```python
queries = [
    "war and not dog and not cat or crumbles",
    "dog or cat and not man",
    "war and not cat and fumble"
]
```
Simply replace these strings with other Boolean expressions to test various combinations of keywords and operators.

## Performance Overview

- **Initialization**:
  - Inverted Index initialized in 8.156273365020752 seconds.
  - Term-Document Matrix initialized in 17.222548007965088 seconds.

- **Query Processing**:
  - Inverted Index: Processed 3 queries in 0.0001926422119140625 seconds.
  - Term-Document Matrix: Processed 3 queries in 0.00051116943359375 seconds.

These results provide insights into the efficiency of different data structures for search operations, particularly highlighting the faster performance of the inverted index for complex queries.

### Usage
Once both servers are running, access the web interface at http://localhost:3000 and perform searches using the input field. Results will be displayed based on the documents indexed by the Python backend.


### Conclusion

The project demonstrates the use of both inverted index and term-document matrix to facilitate efficient search operations. Based on our performance analysis, the inverted index provides superior performance in most practical applications, particularly where search frequency is high and data dynamism is low.
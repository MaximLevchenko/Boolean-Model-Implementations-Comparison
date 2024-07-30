import React, { useState } from 'react';

import axios from 'axios';
import './Search.css';
function Search() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');
  const [searchMethod, setSearchMethod] = useState('inverted_index'); // New state for search method

  const isValidQuery = (query) => {
    // Check for matching parentheses
    const parenthesesMatch = query.split('').reduce((acc, char) => {
      if (acc < 0) { // early return if more closing parentheses
        return acc;
      } else if (char === '(') {
        return acc + 1;
      } else if (char === ')') {
        return acc - 1;
      }
      return acc;
    }, 0) === 0;

    // Check for presence of input after logical operators
    const hasInputAfterOperators = !/\b(AND|OR|NOT)\b\s*$/.test(query.trim());

    return parenthesesMatch && hasInputAfterOperators;
  };

  const handleSearch = async () => {
    if (!isValidQuery(query)) {
      setError('Please input a valid query with matching parentheses and input after logical operators (AND, OR, NOT).');
      setResults([]);
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/', {
        query,
        method: searchMethod // Send the selected search method to the backend
      });
      setError(''); // Clear error if the query is valid
      setResults(response.data);
    } catch (error) {
      // Check if the error response has the expected format and use the provided message
      if (error.response && error.response.status === 400 && error.response.data.error) {
        setError(error.response.data.error);
      } else {
        console.error("Failed to fetch search results:", error);
        setError('An error occurred while fetching the search results.');
      }
      setResults([]);
    }
  };

  return (
    <div className="search-wrapper">
      <div className="search-options">
        <label htmlFor="search-method">Search Method:</label>
        <select
          id="search-method"
          value={searchMethod}
          onChange={(e) => setSearchMethod(e.target.value)}
        >
          <option value="inverted_index">Inverted Index</option>
          <option value="term_document_matrix">Term-Document Matrix</option>
        </select>
      </div>
      <input
        className="search-input"
        type="text"
        value={query}
        placeholder="Enter search terms..."
        onChange={(e) => setQuery(e.target.value)}
      />
      <button className="search-button" onClick={handleSearch}>Search</button>
      {error && <div className="error-message">{error}</div>}
      <ul className="results-container">
        {results.map((result, index) => (
          <li key={index} className="result-item">
            <span className="result-title">{result.title}</span>
            <a href={result.url} target="_blank" rel="noopener noreferrer" className="sample-button">
              Get {result.fileType} files
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Search;

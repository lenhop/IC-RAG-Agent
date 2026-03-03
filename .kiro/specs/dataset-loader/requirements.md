# Requirements Document

## Introduction

This document specifies the requirements for a centralized Dataset Loader module for RAG evaluation. The module will load and validate test cases from amazon_fqa.csv, provide auto-retrieval of relevant contexts as a fallback, and eliminate duplicate code across existing evaluation test files.

## Glossary

- **Dataset_Loader**: The centralized module responsible for loading, validating, and enriching test cases from CSV files
- **Test_Case**: A dictionary containing question, ground truth answer, and optional metadata for RAG evaluation
- **RAG_Pipeline**: The Retrieval-Augmented Generation pipeline used for retrieving relevant document contexts
- **CSV_File**: The amazon_fqa.csv file containing FAQ questions and answers
- **Contexts**: A list of relevant document chunks retrieved for a given question
- **Ground_Truth**: The expected correct answer for a test question (mapped from 'answer' column in CSV)
- **Vector_Store**: The Chroma vector database used for semantic search and context retrieval
- **Embedder**: The component that converts text into vector embeddings for semantic search

## Requirements

### Requirement 1: Load Test Cases from CSV

**User Story:** As a RAG evaluation developer, I want to load test cases from a CSV file with configurable path and limit, so that I can run evaluation tests on a consistent dataset.

#### Acceptance Criteria

1. WHEN a CSV path is provided, THE Dataset_Loader SHALL load test cases from that specific file path
2. WHEN no CSV path is provided, THE Dataset_Loader SHALL load from the RAG_FAQ_CSV environment variable
3. WHEN the RAG_FAQ_CSV environment variable is not set, THE Dataset_Loader SHALL load from the default path "data/intent_classification/fqa/amazon_fqa.csv"
4. WHEN a limit parameter is provided, THE Dataset_Loader SHALL return at most that number of test cases
5. WHEN no limit is provided, THE Dataset_Loader SHALL return all test cases from the CSV file
6. WHEN the CSV file is not found, THE Dataset_Loader SHALL raise a FileNotFoundError with a clear error message including the attempted path
7. WHEN the CSV file is empty, THE Dataset_Loader SHALL print a warning message and return an empty list
8. THE Dataset_Loader SHALL parse CSV files with UTF-8 encoding
9. THE Dataset_Loader SHALL map the 'question' column to the 'question' field in test cases
10. THE Dataset_Loader SHALL map the 'answer' column to the 'ground_truth' field in test cases
11. THE Dataset_Loader SHALL map the 'category' column to the 'category' field in test cases
12. THE Dataset_Loader SHALL map the 'source' column to the 'source' field in test cases
13. THE Dataset_Loader SHALL generate sequential IDs in the format "faq_001", "faq_002", etc. for each test case
14. THE Dataset_Loader SHALL initialize the 'contexts' field as an empty list for each test case

### Requirement 2: Validate Test Case Structure

**User Story:** As a RAG evaluation developer, I want to validate that test cases have required fields, so that I can catch data quality issues early before running expensive evaluations.

#### Acceptance Criteria

1. WHEN validating an empty test dataset, THE Dataset_Loader SHALL print an error message and return False
2. WHEN a test case is missing the 'question' field, THE Dataset_Loader SHALL print an error message with the case ID and return False
3. WHEN a test case is missing the 'ground_truth' field, THE Dataset_Loader SHALL print an error message with the case ID and return False
4. WHEN a test case is missing the 'contexts' field, THE Dataset_Loader SHALL print a warning message indicating that retrieval metrics will use ground_truth fallback and return True
5. WHEN a test case is missing optional fields (category, source, id), THE Dataset_Loader SHALL print a warning message and return True
6. WHEN all test cases have required fields, THE Dataset_Loader SHALL return True
7. THE Dataset_Loader SHALL use the 'id' field from test cases for error messages when available
8. WHEN the 'id' field is not available, THE Dataset_Loader SHALL use "case_{index}" format for error messages

### Requirement 3: Auto-Retrieve Relevant Contexts

**User Story:** As a RAG evaluation developer, I want to automatically retrieve relevant contexts for test cases that don't have them, so that I can run retrieval metrics even when manual context annotation is incomplete.

#### Acceptance Criteria

1. WHEN a test case already has a 'contexts' field with content, THE Dataset_Loader SHALL not modify that test case
2. WHEN a test case is missing the 'contexts' field, THE Dataset_Loader SHALL retrieve top-k relevant document chunks using the RAG_Pipeline
3. WHEN a test case has an empty 'question' field, THE Dataset_Loader SHALL skip context retrieval for that test case
4. THE Dataset_Loader SHALL embed the question using the RAG_Pipeline embedder
5. THE Dataset_Loader SHALL query the vector store using the question embedding to retrieve top-k chunks
6. THE Dataset_Loader SHALL store retrieved chunk content in the 'contexts' field
7. THE Dataset_Loader SHALL add a 'contexts_source' field with value "auto_retrieved" for auto-retrieved contexts
8. THE Dataset_Loader SHALL print an info message showing the count of test cases with auto-retrieved contexts
9. THE Dataset_Loader SHALL use the k parameter to determine how many contexts to retrieve per question
10. WHEN k is not specified, THE Dataset_Loader SHALL default to retrieving 5 contexts per question

### Requirement 4: Remove Duplicate Code

**User Story:** As a RAG evaluation developer, I want to eliminate duplicate dataset loading code from test files, so that the codebase is maintainable and changes only need to be made in one place.

#### Acceptance Criteria

1. THE test_retrieval_metrics.py file SHALL import load_fqa_dataset from the Dataset_Loader module
2. THE test_retrieval_metrics.py file SHALL not contain a local load_test_dataset function
3. THE test_generation_llm_judge.py file SHALL import load_fqa_dataset from the Dataset_Loader module
4. THE test_generation_llm_judge.py file SHALL not contain a local load_test_dataset function
5. THE test_retrieval_metrics.py file SHALL use load_fqa_dataset instead of the removed local function
6. THE test_generation_llm_judge.py file SHALL use load_fqa_dataset instead of the removed local function

### Requirement 5: Configure Environment Variable

**User Story:** As a RAG evaluation developer, I want a centralized environment variable for the dataset path, so that I can easily change the dataset location without modifying code.

#### Acceptance Criteria

1. THE .env file SHALL contain a RAG_FAQ_CSV variable
2. THE RAG_FAQ_CSV variable SHALL be set to "data/intent_classification/fqa/amazon_fqa.csv" by default
3. THE Dataset_Loader SHALL read the RAG_FAQ_CSV environment variable when no explicit path is provided
4. WHEN the RAG_FAQ_CSV variable is not set, THE Dataset_Loader SHALL use the default path

### Requirement 6: Update Module Exports

**User Story:** As a RAG evaluation developer, I want the Dataset_Loader functions to be exported from the evaluation module, so that I can import them conveniently.

#### Acceptance Criteria

1. THE src/rag/evaluation/__init__.py file SHALL export the load_fqa_dataset function
2. THE src/rag/evaluation/__init__.py file SHALL export the validate_dataset function
3. THE src/rag/evaluation/__init__.py file SHALL export the add_relevant_contexts function
4. THE src/rag/evaluation/__init__.py file SHALL include all three functions in the __all__ list

### Requirement 7: Unit Test Coverage

**User Story:** As a RAG evaluation developer, I want comprehensive unit tests for the Dataset_Loader module, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE test suite SHALL include a test that loads 10 test cases from amazon_fqa.csv and verifies structure
2. THE test suite SHALL include a test that verifies the limit parameter works correctly
3. THE test suite SHALL include a test that validates a dataset with all required fields returns True
4. THE test suite SHALL include a test that validates a dataset missing the 'question' field returns False
5. THE test suite SHALL include a test that validates a dataset missing 'contexts' prints a warning but returns True
6. THE test suite SHALL include a test that verifies add_relevant_contexts retrieves contexts and adds metadata
7. THE test suite SHALL verify that loaded test cases have the correct ID format (faq_001, faq_002, etc.)
8. THE test suite SHALL verify that loaded test cases contain 'question', 'ground_truth', 'category', and 'source' fields

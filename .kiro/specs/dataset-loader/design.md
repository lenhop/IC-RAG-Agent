# Design Document: Dataset Loader Module

## Overview

The Dataset Loader module provides a centralized, reusable interface for loading and validating RAG evaluation test cases from CSV files. It eliminates duplicate code across evaluation test files and provides automatic context retrieval as a fallback when manual annotations are unavailable.

The module consists of three main functions:
- `load_fqa_dataset()`: Loads and parses CSV files into structured test case dictionaries
- `validate_dataset()`: Validates test case structure and warns about missing fields
- `add_relevant_contexts()`: Auto-retrieves contexts using the RAG pipeline as a fallback

This design follows the principle of separation of concerns: dataset loading is independent of evaluation metrics, making both more maintainable and testable.

## Architecture

### Module Structure

```
src/rag/evaluation/
├── dataset_loader.py       # Core module (this design)
├── retrieval_metrics.py    # Uses dataset_loader
├── generation_metrics.py   # Uses dataset_loader
└── __init__.py            # Exports dataset_loader functions
```

### Data Flow

```
CSV File (amazon_fqa.csv)
    ↓
load_fqa_dataset()
    ↓
List[Dict] (test cases)
    ↓
validate_dataset() ← validation check
    ↓
add_relevant_contexts() ← optional enrichment
    ↓
Evaluation Metrics (retrieval/generation)
```

### Configuration Hierarchy

The module uses a three-tier configuration hierarchy for CSV path resolution:

1. Explicit `csv_path` parameter (highest priority)
2. `RAG_FAQ_CSV` environment variable
3. Default path: `data/intent_classification/fqa/amazon_fqa.csv`

This allows flexibility for different environments (development, testing, production) without code changes.

## Components and Interfaces

### Component 1: CSV Loader

**Function:** `load_fqa_dataset(csv_path, limit, project_root)`

**Purpose:** Load and parse CSV files into structured test case dictionaries

**Parameters:**
- `csv_path: Optional[str]` - Path to CSV file (None = use env var or default)
- `limit: Optional[int]` - Maximum number of test cases to load (None = load all)
- `project_root: Optional[Path]` - Root directory for resolving relative paths

**Returns:** `List[Dict[str, Any]]` - List of test case dictionaries

**Algorithm:**
```
1. Resolve CSV path:
   a. If csv_path provided, use it
   b. Else check RAG_FAQ_CSV environment variable
   c. Else use default path
   d. If relative path, resolve against project_root

2. Validate file exists:
   - If not found, raise FileNotFoundError with path

3. Open CSV with UTF-8 encoding

4. Parse using csv.DictReader:
   For each row (up to limit):
     a. Generate sequential ID: faq_001, faq_002, ...
     b. Map 'question' column → 'question' field
     c. Map 'answer' column → 'ground_truth' field
     d. Map 'category' column → 'category' field
     e. Map 'source' column → 'source' field
     f. Parse 'contexts' column if present (comma/pipe-separated)
     g. Add test case to list

5. Return test cases list
```

**Error Handling:**
- `FileNotFoundError`: CSV file not found at resolved path
- `csv.Error`: Malformed CSV file
- Empty CSV: Return empty list (no error)

### Component 2: Dataset Validator

**Function:** `validate_dataset(test_cases, warn_missing)`

**Purpose:** Validate test case structure and warn about data quality issues

**Parameters:**
- `test_cases: List[Dict[str, Any]]` - Test cases to validate
- `warn_missing: bool` - Whether to emit warnings for missing optional fields (default: True)

**Returns:** `bool` - True if dataset is valid, False if critical fields missing

**Validation Rules:**

1. **Critical validation (returns False if fails):**
   - Dataset must not be empty
   - Each test case must have non-empty 'question' field

2. **Warning-level validation (returns True but warns):**
   - Missing 'ground_truth' field (limits generation evaluation)
   - Missing 'contexts' field (forces fallback to ground_truth for retrieval metrics)
   - Missing optional fields ('category', 'source', 'id')

**Algorithm:**
```
1. Check if test_cases is empty:
   - If empty, warn and return False

2. For each test case:
   a. Check 'question' field:
      - If missing or empty, warn with case ID and return False
   
   b. Track missing 'ground_truth' count
   c. Track missing 'contexts' count

3. After all cases checked:
   a. If any missing ground_truth, warn with count
   b. If all missing contexts, warn about fallback behavior

4. Return True (all critical validations passed)
```

### Component 3: Context Auto-Retriever

**Function:** `add_relevant_contexts(test_cases, pipeline, k, overwrite)`

**Purpose:** Auto-retrieve relevant contexts for test cases missing manual annotations

**Parameters:**
- `test_cases: List[Dict[str, Any]]` - Test cases to enrich (modified in place)
- `pipeline: RAGPipeline` - Pipeline with embedder and vector_store
- `k: int` - Number of contexts to retrieve per question (default: 5)
- `overwrite: bool` - Whether to replace existing contexts (default: False)

**Returns:** `List[Dict[str, Any]]` - Same test_cases list (modified in place)

**Algorithm:**
```
1. Get Chroma collection from pipeline.vector_store

2. For each test case:
   a. Skip if 'contexts' exists and overwrite=False
   b. Skip if 'question' is empty
   
   c. Embed question using pipeline.embedder.embed_query()
   
   d. Query collection:
      - Use question embedding
      - Request k results
      - Include only documents (not metadata/distances)
   
   e. Extract document content from results
   
   f. Store in test case:
      - Set 'contexts' field to list of document strings
      - Handle empty results gracefully

3. Return modified test_cases
```

**Integration with RAG Pipeline:**

The function uses the public Chroma API from `ai_toolkit.chroma`:
- `get_chroma_collection(vector_store)` - Get collection handle
- `query_collection(collection, query_embeddings, n_results, include)` - Query for similar documents

This ensures compatibility with the existing RAG pipeline without tight coupling.

## Data Models

### Test Case Dictionary

```python
{
    "id": str,              # Format: "faq_001", "faq_002", ...
    "question": str,        # The question to answer
    "ground_truth": str,    # Expected correct answer (from 'answer' column)
    "category": str,        # Question category (e.g., "FBA Features")
    "source": str,          # Source of the question (e.g., "Amazon SellerCentral")
    "contexts": List[str],  # Relevant document chunks (optional, can be auto-retrieved)
}
```

**Field Descriptions:**

- `id`: Unique identifier for the test case, generated sequentially
- `question`: The input question for RAG evaluation
- `ground_truth`: The expected answer, used for generation metrics (faithfulness, relevance)
- `category`: Categorical label for analysis and filtering
- `source`: Origin of the question, for traceability
- `contexts`: List of relevant document chunks, used for retrieval metrics (Recall, Precision, MRR)

**Field Requirements:**

- Required: `question`, `ground_truth`
- Optional: `contexts`, `category`, `source`, `id`
- Auto-generated: `id` (if not in CSV)
- Auto-retrievable: `contexts` (via `add_relevant_contexts()`)

### CSV File Format

**Expected Columns:**
- `question` (required) - The question text
- `answer` (required) - The ground truth answer
- `category` (optional) - Question category
- `source` (optional) - Question source
- `contexts` (optional) - Comma or pipe-separated list of relevant chunks

**Example CSV:**
```csv
question,answer,category,source
"How many FBA services are there?","Fulfillment by Amazon offers 3 main services...","FBA Features","Amazon SellerCentral"
"What is the return policy?","Items can be returned within 30 days...","Returns","Amazon Help"
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Path Resolution Hierarchy

*For any* combination of csv_path parameter, RAG_FAQ_CSV environment variable, and default path, the loader should resolve the path according to the priority hierarchy: explicit parameter > environment variable > default path.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Limit Enforcement

*For any* positive integer limit N and any CSV file, loading with limit=N should return at most N test cases, and the returned test cases should be the first N rows from the CSV.

**Validates: Requirements 1.4**

### Property 3: File Not Found Error

*For any* non-existent file path, attempting to load should raise FileNotFoundError with an error message containing the attempted path.

**Validates: Requirements 1.6**

### Property 4: UTF-8 Encoding Support

*For any* CSV file containing valid UTF-8 characters (including non-ASCII characters like Chinese, Arabic, emoji), the loader should correctly parse and preserve those characters in the returned test cases.

**Validates: Requirements 1.8**

### Property 5: Output Structure Consistency

*For any* valid CSV file, each returned test case should have the correct structure: 'id' in format "faq_NNN", 'question' mapped from 'question' column, 'ground_truth' mapped from 'answer' column, 'category' and 'source' fields present, and sequential ID numbering starting from 1.

**Validates: Requirements 1.9, 1.10, 1.11, 1.12, 1.13, 1.14**

### Property 6: Required Field Validation

*For any* test case missing either 'question' or 'ground_truth' fields, validation should return False and emit an error message containing the case ID.

**Validates: Requirements 2.2, 2.3**

### Property 7: Optional Field Warnings

*For any* test case missing 'contexts' field, validation should return True but emit a warning message about fallback behavior.

**Validates: Requirements 2.4, 2.5**

### Property 8: Valid Dataset Acceptance

*For any* dataset where all test cases have 'question' and 'ground_truth' fields, validation should return True.

**Validates: Requirements 2.6**

### Property 9: Context Preservation

*For any* test case that already has a non-empty 'contexts' field, calling add_relevant_contexts with overwrite=False should leave that test case's contexts unchanged.

**Validates: Requirements 3.1**

### Property 10: Context Retrieval Completeness

*For any* test case with a non-empty 'question' field and missing 'contexts' field, calling add_relevant_contexts should populate the 'contexts' field with a list of at most k document strings retrieved from the vector store.

**Validates: Requirements 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

## Error Handling

### File System Errors

**FileNotFoundError:**
- Raised when CSV file doesn't exist at resolved path
- Error message includes the full resolved path
- Allows caller to distinguish between missing file and other errors

**Permission Errors:**
- Handled by Python's built-in file operations
- Propagated to caller with original exception

### CSV Parsing Errors

**Malformed CSV:**
- `csv.Error` raised by csv.DictReader
- Propagated to caller with row number if available

**Empty CSV:**
- Returns empty list (not an error)
- Validation will catch and warn about empty dataset

### Data Quality Issues

**Missing Required Fields:**
- Detected by `validate_dataset()`
- Returns False with warning messages
- Caller decides whether to proceed or abort

**Missing Optional Fields:**
- Detected by `validate_dataset()`
- Returns True with warning messages
- Evaluation can proceed with limitations

### Retrieval Errors

**Vector Store Unavailable:**
- `add_relevant_contexts()` catches exceptions
- Sets 'contexts' to empty list for affected test cases
- Continues processing remaining test cases

**Embedding Failures:**
- Caught per test case
- Sets 'contexts' to empty list for that case
- Continues with next test case

## Testing Strategy

### Dual Testing Approach

The module will be tested using both unit tests and property-based tests:

**Unit Tests:**
- Specific examples demonstrating correct behavior
- Edge cases (empty CSV, missing columns, special characters)
- Integration with RAG pipeline
- Error conditions (file not found, malformed CSV)

**Property-Based Tests:**
- Universal properties across all inputs
- Comprehensive input coverage through randomization
- Each property test runs minimum 100 iterations
- Tests reference design document properties

### Unit Test Coverage

1. **Load dataset with limit** - Verify loading 10 test cases with correct structure
2. **Limit parameter** - Verify limit=3 returns exactly 3 cases
3. **Validation - valid dataset** - Verify dataset with all required fields returns True
4. **Validation - missing question** - Verify missing question returns False
5. **Validation - missing contexts warning** - Verify missing contexts warns but returns True
6. **Add relevant contexts** - Verify auto-retrieval adds contexts and metadata

### Property-Based Test Configuration

- Library: `hypothesis` (Python)
- Minimum iterations: 100 per property
- Tag format: `# Feature: dataset-loader, Property N: [property text]`
- Each correctness property implemented as a single property-based test

### Test Data Requirements

- Sample CSV files with various structures
- Test cases with UTF-8 characters (Chinese, Arabic, emoji)
- Empty CSV files
- CSV files with missing columns
- Mock RAG pipeline for context retrieval tests

### Integration Testing

The module integrates with:
- `RAGPipeline` for context retrieval
- `ai_toolkit.chroma` for vector store queries
- Evaluation metrics modules (retrieval_metrics, generation_metrics)

Integration tests verify:
- Correct usage of Chroma public API
- Compatibility with RAGPipeline interface
- Proper data flow to evaluation metrics

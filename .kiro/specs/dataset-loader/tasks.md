# Implementation Plan: Dataset Loader Module

## Overview

This implementation plan covers the creation of a centralized dataset loader module for RAG evaluation. The module is already implemented in `src/rag/evaluation/dataset_loader.py`, so this plan focuses on:

1. Creating comprehensive unit tests (currently missing)
2. Verifying the existing implementation matches the requirements
3. Ensuring proper integration with evaluation test files

The implementation follows a test-driven approach: write tests first to verify the existing implementation, then make any necessary adjustments.

## Tasks

- [ ] 1. Create unit test file and test infrastructure
  - Create `tests/evaluation/test_dataset_loader.py`
  - Set up test fixtures and sample CSV files
  - Import necessary modules (pytest, dataset_loader functions)
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8_

- [ ]* 1.1 Write unit test for loading dataset with structure verification
  - Test loading 10 test cases from amazon_fqa.csv
  - Verify correct ID format (faq_001, faq_002, etc.)
  - Verify presence of required fields (question, ground_truth, category, source)
  - _Requirements: 7.1, 7.7, 7.8_

- [ ]* 1.2 Write unit test for limit parameter
  - Test that limit=3 returns exactly 3 test cases
  - Verify that returned cases are the first 3 from the CSV
  - _Requirements: 7.2_

- [ ]* 1.3 Write unit test for valid dataset validation
  - Create test cases with all required fields
  - Verify validate_dataset returns True
  - _Requirements: 7.3_

- [ ]* 1.4 Write unit test for missing question validation
  - Create test case without 'question' field
  - Verify validate_dataset returns False
  - Verify error message is emitted
  - _Requirements: 7.4_

- [ ]* 1.5 Write unit test for missing contexts warning
  - Create test case without 'contexts' field but with required fields
  - Verify validate_dataset returns True
  - Verify warning message is emitted
  - _Requirements: 7.5_

- [ ]* 1.6 Write unit test for add_relevant_contexts
  - Create test case without contexts
  - Call add_relevant_contexts with mock RAG pipeline
  - Verify contexts are added
  - Verify contexts_source metadata is added
  - _Requirements: 7.6_

- [ ]* 1.7 Write property test for path resolution hierarchy
  - **Property 1: Path Resolution Hierarchy**
  - **Validates: Requirements 1.1, 1.2, 1.3**
  - Generate random combinations of csv_path, env var, and default
  - Verify correct priority: explicit > env var > default

- [ ]* 1.8 Write property test for limit enforcement
  - **Property 2: Limit Enforcement**
  - **Validates: Requirements 1.4**
  - Generate random limit values
  - Verify returned list length <= limit
  - Verify returned cases are first N rows

- [ ]* 1.9 Write property test for file not found error
  - **Property 3: File Not Found Error**
  - **Validates: Requirements 1.6**
  - Generate random non-existent paths
  - Verify FileNotFoundError is raised
  - Verify error message contains the path

- [ ]* 1.10 Write property test for UTF-8 encoding support
  - **Property 4: UTF-8 Encoding Support**
  - **Validates: Requirements 1.8**
  - Create CSV files with various UTF-8 characters (Chinese, Arabic, emoji)
  - Verify characters are preserved in loaded test cases

- [ ]* 1.11 Write property test for output structure consistency
  - **Property 5: Output Structure Consistency**
  - **Validates: Requirements 1.9, 1.10, 1.11, 1.12, 1.13, 1.14**
  - Generate random valid CSV files
  - Verify each test case has correct structure and field mappings
  - Verify sequential ID numbering

- [ ]* 1.12 Write property test for required field validation
  - **Property 6: Required Field Validation**
  - **Validates: Requirements 2.2, 2.3**
  - Generate test cases missing question or ground_truth
  - Verify validation returns False with error message

- [ ]* 1.13 Write property test for optional field warnings
  - **Property 7: Optional Field Warnings**
  - **Validates: Requirements 2.4, 2.5**
  - Generate test cases missing contexts
  - Verify validation returns True but emits warning

- [ ]* 1.14 Write property test for valid dataset acceptance
  - **Property 8: Valid Dataset Acceptance**
  - **Validates: Requirements 2.6**
  - Generate datasets with all required fields
  - Verify validation returns True

- [ ]* 1.15 Write property test for context preservation
  - **Property 9: Context Preservation**
  - **Validates: Requirements 3.1**
  - Generate test cases with existing contexts
  - Call add_relevant_contexts with overwrite=False
  - Verify contexts remain unchanged

- [ ]* 1.16 Write property test for context retrieval completeness
  - **Property 10: Context Retrieval Completeness**
  - **Validates: Requirements 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**
  - Generate test cases without contexts
  - Call add_relevant_contexts
  - Verify contexts are populated with at most k documents

- [ ] 2. Verify integration with existing test files
  - Verify test_retrieval_metrics.py imports from dataset_loader
  - Verify test_generation_llm_judge.py imports from dataset_loader
  - Verify no duplicate load_test_dataset functions exist
  - Run integration tests to ensure compatibility
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 3. Verify environment variable configuration
  - Check that .env contains RAG_FAQ_CSV variable
  - Verify default value matches specification
  - Test that dataset_loader reads the environment variable correctly
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ] 4. Verify module exports
  - Check src/rag/evaluation/__init__.py exports load_fqa_dataset
  - Check src/rag/evaluation/__init__.py exports validate_dataset
  - Check src/rag/evaluation/__init__.py exports add_relevant_contexts
  - Verify all three functions are in __all__ list
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 5. Checkpoint - Run all tests and verify implementation
  - Run pytest on test_dataset_loader.py
  - Verify all unit tests pass
  - Verify all property tests pass (100+ iterations each)
  - Run integration tests (test_retrieval_metrics.py, test_generation_llm_judge.py)
  - Ensure all tests pass, ask the user if questions arise

- [ ] 6. Documentation and final verification
  - Verify docstrings are complete and accurate
  - Verify error messages are clear and helpful
  - Verify code follows project style guidelines
  - Update any relevant documentation files

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for faster MVP
- The core implementation already exists in `src/rag/evaluation/dataset_loader.py`
- Focus is on comprehensive testing to verify correctness
- Property tests should run minimum 100 iterations each
- Integration tests verify compatibility with existing evaluation modules
- All tests reference specific requirements for traceability

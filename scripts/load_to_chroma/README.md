# Chroma load scripts

Use **`src.utils`** for **`bootstrap_project`** / **`resolve_path`** (environment
and path setup). Use **`src.chroma`** for Chroma ingest and verification. These
scripts are thin CLIs over those APIs.

| Script | Purpose |
|--------|---------|
| `load_documents_to_chroma.py` | PDFs only → documents collection (`load_pdf_directory_to_chroma`) |
| `load_intent_registry_to_chroma.py` | Intent registry CSV (`text`, `intent`; optional `workflow`, else workflow=intent) |
| `load_csv_to_chroma.py` | Generic single-column CSV → Chroma |
| `verify_chroma_load.py` | Check documents + intent_registry counts |

Imports: `from src.utils import bootstrap_project, resolve_path` and
`from src.chroma import ...` (do not depend on `src.rag` from new scripts).

Backward compatibility: `src.rag.chroma_loaders` and `src.rag.vector_registry_loader`
re-export the same functions where applicable.

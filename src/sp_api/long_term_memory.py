"""Long-term memory module using ChromaDB for persistent semantic memory."""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root for resolving paths
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_chroma_path(project_root: Optional[Path] = None) -> str:
    """Get ChromaDB path from env or default."""
    root = project_root or _PROJECT_ROOT
    path = os.getenv("SP_API_CHROMA_PATH", str(root / "data" / "vector_store" / "sp_api_memory"))
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    return str(p.resolve())


def _get_top_k() -> int:
    """Get top-K memories to retrieve from env or default."""
    return int(os.getenv("SP_API_MEMORY_TOP_K", "3"))


class LongTermMemory:
    """ChromaDB-backed long-term semantic memory for SP-API agent.

    Stores session summaries and user preferences as embeddings,
    enabling semantic retrieval of relevant past interactions.
    """

    def __init__(
        self,
        chroma_client=None,
        collection_name: str = "sp_api_long_term_memory",
        embedding_fn=None,
        project_root: Optional[Path] = None,
    ):
        """Initialize long-term memory with ChromaDB.

        Args:
            chroma_client: Optional ChromaDB client instance. If None, creates a new one.
            collection_name: Name of the ChromaDB collection.
            embedding_fn: Optional embedding function. If None, creates default.
            project_root: Optional project root for path resolution.
        """
        self._project_root = project_root or _PROJECT_ROOT
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn
        self._client = chroma_client
        self._collection = None
        self._top_k = _get_top_k()

        self._initialize()

    def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            if self._client is None:
                chroma_path = _get_chroma_path(self._project_root)
                Path(chroma_path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=Settings(anonymized_telemetry=False),
                )

            # Get or create collection
            try:
                self._collection = self._client.get_collection(name=self._collection_name)
            except Exception:
                self._collection = self._client.create_collection(
                    name=self._collection_name,
                    metadata={"description": "SP-API Agent long-term memory"}
                )

            # Initialize embedding function if not provided
            if self._embedding_fn is None:
                self._embedding_fn = self._create_embedding_function()

        except ImportError:
            # ChromaDB not available - graceful degradation
            self._client = None
            self._collection = None
        except Exception:
            # Other errors - graceful degradation
            self._client = None
            self._collection = None

    def _create_embedding_function(self) -> Optional[Any]:
        """Create embedding function using project's RAG embeddings."""
        try:
            from src.rag import create_embeddings

            # Use minilm for long-term memory (fast, efficient)
            return create_embeddings(model_type="minilm", project_root=self._project_root)
        except Exception:
            return None

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text using the embedding function."""
        if self._embedding_fn is None:
            return None
        try:
            return self._embedding_fn.embed_query(text)
        except Exception:
            return None

    def store(self, user_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory (session summary, user preference, insight).

        Args:
            user_id: User identifier.
            content: Memory content to store.
            metadata: Optional metadata dictionary.

        Returns:
            Memory ID (empty string if storage failed).
        """
        if self._collection is None:
            return ""

        try:
            memory_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat() + "Z"

            metadatas = {
                "user_id": user_id,
                "timestamp": timestamp,
            }
            if metadata:
                metadatas.update(metadata)

            # Generate embedding
            embedding = self._embed_text(content)
            if embedding is None:
                # Store without embedding - ChromaDB will use default
                self._collection.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadatas],
                )
            else:
                self._collection.add(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadatas],
                    embeddings=[embedding],
                )

            return memory_id
        except Exception:
            return ""

    def recall(
        self, user_id: str, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K memories semantically relevant to the query.

        Args:
            user_id: User identifier to filter memories.
            query: Query text to search for relevant memories.
            top_k: Number of memories to retrieve (default from config).

        Returns:
            List of {content, metadata, score} dictionaries.
        """
        if self._collection is None:
            return []

        try:
            k = top_k or self._top_k

            # Generate query embedding
            query_embedding = self._embed_text(query)
            if query_embedding is None:
                return []

            # Query with user_id filter
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )

            memories = []
            if results and results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results.get("metadatas", [{}])[0][i] if results.get("metadatas") else {}
                    distance = results.get("distances", [[]])[0][i] if results.get("distances") else 0.0
                    # Convert distance to similarity score (lower distance = higher similarity)
                    score = 1.0 / (1.0 + distance) if distance > 0 else 1.0

                    memories.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": score,
                    })

            return memories
        except Exception:
            return []

    def store_session_summary(
        self, user_id: str, session_id: str, turns: List[Dict[str, Any]]
    ) -> str:
        """Summarize a session's conversation turns and store as a memory.

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            turns: List of conversation turns (each with query, response, etc.).

        Returns:
            Memory ID (empty string if storage failed).
        """
        if not turns:
            return ""

        try:
            # Generate summary using simple approach (LLM summary can be added later)
            summary_parts = []
            for turn in turns[-5:]:  # Summarize last 5 turns
                query = turn.get("query", "")
                response = turn.get("response", "")
                if query and response:
                    summary_parts.append(f"Q: {query[:100]}... A: {response[:100]}...")

            summary = f"Session {session_id} summary:\n" + "\n".join(summary_parts)

            # Extract topics from queries (simple keyword extraction)
            topics = self._extract_topics(turns)

            metadata = {
                "session_id": session_id,
                "type": "session_summary",
                "turn_count": len(turns),
                "topics": ",".join(topics),
            }

            return self.store(user_id, summary, metadata)
        except Exception:
            return ""

    def _extract_topics(self, turns: List[Dict[str, Any]]) -> List[str]:
        """Extract simple topics from conversation turns.

        Args:
            turns: List of conversation turns.

        Returns:
            List of topic keywords.
        """
        topics = set()
        keywords = {
            "inventory", "order", "shipment", "catalog", "product",
            "financial", "report", "fee", "fba", "eligibility",
            "price", "stock", "refund", "return",
        }

        for turn in turns:
            query = turn.get("query", "").lower()
            for keyword in keywords:
                if keyword in query:
                    topics.add(keyword)

        return list(topics)[:5]  # Max 5 topics

    def get_user_context(self, user_id: str, query: str, max_memories: Optional[int] = None) -> str:
        """Convenience method: recall + format into a context string for prompt injection.

        Args:
            user_id: User identifier.
            query: Query text to find relevant memories.
            max_memories: Maximum memories to include (default from config).

        Returns:
            Formatted context string for prompt injection.
        """
        memories = self.recall(user_id, query, top_k=max_memories or self._top_k)

        if not memories:
            return ""

        context_parts = []
        for i, mem in enumerate(memories, 1):
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})
            session_id = metadata.get("session_id", "unknown")
            timestamp = metadata.get("timestamp", "")

            context_parts.append(f"[Memory {i}] (Session: {session_id}, Time: {timestamp})")
            context_parts.append(content)

        return "Relevant past interactions:\n" + "\n".join(context_parts)

    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user.

        Args:
            user_id: User identifier.

        Returns:
            Number of memories deleted.
        """
        if self._collection is None:
            return 0

        try:
            # Get all memories for user
            results = self._collection.get(
                where={"user_id": user_id},
                include=["documents"],
            )

            if not results or not results.get("ids"):
                return 0

            ids_to_delete = results["ids"]
            self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def get_memory_count(self, user_id: Optional[str] = None) -> int:
        """Get count of stored memories.

        Args:
            user_id: Optional user ID to filter. If None, returns total count.

        Returns:
            Number of memories stored.
        """
        if self._collection is None:
            return 0

        try:
            if user_id:
                results = self._collection.get(where={"user_id": user_id})
                return len(results.get("ids", [])) if results else 0
            else:
                return self._collection.count()
        except Exception:
            return 0

    def is_available(self) -> bool:
        """Check if long-term memory is available."""
        return self._collection is not None

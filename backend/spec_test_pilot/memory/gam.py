"""
GAM-style memory system for SpecTestPilot.

Implements:
- PageStore: Append-only storage for pages (id, title, tags, content, timestamp)
- Memorizer: Produces memos from runs and stores artifacts as pages
- Researcher: Deep-research loop (plan → search → integrate → reflect)

Uses:
- rank_bm25 for keyword search
- sentence-transformers + FAISS for vector search
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Literal
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from rank_bm25 import BM25Okapi
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False


# Default conventions for test generation
DEFAULT_CONVENTIONS = [
    {
        "title": "REST API Testing Conventions",
        "tags": ["convention", "rest", "testing"],
        "content": (
            "1. Every endpoint needs at least one happy path test with valid inputs.\n"
            "2. Include negative tests for validation errors (400), auth failures (401/403), "
            "and not found (404).\n"
            "3. Test idempotency for PUT/DELETE operations."
        )
    },
    {
        "title": "Authentication Testing Patterns",
        "tags": ["convention", "auth", "security"],
        "content": (
            "1. Test with valid credentials returns expected response.\n"
            "2. Test with missing auth header returns 401.\n"
            "3. Test with invalid/expired token returns 401 or 403."
        )
    },
    {
        "title": "Request Validation Patterns",
        "tags": ["convention", "validation", "negative"],
        "content": (
            "1. Test missing required fields returns 400 with field name in error.\n"
            "2. Test invalid field types (string vs int) returns 400.\n"
            "3. Test boundary values for numeric fields."
        )
    },
    {
        "title": "Response Schema Validation",
        "tags": ["validator", "schema", "contract"],
        "content": (
            "1. Validate response matches documented schema.\n"
            "2. Check required fields are present.\n"
            "3. Verify data types match specification."
        )
    },
    {
        "title": "Pagination Testing",
        "tags": ["convention", "pagination", "list"],
        "content": (
            "1. Test default pagination returns limited results.\n"
            "2. Test page/offset parameters work correctly.\n"
            "3. Test invalid pagination params return 400."
        )
    }
]


@dataclass
class Session:
    """Represents a GAM session with clear boundaries."""
    session_id: str
    tenant_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_transcript_entry(self, role: str, content: str, timestamp: Optional[float] = None):
        """Add entry to session transcript."""
        self.transcript.append({
            "role": role,
            "content": content,
            "timestamp": timestamp or time.time()
        })
    
    def add_tool_output(self, tool_name: str, output: Any, timestamp: Optional[float] = None):
        """Add tool output to session."""
        self.tool_outputs.append({
            "tool": tool_name,
            "output": output,
            "timestamp": timestamp or time.time()
        })
    
    def add_artifact(self, name: str, content: str, artifact_type: str, timestamp: Optional[float] = None):
        """Add code/log artifact to session."""
        self.artifacts.append({
            "name": name,
            "content": content,
            "type": artifact_type,
            "timestamp": timestamp or time.time()
        })
    
    def end_session(self):
        """Mark session as ended."""
        self.end_time = time.time()
    
    def get_full_content(self) -> str:
        """Get lossless session content for page storage."""
        content_parts = []
        
        # Session metadata
        content_parts.append(f"Session ID: {self.session_id}")
        if self.tenant_id:
            content_parts.append(f"Tenant ID: {self.tenant_id}")
        content_parts.append(f"Duration: {(self.end_time or time.time()) - self.start_time:.2f}s")
        content_parts.append("")
        
        # Transcript
        if self.transcript:
            content_parts.append("=== TRANSCRIPT ===")
            for entry in self.transcript:
                timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
                content_parts.append(f"[{timestamp}] {entry['role']}: {entry['content']}")
            content_parts.append("")
        
        # Tool outputs
        if self.tool_outputs:
            content_parts.append("=== TOOL OUTPUTS ===")
            for output in self.tool_outputs:
                timestamp = datetime.fromtimestamp(output["timestamp"]).strftime("%H:%M:%S")
                content_parts.append(f"[{timestamp}] {output['tool']}: {str(output['output'])[:500]}...")
            content_parts.append("")
        
        # Artifacts  
        if self.artifacts:
            content_parts.append("=== ARTIFACTS ===")
            for artifact in self.artifacts:
                timestamp = datetime.fromtimestamp(artifact["timestamp"]).strftime("%H:%M:%S")
                content_parts.append(f"[{timestamp}] {artifact['name']} ({artifact['type']}):")
                content_parts.append(artifact['content'])
                content_parts.append("")
        
        return "\n".join(content_parts)


@dataclass
class Page:
    """A page in the memory store."""
    id: str
    title: str
    tags: List[str]
    content: str
    timestamp: float = field(default_factory=time.time)
    source: Literal["convention", "existing_tests", "runbook", "validator", "memo"] = "memo"
    tenant_id: Optional[str] = None  # Add tenant isolation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "tags": self.tags,
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "tenant_id": self.tenant_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Page":
        """Create from dictionary."""
        return cls(**data)


class PageStore:
    """
    Append-only storage for memory pages.
    
    Supports both BM25 keyword search and vector similarity search.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_vector_search: bool = True
    ):
        """
        Initialize PageStore.
        
        Args:
            embedding_model: Sentence transformer model name
            use_vector_search: Whether to use vector search (requires sentence-transformers)
        """
        self.pages: List[Page] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # BM25 index
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_docs: List[List[str]] = []
        
        # Vector search
        self.use_vector_search = use_vector_search and VECTOR_SEARCH_AVAILABLE
        self._embedder: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[Any] = None
        self._embedding_dim: int = 384  # Default for MiniLM
        
        if self.use_vector_search:
            try:
                self._embedder = SentenceTransformer(embedding_model)
                self._embedding_dim = self._embedder.get_sentence_embedding_dimension()
                self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)  # Inner product
            except Exception:
                self.use_vector_search = False
        
        # Load default conventions
        self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default convention pages."""
        for conv in DEFAULT_CONVENTIONS:
            self.add_page(
                title=conv["title"],
                tags=conv["tags"],
                content=conv["content"],
                source="convention"
            )
    
    def _generate_id(self, title: str, content: str) -> str:
        """Generate unique page ID."""
        hash_input = f"{title}:{content}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all pages."""
        self._tokenized_docs = [
            self._tokenize(f"{p.title} {' '.join(p.tags)} {p.content}")
            for p in self.pages
        ]
        if self._tokenized_docs:
            self._bm25 = BM25Okapi(self._tokenized_docs)
    
    def add_page(
        self,
        title: str,
        tags: List[str],
        content: str,
        source: Literal["convention", "existing_tests", "runbook", "validator", "memo"] = "memo",
        tenant_id: Optional[str] = None
    ) -> Page:
        """
        Add a new page to the store.
        
        Args:
            title: Page title
            tags: List of tags
            content: Page content
            source: Source type
            tenant_id: Tenant ID for isolation
            
        Returns:
            Created Page
        """
        page_id = self._generate_id(title, content)
        page = Page(
            id=page_id,
            title=title,
            tags=tags,
            content=content,
            source=source,
            tenant_id=tenant_id
        )
        
        idx = len(self.pages)
        self.pages.append(page)
        self._id_to_idx[page_id] = idx
        
        # Update BM25
        self._rebuild_bm25()
        
        # Update vector index
        if self.use_vector_search and self._embedder is not None:
            text = f"{title} {' '.join(tags)} {content}"
            embedding = self._embedder.encode([text], normalize_embeddings=True)
            self._faiss_index.add(embedding.astype(np.float32))
        
        return page
    
    def get_page(self, page_id: str) -> Optional[Page]:
        """Get page by ID."""
        idx = self._id_to_idx.get(page_id)
        if idx is not None:
            return self.pages[idx]
        return None
    
    def search_bm25(self, query: str, top_k: int = 5, tenant_id: Optional[str] = None) -> List[Tuple[Page, float]]:
        """
        Search pages using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
            tenant_id: Filter by tenant ID for isolation
            
        Returns:
            List of (Page, score) tuples
        """
        if not self._bm25 or not self.pages:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                page = self.pages[idx]
                # Apply tenant filtering
                if tenant_id is None or page.tenant_id is None or page.tenant_id == tenant_id:
                    results.append((page, float(scores[idx])))
        
        return results
    
    def search_vector(self, query: str, top_k: int = 5, tenant_id: Optional[str] = None) -> List[Tuple[Page, float]]:
        """
        Search pages using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            tenant_id: Filter by tenant ID for isolation
            
        Returns:
            List of (Page, score) tuples
        """
        if not self.use_vector_search or not self._embedder or not self.pages:
            return []
        
        query_embedding = self._embedder.encode([query], normalize_embeddings=True)
        scores, indices = self._faiss_index.search(
            query_embedding.astype(np.float32), 
            min(top_k, len(self.pages))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:
                page = self.pages[idx]
                # Apply tenant filtering
                if tenant_id is None or page.tenant_id is None or page.tenant_id == tenant_id:
                    results.append((page, float(score)))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        tenant_id: Optional[str] = None
    ) -> List[Tuple[Page, float]]:
        """
        Hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results
            bm25_weight: Weight for BM25 scores (1 - bm25_weight for vector)
            tenant_id: Filter by tenant ID for isolation
            
        Returns:
            List of (Page, score) tuples
        """
        bm25_results = self.search_bm25(query, top_k * 2, tenant_id=tenant_id)
        vector_results = self.search_vector(query, top_k * 2, tenant_id=tenant_id)
        
        # Normalize and combine scores
        page_scores: Dict[str, float] = {}
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results)
            for page, score in bm25_results:
                normalized = score / max_bm25 if max_bm25 > 0 else 0
                page_scores[page.id] = bm25_weight * normalized
        
        # Normalize and add vector scores
        if vector_results:
            max_vec = max(s for _, s in vector_results)
            for page, score in vector_results:
                normalized = score / max_vec if max_vec > 0 else 0
                page_scores[page.id] = page_scores.get(page.id, 0) + (1 - bm25_weight) * normalized
        
        # Sort by combined score
        sorted_ids = sorted(page_scores.keys(), key=lambda x: page_scores[x], reverse=True)
        
        results = []
        for page_id in sorted_ids[:top_k]:
            page = self.get_page(page_id)
            if page:
                results.append((page, page_scores[page_id]))
        
        return results
    
    def search_by_tags(
        self, tags: List[str], top_k: int = 5, tenant_id: Optional[str] = None
    ) -> List[Page]:
        """Search pages by tags with optional tenant filtering."""
        tag_set = set(tags)
        scored = []
        for page in self.pages:
            if tenant_id is not None and page.tenant_id not in (None, tenant_id):
                continue
            overlap = len(tag_set & set(page.tags))
            if overlap > 0:
                scored.append((page, overlap))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]

    def search_by_page_ids(
        self, page_ids: List[str], tenant_id: Optional[str] = None
    ) -> List[Page]:
        """Retrieve pages by explicit page IDs with optional tenant filtering."""
        seen = set()
        pages: List[Page] = []
        for page_id in page_ids:
            if page_id in seen:
                continue
            seen.add(page_id)
            page = self.get_page(page_id)
            if not page:
                continue
            if tenant_id is not None and page.tenant_id not in (None, tenant_id):
                continue
            pages.append(page)
        return pages


class Memorizer:
    """
    Produces memos from agent runs and stores artifacts as pages.
    Supports session-based lossless storage with contextual headers.
    """
    
    def __init__(self, page_store: PageStore):
        """
        Initialize Memorizer.
        
        Args:
            page_store: PageStore instance
        """
        self.page_store = page_store
        self._active_sessions: Dict[str, Session] = {}
    
    def start_session(self, tenant_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new GAM session with clear boundaries.
        
        Args:
            tenant_id: Tenant ID for isolation
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )
        self._active_sessions[session_id] = session
        return session_id
    
    def add_to_session(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add content to an active session.
        
        Args:
            session_id: Session ID
            role: Role (user, assistant, system, tool)
            content: Content to add
            tool_outputs: Tool outputs to record
            artifacts: Code/log artifacts to record
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._active_sessions[session_id]
        session.add_transcript_entry(role, content)
        
        if tool_outputs:
            for tool_output in tool_outputs:
                session.add_tool_output(tool_output["tool"], tool_output["output"])
        
        if artifacts:
            for artifact in artifacts:
                session.add_artifact(artifact["name"], artifact["content"], artifact["type"])
    
    def end_session_with_memo(
        self,
        session_id: str,
        spec_title: str,
        endpoints_count: int,
        tests_generated: int,
        key_decisions: List[str],
        issues_found: List[str]
    ) -> Tuple[List[Page], Page]:
        """
        End session and create memo with lossless page storage.
        
        Args:
            session_id: Session ID to end
            spec_title: Spec title for memo
            endpoints_count: Number of endpoints processed
            tests_generated: Number of tests generated  
            key_decisions: Key decisions made
            issues_found: Issues discovered
            
        Returns:
            (lossless_pages, memo_page) tuple
        """
        if session_id not in self._active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self._active_sessions[session_id]
        session.end_session()
        
        # Create lossless page(s) from session content
        lossless_pages = self._create_session_pages(session, spec_title)
        
        # Create contextual memo with page_id pointers
        memo_page = self._create_contextual_memo(
            session, spec_title, endpoints_count, tests_generated, 
            key_decisions, issues_found, lossless_pages
        )
        
        # Clean up session
        del self._active_sessions[session_id]
        
        return lossless_pages, memo_page
    
    def _create_session_pages(self, session: Session, spec_title: str) -> List[Page]:
        """Create lossless pages from session content with chunking."""
        full_content = session.get_full_content()
        
        # Implement chunking strategy for long sessions (~2048 tokens ≈ 8192 chars)
        MAX_CHUNK_SIZE = 8192
        
        if len(full_content) <= MAX_CHUNK_SIZE:
            # Single page for short sessions
            page = self.page_store.add_page(
                title=f"Session: {spec_title} ({session.session_id[:8]})",
                tags=["session", "lossless", spec_title.lower().replace(" ", "_")],
                content=full_content,
                source="memo",
                tenant_id=session.tenant_id
            )
            return [page]
        else:
            # Multiple pages for long sessions (chunking)
            pages = []
            chunks = self._chunk_content(full_content, MAX_CHUNK_SIZE)
            
            for i, chunk in enumerate(chunks):
                page = self.page_store.add_page(
                    title=f"Session: {spec_title} ({session.session_id[:8]}) - Part {i+1}",
                    tags=["session", "lossless", "chunked", spec_title.lower().replace(" ", "_")],
                    content=chunk,
                    source="memo", 
                    tenant_id=session.tenant_id
                )
                pages.append(page)
            
            return pages
    
    def _chunk_content(self, content: str, max_size: int) -> List[str]:
        """Chunk content intelligently at natural boundaries."""
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            if current_size + line_size > max_size and current_chunk:
                # Finish current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _create_contextual_memo(
        self, 
        session: Session, 
        spec_title: str, 
        endpoints_count: int,
        tests_generated: int, 
        key_decisions: List[str],
        issues_found: List[str],
        lossless_pages: List[Page]
    ) -> Page:
        """Create contextual memo using prior memory and page_id pointers."""
        
        # Generate contextual header using prior memory
        contextual_header = self._generate_contextual_header(spec_title, session.tenant_id)
        
        # Build memo content with page_id pointers
        content_parts = [
            f"Context: {contextual_header}",
            f"Spec: {spec_title}",
            f"Endpoints: {endpoints_count}, Tests: {tests_generated}",
        ]
        
        if key_decisions:
            content_parts.append(f"Decisions: {'; '.join(key_decisions[:3])}")
        
        if issues_found:
            content_parts.append(f"Issues: {'; '.join(issues_found[:3])}")
        
        # Add page_id pointers to lossless pages
        if lossless_pages:
            page_refs = [f"page_id:{page.id}" for page in lossless_pages]
            content_parts.append(f"Full session data: {', '.join(page_refs)}")
        
        content = "\n".join(content_parts)
        
        return self.page_store.add_page(
            title=f"{contextual_header}: {spec_title} ({session.session_id[:8]})",
            tags=["memo", "run", "contextual", spec_title.lower().replace(" ", "_")],
            content=content,
            source="memo",
            tenant_id=session.tenant_id
        )
    
    def _generate_contextual_header(self, spec_title: str, tenant_id: Optional[str]) -> str:
        """Generate contextual header using prior memory."""
        # Search for related previous sessions
        related_results = self.page_store.search_bm25(
            spec_title, top_k=3, tenant_id=tenant_id
        )
        
        if not related_results:
            return "Initial Run"
        
        # Analyze prior runs for context
        prior_runs = [r[0] for r in related_results if "memo" in r[0].tags]
        
        if len(prior_runs) == 0:
            return "First API Analysis"
        elif len(prior_runs) == 1:
            return "Follow-up Analysis" 
        elif any("v2" in r.title.lower() or "enhanced" in r.content.lower() for r in prior_runs):
            return "Enhanced Version Analysis"
        else:
            return f"Iteration {len(prior_runs) + 1} Analysis"
    
    def create_memo(
        self,
        run_id: str,
        spec_title: str,
        endpoints_count: int,
        tests_generated: int,
        key_decisions: List[str],
        issues_found: List[str]
    ) -> Page:
        """
        Create a memo summarizing an agent run.
        
        Args:
            run_id: Unique run identifier
            spec_title: Title of the processed spec
            endpoints_count: Number of endpoints detected
            tests_generated: Number of tests generated
            key_decisions: Key decisions made during generation
            issues_found: Issues or missing info found
            
        Returns:
            Created memo Page
        """
        content_parts = [
            f"Spec: {spec_title}",
            f"Endpoints: {endpoints_count}, Tests: {tests_generated}",
        ]
        
        if key_decisions:
            content_parts.append(f"Decisions: {'; '.join(key_decisions[:3])}")
        
        if issues_found:
            content_parts.append(f"Issues: {'; '.join(issues_found[:3])}")
        
        content = "\n".join(content_parts)
        
        return self.page_store.add_page(
            title=f"Run Memo: {spec_title} ({run_id[:8]})",
            tags=["memo", "run", spec_title.lower().replace(" ", "_")],
            content=content,
            source="memo"
        )
    
    def store_artifact(
        self,
        title: str,
        content: str,
        artifact_type: Literal["existing_tests", "runbook", "validator"]
    ) -> Page:
        """
        Store an artifact as a page.
        
        Args:
            title: Artifact title
            content: Artifact content
            artifact_type: Type of artifact
            
        Returns:
            Created Page
        """
        return self.page_store.add_page(
            title=title,
            tags=[artifact_type, "artifact"],
            content=content,
            source=artifact_type
        )


@dataclass
class ResearchResult:
    """Result of a research iteration."""
    plan: List[str]
    memory_excerpts: List[Dict[str, str]]
    reflection: str
    should_continue: bool
    iteration: int


class Researcher:
    """
    Deep-research loop: plan → search → integrate → reflect.
    
    Max 2 iterations per the spec requirement.
    """
    
    MAX_REFLECTIONS = 2
    MAX_EXCERPTS = 5
    MAX_EXCERPT_LENGTH = 200  # ~2 lines
    
    def __init__(self, page_store: PageStore):
        """
        Initialize Researcher.
        
        Args:
            page_store: PageStore instance
        """
        self.page_store = page_store
    
    def plan(self, context: Dict[str, Any]) -> List[str]:
        """
        Create research plan based on context.
        
        Args:
            context: Dict with spec_title, endpoints, auth_type, etc.
            
        Returns:
            List of plan steps
        """
        plan = []
        
        # Always look for conventions
        plan.append("Search for REST API testing conventions")
        
        # Auth-specific research
        auth_type = context.get("auth_type", "unknown")
        if auth_type not in ["none", "unknown"]:
            plan.append(f"Search for {auth_type} authentication testing patterns")
        
        # Endpoint-specific research
        endpoints = context.get("endpoints", [])
        methods = set(e.get("method", "") for e in endpoints)
        
        if "POST" in methods or "PUT" in methods:
            plan.append("Search for request validation testing patterns")
        
        if any("list" in e.get("path", "").lower() or 
               e.get("path", "").endswith("s") 
               for e in endpoints):
            plan.append("Search for pagination testing patterns")
        
        return plan[:4]  # Limit plan steps
    
    def search(
        self,
        plan: List[str],
        tenant_id: Optional[str] = None,
        prior_page_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Page, float]]:
        """
        Execute search based on plan.
        
        Implements retrieval tools over pages:
        - retrieve_by_query: hybrid search per query
        - retrieve_by_group: tag/group-based retrieval
        - retrieve_by_page_ids: explicit prior page references
        
        Retrieval calls are executed in parallel and merged.
        
        Args:
            plan: List of search queries
            tenant_id: Optional tenant scope
            prior_page_ids: Optional page IDs from previous iteration
            
        Returns:
            List of (Page, score) tuples
        """
        all_results: Dict[str, Tuple[Page, float]] = {}

        futures = []
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(plan) + 3))) as executor:
            for query in plan:
                futures.append(
                    executor.submit(self.retrieve_by_query, query, tenant_id)
                )
            
            group_tags = self._derive_group_tags(plan)
            if group_tags:
                futures.append(
                    executor.submit(self.retrieve_by_group, group_tags, tenant_id)
                )
            
            if prior_page_ids:
                futures.append(
                    executor.submit(
                        self.retrieve_by_page_ids,
                        prior_page_ids[: self.MAX_EXCERPTS * 2],
                        tenant_id,
                    )
                )
            
            for future in as_completed(futures):
                results = future.result()
                for page, score in results:
                    if page.id not in all_results or all_results[page.id][1] < score:
                        all_results[page.id] = (page, score)
        
        # Sort by score and return
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:self.MAX_EXCERPTS * 2]

    def retrieve_by_query(
        self, query: str, tenant_id: Optional[str] = None
    ) -> List[Tuple[Page, float]]:
        """Tool: retrieve pages by free-text query."""
        return self.page_store.hybrid_search(query, top_k=3, tenant_id=tenant_id)

    def retrieve_by_group(
        self, group_tags: List[str], tenant_id: Optional[str] = None
    ) -> List[Tuple[Page, float]]:
        """Tool: retrieve pages by semantic group tags."""
        pages = self.page_store.search_by_tags(
            group_tags, top_k=self.MAX_EXCERPTS * 2, tenant_id=tenant_id
        )
        # Tag retrieval has weaker evidence than exact query matches.
        return [(page, 0.35 - idx * 0.02) for idx, page in enumerate(pages)]

    def retrieve_by_page_ids(
        self, page_ids: List[str], tenant_id: Optional[str] = None
    ) -> List[Tuple[Page, float]]:
        """Tool: retrieve previously known page IDs."""
        pages = self.page_store.search_by_page_ids(page_ids, tenant_id=tenant_id)
        # Direct page-id retrieval is high-confidence.
        return [(page, 0.60 - idx * 0.03) for idx, page in enumerate(pages)]

    def _derive_group_tags(self, plan: List[str]) -> List[str]:
        """Infer group tags from plan items."""
        tags = set()
        plan_text = " ".join(plan).lower()

        mapping = {
            "auth": ["auth", "security"],
            "security": ["security", "auth"],
            "validation": ["validation", "negative"],
            "schema": ["schema", "contract", "validator"],
            "pagination": ["pagination", "list"],
            "rest": ["rest", "testing", "convention"],
        }

        for token, mapped in mapping.items():
            if token in plan_text:
                tags.update(mapped)

        if not tags:
            tags.add("convention")

        return list(tags)
    
    def integrate(
        self,
        search_results: List[Tuple[Page, float]]
    ) -> List[Dict[str, str]]:
        """
        Integrate search results into memory excerpts.
        
        Args:
            search_results: List of (Page, score) tuples
            
        Returns:
            List of memory excerpt dicts
        """
        excerpts = []
        
        for page, score in search_results[:self.MAX_EXCERPTS]:
            # Truncate content to ~2 lines
            content = page.content
            if len(content) > self.MAX_EXCERPT_LENGTH:
                content = content[:self.MAX_EXCERPT_LENGTH] + "..."
            
            excerpts.append({
                "source": page.source,
                "excerpt": content
            })
        
        return excerpts
    
    def reflect(
        self,
        context: Dict[str, Any],
        excerpts: List[Dict[str, str]],
        iteration: int
    ) -> Tuple[str, bool]:
        """
        Reflect on research quality and decide if another iteration is needed.
        
        Args:
            context: Research context
            excerpts: Current excerpts
            iteration: Current iteration number
            
        Returns:
            (reflection_text, should_continue)
        """
        if iteration >= self.MAX_REFLECTIONS:
            return (
                f"Completed {iteration} research iterations. "
                f"Found {len(excerpts)} relevant excerpts covering conventions and patterns.",
                False
            )
        
        # Check coverage
        sources = set(e["source"] for e in excerpts)
        missing = []
        
        if "convention" not in sources:
            missing.append("testing conventions")
        
        auth_type = context.get("auth_type", "unknown")
        if auth_type not in ["none", "unknown"] and not any(
            "auth" in e["excerpt"].lower() for e in excerpts
        ):
            missing.append("auth testing patterns")
        
        if missing and iteration < self.MAX_REFLECTIONS:
            return (
                f"Iteration {iteration}: Found {len(excerpts)} excerpts. "
                f"Missing coverage for: {', '.join(missing)}. Continuing research.",
                True
            )
        
        return (
            f"Research complete after {iteration} iteration(s). "
            f"Found {len(excerpts)} relevant excerpts from {len(sources)} sources.",
            False
        )
    
    def research(self, context: Dict[str, Any]) -> ResearchResult:
        """
        Execute full research loop.
        
        Args:
            context: Dict with spec_title, endpoints, auth_type, etc.
            
        Returns:
            ResearchResult with plan, excerpts, and reflection
        """
        all_excerpts: List[Dict[str, str]] = []
        all_plan: List[str] = []
        tracked_page_ids: List[str] = []
        tenant_id = context.get("tenant_id")
        
        for iteration in range(1, self.MAX_REFLECTIONS + 1):
            # Plan
            plan = self.plan(context)
            all_plan.extend(plan)
            
            # Search
            results = self.search(
                plan,
                tenant_id=tenant_id,
                prior_page_ids=tracked_page_ids,
            )
            tracked_page_ids = [page.id for page, _ in results]
            
            # Integrate
            excerpts = self.integrate(results)
            
            # Deduplicate excerpts
            seen = set()
            for exc in excerpts:
                key = exc["excerpt"][:50]
                if key not in seen:
                    seen.add(key)
                    all_excerpts.append(exc)
            
            # Reflect
            reflection, should_continue = self.reflect(context, all_excerpts, iteration)
            
            if not should_continue:
                return ResearchResult(
                    plan=list(set(all_plan)),
                    memory_excerpts=all_excerpts[:self.MAX_EXCERPTS],
                    reflection=reflection,
                    should_continue=False,
                    iteration=iteration
                )
        
        return ResearchResult(
            plan=list(set(all_plan)),
            memory_excerpts=all_excerpts[:self.MAX_EXCERPTS],
            reflection=f"Completed maximum {self.MAX_REFLECTIONS} iterations.",
            should_continue=False,
            iteration=self.MAX_REFLECTIONS
        )


class GAMMemorySystem:
    """
    Complete GAM-style memory system combining PageStore, Memorizer, and Researcher.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_vector_search: bool = True
    ):
        """
        Initialize GAM memory system.
        
        Args:
            embedding_model: Sentence transformer model name
            use_vector_search: Whether to use vector search
        """
        self.page_store = PageStore(
            embedding_model=embedding_model,
            use_vector_search=use_vector_search
        )
        self.memorizer = Memorizer(self.page_store)
        self.researcher = Researcher(self.page_store)
    
    def research(self, context: Dict[str, Any]) -> ResearchResult:
        """Execute research loop."""
        return self.researcher.research(context)
    
    # Legacy API (backward compatibility)
    def create_memo(self, **kwargs) -> Page:
        """Create a run memo (legacy method)."""
        return self.memorizer.create_memo(**kwargs)
    
    def add_page(self, **kwargs) -> Page:
        """Add a page to the store."""
        return self.page_store.add_page(**kwargs)
    
    def search(self, query: str, top_k: int = 5, tenant_id: Optional[str] = None) -> List[Tuple[Page, float]]:
        """Hybrid search with optional tenant scoping."""
        return self.page_store.hybrid_search(query, top_k, tenant_id=tenant_id)
    
    # Enhanced session-based API
    def start_session(self, tenant_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new GAM session."""
        return self.memorizer.start_session(tenant_id=tenant_id, metadata=metadata)
    
    def add_to_session(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None
    ):
        """Add content to active session."""
        return self.memorizer.add_to_session(
            session_id, role, content, tool_outputs, artifacts
        )
    
    def end_session_with_memo(
        self,
        session_id: str,
        spec_title: str,
        endpoints_count: int,
        tests_generated: int,
        key_decisions: List[str],
        issues_found: List[str]
    ) -> Tuple[List[Page], Page]:
        """End session and create lossless pages + contextual memo."""
        return self.memorizer.end_session_with_memo(
            session_id, spec_title, endpoints_count, tests_generated,
            key_decisions, issues_found
        )

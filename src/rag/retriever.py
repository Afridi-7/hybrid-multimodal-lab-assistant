"""
Clinical knowledge retriever using ChromaDB or numpy fallback.

Provides semantic search over hematology knowledge base with embeddings and
best-effort internet-backed evidence augmentation.
"""

import html as html_lib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import parse_qs, unquote, urlparse
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.config.config_loader import Config
from src.rag.pdf_processor import process_pdf_library, validate_pdf_library

logger = logging.getLogger(__name__)


class ClinicalRetriever:
    """Retrieval system for hematology clinical knowledge."""
    
    def __init__(self, config: Config):
        """
        Initialize clinical retriever.
        
        Args:
            config: Configuration object with RAG parameters
        """
        self.config = config

        # Load configuration
        self.pdf_directory = config.get('rag.pdf_directory', 'LLM_RAG_Pipline/pdfs')
        # Resolve relative pdf_directory against the repo root so that the
        # pipeline works regardless of the process's current working directory
        # (e.g., uvicorn launched from backend/ instead of repo root).
        _pdf_path = Path(self.pdf_directory)
        if not _pdf_path.is_absolute():
            _repo_root = Path(__file__).resolve().parent.parent.parent
            self.pdf_directory = str((_repo_root / _pdf_path).resolve())
        self.pdf_sources = config.get('rag.pdf_sources', [])
        self.pdf_missing_strategy = config.get('rag.pdf_missing_strategy', 'fail')
        self.pdf_download_base_url = config.get('rag.pdf_download_base_url', '')
        self.pdf_download_timeout_seconds = config.get('rag.pdf_download_timeout_seconds', 30)
        self.internet_enabled = config.get('rag.internet.enabled', True)
        self.internet_top_k = config.get('rag.internet.top_k', 3)
        self.internet_search_timeout_seconds = config.get('rag.internet.search_timeout_seconds', 10)
        self.internet_fetch_timeout_seconds = config.get('rag.internet.fetch_timeout_seconds', 10)
        self.internet_max_chars = config.get('rag.internet.max_chars_per_source', 4000)
        self.internet_user_agent = config.get(
            'rag.internet.user_agent',
            'BloodSmearDomainExpert/1.0 (+https://github.com/)'
        )
        # Optional allow-list of domain suffixes for web augmentation. When
        # set, only URLs whose hostname ends in one of these suffixes are
        # fetched. This is the recommended setting for clinical use.
        allow = config.get('rag.internet.allowed_domains', []) or []
        self.internet_allowed_domains = [d.lower().lstrip('.') for d in allow]
        self.chunk_size = config.get('rag.chunking.chunk_size', 500)
        self.overlap = config.get('rag.chunking.overlap', 50)
        self.top_k = config.get('rag.retrieval.top_k', 5)
        self.min_chunk_length = config.get('rag.retrieval.min_chunk_length', 80)
        
        # Embedding configuration
        embedding_model_name = config.get('rag.embedding.model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_dim = config.get('rag.embedding.embedding_dim', 384)
        self.batch_size = config.get('rag.embedding.batch_size', 64)
        self.embedding_model_name = embedding_model_name
        # Optional text put in front of every *query* (not the chunks). Some
        # retrieval models, e.g. the BGE family, are trained with such a prefix.
        self.query_instruction = config.get('rag.embedding.query_instruction', '') or ''
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        
        # Vector store configuration
        self.vector_store_type = config.get('rag.vector_store.type', 'chromadb')
        self.chromadb_path = config.get('rag.vector_store.chromadb_path', 'LLM_RAG_Pipline/chroma_db')
        # Resolve relative chromadb_path against the repo root for the same
        # cwd-independence reason as pdf_directory above.
        _chroma_path = Path(self.chromadb_path)
        if not _chroma_path.is_absolute():
            _repo_root = Path(__file__).resolve().parent.parent.parent
            self.chromadb_path = str((_repo_root / _chroma_path).resolve())
        self.collection_name = config.get('rag.vector_store.collection_name', 'hematology_knowledge')
        
        # Storage
        self.chunks = []
        self.embeddings = None
        self.chroma_client = None
        self.chroma_collection = None
        self._web_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("ClinicalRetriever initialized")
    
    def build_index(self) -> Dict[str, Any]:
        """
        Build retrieval index from PDF library.
        
        Returns:
            Dictionary with build statistics:
                {
                    'total_chunks': int,
                    'pdfs_processed': int,
                    'index_type': str,
                    'embedding_dim': int
                }
                
        Examples:
            >>> retriever = ClinicalRetriever(config)
            >>> stats = retriever.build_index()
            >>> print(f"Indexed {stats['total_chunks']} chunks")
        """
        logger.info("Building retrieval index...")
        
        # Validate PDF library
        validation = validate_pdf_library(self.pdf_directory, self.pdf_sources)
        if not validation['valid']:
            logger.warning(f"Missing PDFs: {validation['missing_files']}")
            self._handle_missing_pdfs(validation)
        
        # Process PDFs into chunks
        self.chunks = process_pdf_library(
            self.pdf_directory,
            self.pdf_sources,
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        if not self.chunks:
            if self.internet_enabled:
                logger.warning("No local PDF chunks extracted; proceeding with internet-backed retrieval only")
                self.embeddings = None
                return {
                    'total_chunks': 0,
                    'pdfs_processed': 0,
                    'index_type': 'internet_only',
                    'embedding_dim': self.embedding_dim
                }

            raise ValueError("No chunks extracted from PDF library")
        
        logger.info(f"Extracted {len(self.chunks)} chunks from PDFs")
        
        # Create embeddings
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        logger.info(f"Encoding {len(chunk_texts)} chunks...")
        
        self.embeddings = self.embedder.encode(
            chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Created embeddings: {self.embeddings.shape}")
        
        # Initialize vector store
        if self.vector_store_type == 'chromadb':
            try:
                self._init_chromadb()
                index_type = 'chromadb'
            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                logger.info("Falling back to numpy-based retrieval")
                index_type = 'numpy_fallback'
        else:
            index_type = 'numpy_fallback'
        
        return {
            'total_chunks': len(self.chunks),
            'pdfs_processed': len(set(c['source'] for c in self.chunks)),
            'index_type': 'hybrid' if self.internet_enabled else index_type,
            'embedding_dim': self.embedding_dim
        }

    def _handle_missing_pdfs(self, validation: Dict[str, Any]) -> None:
        """Apply configured strategy when required PDFs are missing."""
        strategy = str(self.pdf_missing_strategy).lower()
        missing = validation.get('missing_files', [])
        if not missing:
            return

        if strategy == 'download':
            self._download_missing_pdfs(missing)
            post_check = validate_pdf_library(self.pdf_directory, self.pdf_sources)
            if not post_check['valid']:
                raise FileNotFoundError(
                    f"PDF auto-download incomplete. Missing files: {post_check['missing_files']}. "
                    f"Place files under '{self.pdf_directory}' or set rag.pdf_missing_strategy to 'warn'."
                )
            return

        if strategy == 'warn':
            logger.warning(
                "Continuing with partial PDF library. Missing files: %s",
                missing
            )
            return

        if self.internet_enabled:
            logger.warning(
                "Continuing without the missing PDFs because internet-backed retrieval is enabled. Missing files: %s",
                missing
            )
            return

        raise FileNotFoundError(
            f"Missing required PDF files: {missing}. "
            f"Expected directory: '{self.pdf_directory}'. "
            "Either add the files, set rag.pdf_missing_strategy to 'warn', "
            "or configure rag.pdf_missing_strategy='download' with rag.pdf_download_base_url."
        )

    def _download_missing_pdfs(self, missing_files: List[str]) -> None:
        """Download missing PDF files from configured base URL."""
        if not self.pdf_download_base_url:
            raise ValueError(
                "rag.pdf_missing_strategy is 'download' but rag.pdf_download_base_url is empty"
            )

        target_dir = Path(self.pdf_directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        base = self.pdf_download_base_url.rstrip('/')
        for filename in missing_files:
            url = f"{base}/{filename}"
            target_path = target_dir / filename
            logger.info(f"Downloading missing PDF: {url}")
            try:
                response = requests.get(url, timeout=self.pdf_download_timeout_seconds)
                response.raise_for_status()
                target_path.write_bytes(response.content)
                logger.info(f"Downloaded PDF to: {target_path}")
            except Exception as exc:
                logger.error(f"Failed to download {filename}: {exc}")
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB vector store, reusing existing index when fresh."""
        import chromadb
        from chromadb.config import Settings

        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(Path(self.chromadb_path)),
            settings=Settings(anonymized_telemetry=False)
        )

        expected_count = len(self.chunks)
        existing = None
        try:
            existing = self.chroma_client.get_collection(name=self.collection_name)
        except Exception:
            existing = None

        if existing is not None:
            try:
                existing_count = existing.count()
            except Exception:
                existing_count = -1

            try:
                existing_model = (existing.metadata or {}).get('embedding_model')
            except Exception:
                existing_model = None

            # Reuse only if the stored vectors were made by the SAME embedding
            # model. Counting chunks alone is not enough: changing the model
            # keeps the count but makes every stored vector incompatible.
            if (existing_count == expected_count and expected_count > 0
                    and existing_model == self.embedding_model_name):
                logger.info(
                    "Reusing existing ChromaDB collection '%s' with %d chunks",
                    self.collection_name, existing_count
                )
                self.chroma_collection = existing
                return

            logger.info(
                "Existing ChromaDB collection is stale (chunks have=%s, expected=%d; "
                "model have=%s, expected=%s); rebuilding",
                existing_count, expected_count, existing_model, self.embedding_model_name
            )
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
            except Exception as exc:
                logger.debug("delete_collection failed: %s", exc)

        # Create collection
        self.chroma_collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model_name}
        )
        
        # Add documents
        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        documents = [chunk['text'] for chunk in self.chunks]
        metadatas = [
            {
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'total_chunks': chunk['total_chunks']
            }
            for chunk in self.chunks
        ]
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.chroma_collection.add(
                ids=ids[i:batch_end],
                embeddings=self.embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        logger.info(f"Added {len(ids)} chunks to ChromaDB collection")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for query.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve (None to use config default)
            
        Returns:
            List of retrieved chunks with metadata and scores:
                [
                    {
                        'text': str,
                        'source': str,
                        'score': float,
                        'chunk_id': int
                    },
                    ...
                ]
                
        Examples:
            >>> results = retriever.retrieve("What causes leukocytosis?")
            >>> for result in results:
            ...     print(f"[{result['source']}]: {result['text'][:100]}...")
        """
        k = top_k if top_k is not None else self.top_k
        
        logger.info(f"Retrieving top-{k} chunks for query: {query[:50]}...")
        
        # Encode query
        query_text = getattr(self, 'query_instruction', '') + query
        query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)[0]

        local_results: List[Dict[str, Any]] = []
        if self.chunks:
            if self.chroma_collection is not None:
                local_results = self._retrieve_chromadb(query_embedding, k)
            elif self.embeddings is not None:
                local_results = self._retrieve_numpy(query_embedding, k)

        web_results: List[Dict[str, Any]] = []
        if self.internet_enabled:
            web_results = self._retrieve_internet(query, query_embedding, k)

        return self._merge_retrieval_results(local_results, web_results, k)

    def _retrieve_internet(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve best-effort internet evidence for the query."""
        cache_key = query.strip().lower()
        if cache_key in self._web_cache:
            return self._web_cache[cache_key][:k]

        try:
            search_results = self._search_web(query, max_results=max(k, self.internet_top_k))
        except Exception as exc:
            logger.warning(f"Web search failed: {exc}")
            return []

        candidates: List[Dict[str, Any]] = []
        for rank, result in enumerate(search_results, 1):
            if not self._url_allowed(result['url']):
                logger.debug("Skipping disallowed URL (not in allow-list): %s", result['url'])
                continue
            try:
                page_text = self._fetch_web_page_text(result['url'])
            except Exception as exc:
                logger.debug(f"Failed to fetch {result['url']}: {exc}")
                page_text = ""

            text_parts = [result['title']]
            if page_text:
                text_parts.append(page_text)
            elif result.get('snippet'):
                text_parts.append(result['snippet'])

            candidate_text = "\n".join(part for part in text_parts if part).strip()
            if len(candidate_text) < self.min_chunk_length:
                continue

            candidate_embedding = self.embedder.encode([candidate_text[: self.internet_max_chars]], convert_to_numpy=True)[0]
            similarity = self._cosine_similarity(query_embedding, candidate_embedding)
            source_label = f"web:{urlparse(result['url']).netloc or result['url']}"

            candidates.append({
                'text': candidate_text,
                'source': source_label,
                'url': result['url'],
                'title': result['title'],
                'score': float(similarity),
                'chunk_id': rank,
                'total_chunks': len(search_results),
                'source_type': 'web',
            })

        candidates.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        self._web_cache[cache_key] = candidates
        logger.info(f"Retrieved {len(candidates)} chunks from the web")
        return candidates[:k]

    def _search_web(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search the public web for query-relevant pages."""
        search_url = "https://html.duckduckgo.com/html/"
        response = requests.get(
            search_url,
            params={'q': query},
            headers={'User-Agent': self.internet_user_agent},
            timeout=self.internet_search_timeout_seconds,
        )
        response.raise_for_status()

        html_text = response.text
        results: List[Dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(html_text):
            raw_url = html_lib.unescape(match.group(1))
            title = self._clean_html_text(html_lib.unescape(match.group(2)))
            url = self._normalize_search_result_url(raw_url)

            if not url:
                continue

            results.append({
                'title': title or url,
                'url': url,
                'snippet': '',
            })

            if len(results) >= max_results:
                break

        return results

    def _normalize_search_result_url(self, raw_url: str) -> str:
        """Normalize search-engine redirect URLs to direct links."""
        decoded_url = html_lib.unescape(raw_url)
        parsed = urlparse(decoded_url)
        query = parse_qs(parsed.query)

        if 'uddg' in query and query['uddg']:
            return unquote(query['uddg'][0])

        if parsed.scheme in {'http', 'https'}:
            return decoded_url

        if decoded_url.startswith('//'):
            return f"https:{decoded_url}"

        if decoded_url.startswith('/'):
            return f"https://duckduckgo.com{decoded_url}"

        return decoded_url if decoded_url.startswith('http') else ''

    def _url_allowed(self, url: str) -> bool:
        """Return True if URL is permitted by the configured allow-list.

        An empty allow-list means "any host" (preserves prior behaviour).
        Only http(s) schemes are ever allowed.
        """
        if not url:
            return False
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme not in {'http', 'https'}:
            return False
        host = (parsed.hostname or '').lower()
        if not host:
            return False
        if not self.internet_allowed_domains:
            return True
        return any(
            host == d or host.endswith('.' + d)
            for d in self.internet_allowed_domains
        )

    def _fetch_web_page_text(self, url: str) -> str:
        """Fetch and clean a web page into plain text."""
        response = requests.get(
            url,
            headers={'User-Agent': self.internet_user_agent},
            timeout=self.internet_fetch_timeout_seconds,
        )
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
            return self._clean_text(response.text)

        return self._extract_text_from_html(response.text)

    def _extract_text_from_html(self, html_text: str) -> str:
        """Convert HTML content into readable plain text."""
        html_text = re.sub(r'(?is)<(script|style|noscript).*?>.*?</\1>', ' ', html_text)
        html_text = re.sub(r'(?is)<head.*?>.*?</head>', ' ', html_text)
        html_text = re.sub(r'(?is)<br\s*/?>', '\n', html_text)
        html_text = re.sub(r'(?is)</p\s*>', '\n', html_text)
        html_text = re.sub(r'(?is)<[^>]+>', ' ', html_text)
        html_text = html_lib.unescape(html_text)
        return self._clean_text(html_text)

    def _clean_html_text(self, text: str) -> str:
        """Clean HTML fragments returned by search results."""
        return self._clean_text(html_lib.unescape(re.sub(r'(?is)<[^>]+>', ' ', text)))

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace in extracted text."""
        return re.sub(r'\s+', ' ', text).strip()

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Compute cosine similarity with numerical stability."""
        epsilon = 1e-8
        denominator = (np.linalg.norm(left) * np.linalg.norm(right)) + epsilon
        return float(np.dot(left, right) / denominator)

    def _merge_retrieval_results(
        self,
        local_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate local and web retrieval results."""
        merged: List[Dict[str, Any]] = []
        seen = set()

        for item in local_results + web_results:
            key = (
                item.get('source', ''),
                item.get('chunk_id', ''),
                item.get('text', '')[:200],
            )
            if key in seen:
                continue

            seen.add(key)
            merged.append(item)

        merged.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        return merged[:k]
    
    def _retrieve_chromadb(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Retrieve using ChromaDB."""
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Format results
        retrieved = []
        for i in range(len(results['documents'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'score': float(1.0 - results['distances'][0][i]),  # Convert distance to similarity
                'chunk_id': results['metadatas'][0][i]['chunk_id']
            })
        
        logger.info(f"Retrieved {len(retrieved)} chunks using ChromaDB")
        return retrieved
    
    def _retrieve_numpy(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Retrieve using numpy cosine similarity."""
        # Compute cosine similarities with numerical stability
        epsilon = 1e-8
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Normalize with epsilon to prevent division by zero
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + epsilon
        similarities = similarities / norms
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Format results
        retrieved = []
        for idx in top_indices:
            retrieved.append({
                'text': self.chunks[idx]['text'],
                'source': self.chunks[idx]['source'],
                'score': float(similarities[idx]),
                'chunk_id': self.chunks[idx]['chunk_id']
            })
        
        logger.info(f"Retrieved {len(retrieved)} chunks using numpy fallback")
        return retrieved
    
    def format_retrieved_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into context string with citations.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string with numbered citations
            
        Examples:
            >>> context = retriever.format_retrieved_context(results)
            >>> print(context)
            [Reference 1 — essentials_haematology.pdf]
            Leukocytosis is an increase in white blood cell count...
            
            [Reference 2 — concise_haematology.pdf]
            Common causes include infection, inflammation...
        """
        if not retrieved_chunks:
            return ""
        
        formatted_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = f"[Reference {i} — {chunk['source']}]"
            formatted_parts.append(f"{citation}\n{chunk['text']}")
        
        return "\n\n".join(formatted_parts)
    
    def check_retrieval_quality(self, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check quality of retrieved results.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Dictionary with quality metrics:
                {
                    'sufficient_evidence': bool,
                    'total_length': int,
                    'num_chunks': int,
                    'avg_score': float
                }
        """
        if not retrieved_chunks:
            return {
                'sufficient_evidence': False,
                'total_length': 0,
                'num_chunks': 0,
                'avg_score': 0.0
            }
        
        total_text = ''.join(c['text'] for c in retrieved_chunks)
        avg_score = np.mean([c['score'] for c in retrieved_chunks])
        
        return {
            'sufficient_evidence': len(total_text) >= self.min_chunk_length,
            'total_length': len(total_text),
            'num_chunks': len(retrieved_chunks),
            'avg_score': float(avg_score)
        }

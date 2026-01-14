"""
Retrieval Module

Implements semantic search, reranking, and context preparation for RAG.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from openai import OpenAI

import config
from embedder import EmbeddedChunk, EmbeddingGenerator
from vector_store import PgVectorStore, get_vector_store

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk: EmbeddedChunk
    score: float
    rerank_score: Optional[float] = None
    
    @property
    def final_score(self) -> float:
        """Get final score (rerank if available, else original)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "metadata": self.chunk.metadata,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
        }


class Retriever:
    """
    Retrieves relevant context for RAG using semantic search.
    """
    
    def __init__(
        self,
        vector_store: PgVectorStore = None,
        embedding_generator: EmbeddingGenerator = None,
        top_k: int = None,
        rerank: bool = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.top_k = top_k or config.TOP_K
        self.rerank = rerank if rerank is not None else config.RERANK_ENABLED
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        use_hybrid: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Number of results to return
            filters: Metadata filters
            use_hybrid: Use hybrid search (semantic + keyword)
        
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        if use_hybrid:
            raw_results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k * 2 if self.rerank else top_k,
            )
        else:
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2 if self.rerank else top_k,
                filters=filters,
            )
        
        # Convert to RetrievalResult
        results = [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in raw_results
        ]
        
        # Optionally rerank
        if self.rerank and results:
            results = self._rerank_results(query, results)
        
        # Return top_k
        results = sorted(results, key=lambda x: x.final_score, reverse=True)
        return results[:top_k]
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Rerank results using LLM-based scoring.
        
        Uses a lightweight prompt to score relevance of each result.
        """
        reranked = []
        
        for result in results:
            # Score relevance using GPT
            score = self._score_relevance(query, result.chunk.content)
            result.rerank_score = score
            reranked.append(result)
        
        return reranked
    
    def _score_relevance(self, query: str, content: str) -> float:
        """
        Score relevance of content to query using LLM.
        
        Returns score between 0 and 1.
        """
        # Truncate content if too long
        max_content_len = 500
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        
        prompt = f"""Rate the relevance of the following document excerpt to the query on a scale of 0-10.
Only respond with a single number.

Query: {query}

Document excerpt:
{content}

Relevance score (0-10):"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost efficiency
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text) / 10  # Normalize to 0-1
            return min(max(score, 0), 1)
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return 0.5  # Default middle score
    
    def retrieve_with_context(
        self,
        query: str,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: User query
            max_tokens: Maximum tokens for context
            **kwargs: Additional arguments for retrieve()
        
        Returns:
            Tuple of (formatted_context, results)
        """
        results = self.retrieve(query, **kwargs)
        
        # Build context string within token limit
        context_parts = []
        total_tokens = 0
        included_results = []
        
        for result in results:
            chunk_tokens = result.chunk.token_count
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            # Format chunk with metadata
            source_info = self._format_source_info(result.chunk.metadata)
            chunk_text = f"[Source: {source_info}]\n{result.chunk.content}"
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            included_results.append(result)
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(
            f"Retrieved {len(included_results)} chunks, "
            f"~{total_tokens} tokens for context"
        )
        
        return context, included_results
    
    def _format_source_info(self, metadata: Dict) -> str:
        """Format metadata as source citation."""
        parts = []
        
        if metadata.get("company"):
            parts.append(metadata["company"])
        elif metadata.get("ticker"):
            parts.append(metadata["ticker"])
        
        if metadata.get("filing_type"):
            parts.append(metadata["filing_type"])
        
        if metadata.get("filing_date"):
            parts.append(metadata["filing_date"])
        
        if metadata.get("section"):
            parts.append(f"Section: {metadata['section']}")
        
        return ", ".join(parts) if parts else "Unknown source"


class MultiQueryRetriever(Retriever):
    """
    Enhanced retriever that generates multiple query variations
    to improve recall.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = 3
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        """
        Retrieve using multiple query variations.
        """
        top_k = top_k or self.top_k
        
        # Generate query variations
        queries = self._generate_query_variations(query)
        
        # Retrieve for each query
        all_results = {}
        
        for q in queries:
            results = super().retrieve(q, top_k=top_k, **kwargs)
            
            for result in results:
                chunk_id = result.chunk.chunk_id
                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                else:
                    # Keep best score
                    if result.final_score > all_results[chunk_id].final_score:
                        all_results[chunk_id] = result
        
        # Sort and return top_k
        results = list(all_results.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:top_k]
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate alternative phrasings of the query."""
        prompt = f"""Generate {self.num_queries - 1} alternative phrasings of the following question.
Each alternative should capture the same intent but use different words.
Return only the alternatives, one per line.

Original question: {query}

Alternatives:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            
            alternatives = response.choices[0].message.content.strip().split("\n")
            alternatives = [a.strip() for a in alternatives if a.strip()]
            
            return [query] + alternatives[:self.num_queries - 1]
            
        except Exception as e:
            logger.warning(f"Query variation generation failed: {e}")
            return [query]


class ContextualCompressionRetriever(Retriever):
    """
    Retriever that compresses/extracts only relevant parts
    from retrieved documents.
    """
    
    def retrieve_with_context(
        self,
        query: str,
        max_tokens: int = 4000,
        **kwargs,
    ) -> Tuple[str, List[RetrievalResult]]:
        """
        Retrieve and compress context to most relevant parts.
        """
        # Get initial results
        results = self.retrieve(query, top_k=self.top_k * 2, **kwargs)
        
        # Compress each result
        compressed_parts = []
        total_tokens = 0
        included_results = []
        
        for result in results:
            compressed = self._compress_chunk(query, result.chunk.content)
            
            if not compressed:
                continue
            
            # Estimate tokens
            chunk_tokens = len(compressed) // 4
            
            if total_tokens + chunk_tokens > max_tokens:
                break
            
            source_info = self._format_source_info(result.chunk.metadata)
            compressed_parts.append(f"[Source: {source_info}]\n{compressed}")
            total_tokens += chunk_tokens
            included_results.append(result)
        
        context = "\n\n---\n\n".join(compressed_parts)
        
        return context, included_results
    
    def _compress_chunk(self, query: str, content: str) -> str:
        """Extract only relevant parts from content."""
        prompt = f"""Extract only the parts of the following document that are directly relevant to the question.
If nothing is relevant, respond with "NOT_RELEVANT".
Be concise but preserve important details and numbers.

Question: {query}

Document:
{content[:2000]}

Relevant extract:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )
            
            extract = response.choices[0].message.content.strip()
            
            if extract == "NOT_RELEVANT":
                return ""
            
            return extract
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return content[:500]


# Factory function
def get_retriever(
    retriever_type: str = "standard",
    **kwargs,
) -> Retriever:
    """
    Get retriever instance.
    
    Args:
        retriever_type: "standard", "multi_query", or "compression"
    
    Returns:
        Retriever instance
    """
    if retriever_type == "multi_query":
        return MultiQueryRetriever(**kwargs)
    elif retriever_type == "compression":
        return ContextualCompressionRetriever(**kwargs)
    return Retriever(**kwargs)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test retrieval")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=["standard", "multi_query", "compression"])
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--ticker", type=str, help="Filter by ticker")
    
    args = parser.parse_args()
    
    # Initialize
    vector_store = get_vector_store()
    retriever = get_retriever(
        retriever_type=args.retriever or "standard",
        vector_store=vector_store,
    )
    
    # Build filters
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    
    # Retrieve
    context, results = retriever.retrieve_with_context(
        query=args.query,
        filters=filters if filters else None,
        use_hybrid=args.hybrid,
    )
    
    print(f"\n🔍 Query: {args.query}")
    print(f"📊 Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- Result {i} (score: {result.final_score:.3f}) ---")
        print(f"Source: {retriever._format_source_info(result.chunk.metadata)}")
        print(f"Preview: {result.chunk.content[:200]}...")
        print()

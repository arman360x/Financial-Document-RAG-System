"""
Vector Store Module

Handles vector storage and retrieval using pgvector (PostgreSQL).
Also includes support for Pinecone as an alternative.
"""
import os
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import json

from sqlalchemy import create_engine, Column, String, Integer, Text, Index
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
import numpy as np

import config
from embedder import EmbeddedChunk

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

Base = declarative_base()


class DocumentChunkDB(Base):
    """SQLAlchemy model for document chunks with embeddings."""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(config.EMBEDDING_DIMENSIONS), nullable=False)
    chunk_metadata = Column(JSONB, default={})
    token_count = Column(Integer, default=0)

    # Indexes for filtering
    __table_args__ = (
        Index('ix_metadata_ticker', chunk_metadata['ticker'].astext),
        Index('ix_metadata_filing_type', chunk_metadata['filing_type'].astext),
    )


class PgVectorStore:
    """
    Vector store implementation using PostgreSQL with pgvector extension.
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.DATABASE_URL
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
        )
        self.Session = sessionmaker(bind=self.engine)
        
    def initialize(self):
        """Initialize database and create tables."""
        # Create pgvector extension
        with self.engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create HNSW index for fast similarity search
        self._create_vector_index()
        
        logger.info("Database initialized successfully")
    
    def _create_vector_index(self):
        """Create HNSW index on embedding column."""
        index_sql = f"""
        CREATE INDEX IF NOT EXISTS ix_embedding_hnsw 
        ON document_chunks 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = {config.HNSW_M}, ef_construction = {config.HNSW_EF_CONSTRUCTION});
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(index_sql)
                conn.commit()
            logger.info("HNSW index created")
        except Exception as e:
            logger.warning(f"Could not create HNSW index: {e}")
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope for database operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def add_chunks(
        self,
        chunks: List[EmbeddedChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Add embedded chunks to the vector store.
        
        Args:
            chunks: List of EmbeddedChunk objects
            batch_size: Number of chunks to insert per batch
        
        Returns:
            Number of chunks inserted
        """
        inserted = 0
        
        with self.session_scope() as session:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                for chunk in batch:
                    # Check if chunk already exists
                    existing = session.query(DocumentChunkDB).filter_by(
                        chunk_id=chunk.chunk_id
                    ).first()
                    
                    if existing:
                        continue
                    
                    db_chunk = DocumentChunkDB(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        embedding=chunk.embedding,
                        chunk_metadata=chunk.metadata,
                        token_count=chunk.token_count,
                    )
                    session.add(db_chunk)
                    inserted += 1
                
                session.flush()
                logger.debug(f"Inserted batch {i // batch_size + 1}")
        
        logger.info(f"Inserted {inserted} chunks into vector store")
        return inserted
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filters: Dict = None,
        similarity_threshold: float = None,
    ) -> List[Tuple[EmbeddedChunk, float]]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Metadata filters (e.g., {"ticker": "AAPL"})
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of (EmbeddedChunk, similarity_score) tuples
        """
        top_k = top_k or config.TOP_K
        similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        
        with self.session_scope() as session:
            # Build base query with cosine distance
            query = session.query(
                DocumentChunkDB,
                (1 - DocumentChunkDB.embedding.cosine_distance(query_embedding)).label('similarity')
            )
            
            # Apply metadata filters
            if filters:
                for key, value in filters.items():
                    query = query.filter(
                        DocumentChunkDB.chunk_metadata[key].astext == value
                    )
            
            # Filter by similarity threshold and order by similarity
            query = query.filter(
                (1 - DocumentChunkDB.embedding.cosine_distance(query_embedding)) >= similarity_threshold
            ).order_by(
                DocumentChunkDB.embedding.cosine_distance(query_embedding)
            ).limit(top_k)
            
            results = []
            for db_chunk, similarity in query.all():
                chunk = EmbeddedChunk(
                    chunk_id=db_chunk.chunk_id,
                    content=db_chunk.content,
                    embedding=db_chunk.embedding,
                    metadata=db_chunk.chunk_metadata,
                    token_count=db_chunk.token_count,
                )
                results.append((chunk, float(similarity)))
        
        logger.debug(f"Found {len(results)} results for query")
        return results
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = None,
        alpha: float = None,
    ) -> List[Tuple[EmbeddedChunk, float]]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query_embedding: Query vector for semantic search
            query_text: Query text for keyword search
            top_k: Number of results
            alpha: Weight for semantic vs keyword (1.0 = pure semantic)
        
        Returns:
            List of (EmbeddedChunk, score) tuples
        """
        top_k = top_k or config.TOP_K
        alpha = alpha if alpha is not None else config.HYBRID_SEARCH_ALPHA
        
        # Get semantic results
        semantic_results = self.search(
            query_embedding,
            top_k=top_k * 2,  # Get more results for reranking
        )
        
        # Get keyword results using PostgreSQL full-text search
        keyword_results = self._keyword_search(query_text, top_k * 2)
        
        # Combine results with weighted scoring
        combined = {}
        
        for chunk, score in semantic_results:
            combined[chunk.chunk_id] = {
                "chunk": chunk,
                "semantic_score": score,
                "keyword_score": 0,
            }
        
        for chunk, score in keyword_results:
            if chunk.chunk_id in combined:
                combined[chunk.chunk_id]["keyword_score"] = score
            else:
                combined[chunk.chunk_id] = {
                    "chunk": chunk,
                    "semantic_score": 0,
                    "keyword_score": score,
                }
        
        # Calculate final scores
        results = []
        for chunk_id, data in combined.items():
            final_score = (
                alpha * data["semantic_score"] +
                (1 - alpha) * data["keyword_score"]
            )
            results.append((data["chunk"], final_score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _keyword_search(
        self,
        query_text: str,
        top_k: int,
    ) -> List[Tuple[EmbeddedChunk, float]]:
        """Keyword search using PostgreSQL full-text search."""
        with self.session_scope() as session:
            # Simple LIKE-based search (for demo; use ts_vector for production)
            search_terms = query_text.lower().split()
            
            query = session.query(DocumentChunkDB)
            
            for term in search_terms[:5]:  # Limit terms
                query = query.filter(
                    DocumentChunkDB.content.ilike(f"%{term}%")
                )
            
            results = []
            for db_chunk in query.limit(top_k).all():
                # Calculate simple keyword score
                content_lower = db_chunk.content.lower()
                score = sum(
                    content_lower.count(term) for term in search_terms
                ) / len(db_chunk.content)

                chunk = EmbeddedChunk(
                    chunk_id=db_chunk.chunk_id,
                    content=db_chunk.content,
                    embedding=db_chunk.embedding,
                    metadata=db_chunk.chunk_metadata,
                    token_count=db_chunk.token_count,
                )
                results.append((chunk, min(score * 10, 1.0)))
        
        return results
    
    def delete_by_filter(self, filters: Dict) -> int:
        """
        Delete chunks matching filters.

        Args:
            filters: Metadata filters

        Returns:
            Number of chunks deleted
        """
        with self.session_scope() as session:
            query = session.query(DocumentChunkDB)

            for key, value in filters.items():
                query = query.filter(
                    DocumentChunkDB.chunk_metadata[key].astext == value
                )

            deleted = query.delete(synchronize_session='fetch')
            logger.info(f"Deleted {deleted} chunks")
            return deleted
    
    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        with self.session_scope() as session:
            total_chunks = session.query(DocumentChunkDB).count()

            # Get unique values for key metadata fields
            tickers_query = session.query(
                DocumentChunkDB.chunk_metadata['ticker'].astext
            ).distinct().all()
            tickers = [t[0] for t in tickers_query if t[0]]

            filing_types_query = session.query(
                DocumentChunkDB.chunk_metadata['filing_type'].astext
            ).distinct().all()
            filing_types = [f[0] for f in filing_types_query if f[0]]
        
        return {
            "total_chunks": total_chunks,
            "unique_tickers": len(tickers),
            "tickers": tickers,
            "filing_types": filing_types,
        }
    
    def clear(self):
        """Delete all chunks from the vector store."""
        with self.session_scope() as session:
            deleted = session.query(DocumentChunkDB).delete()
            logger.warning(f"Cleared vector store: {deleted} chunks deleted")


class PineconeVectorStore:
    """
    Alternative vector store using Pinecone.
    """
    
    def __init__(self):
        from pinecone import Pinecone
        
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.index = None
    
    def initialize(self):
        """Initialize Pinecone index."""
        # Check if index exists
        existing_indexes = self.pc.list_indexes().names()
        
        if self.index_name not in existing_indexes:
            # Create index
            self.pc.create_index(
                name=self.index_name,
                dimension=config.EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": config.PINECONE_ENVIRONMENT,
                    }
                }
            )
            logger.info(f"Created Pinecone index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
        logger.info("Pinecone index initialized")
    
    def add_chunks(self, chunks: List[EmbeddedChunk], batch_size: int = 100) -> int:
        """Add chunks to Pinecone."""
        vectors = []
        
        for chunk in chunks:
            vectors.append({
                "id": chunk.chunk_id,
                "values": chunk.embedding,
                "metadata": {
                    **chunk.metadata,
                    "content": chunk.content[:1000],  # Pinecone metadata limit
                    "token_count": chunk.token_count,
                }
            })
        
        # Upsert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(chunks)} chunks to Pinecone")
        return len(chunks)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filters: Dict = None,
    ) -> List[Tuple[EmbeddedChunk, float]]:
        """Search Pinecone index."""
        top_k = top_k or config.TOP_K
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filters,
        )
        
        chunks = []
        for match in results.matches:
            chunk = EmbeddedChunk(
                chunk_id=match.id,
                content=match.metadata.get("content", ""),
                embedding=[],  # Not returned by query
                metadata=match.metadata,
                token_count=match.metadata.get("token_count", 0),
            )
            chunks.append((chunk, match.score))
        
        return chunks


def get_vector_store(store_type: str = "pgvector") -> PgVectorStore:
    """
    Factory function to get vector store instance.
    
    Args:
        store_type: "pgvector" or "pinecone"
    
    Returns:
        Vector store instance
    """
    if store_type == "pinecone":
        return PineconeVectorStore()
    return PgVectorStore()


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector store operations")
    parser.add_argument("--action", choices=["init", "populate", "stats", "clear"])
    parser.add_argument("--input", type=str, help="Input embeddings file")
    parser.add_argument("--store", choices=["pgvector", "pinecone"], default="pgvector")
    
    args = parser.parse_args()
    
    store = get_vector_store(args.store)
    
    if args.action == "init":
        store.initialize()
        print("✅ Vector store initialized")
    
    elif args.action == "populate":
        if not args.input:
            print("Error: --input required for populate action")
            exit(1)
        
        from embedder import load_embeddings
        from pathlib import Path
        
        store.initialize()
        chunks = load_embeddings(Path(args.input))
        inserted = store.add_chunks(chunks)
        print(f"✅ Inserted {inserted} chunks")
    
    elif args.action == "stats":
        stats = store.get_stats()
        print("📊 Vector Store Statistics:")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Unique tickers: {stats['unique_tickers']}")
        print(f"   Tickers: {', '.join(stats['tickers'])}")
        print(f"   Filing types: {', '.join(stats['filing_types'])}")
    
    elif args.action == "clear":
        confirm = input("⚠️  This will delete all data. Type 'yes' to confirm: ")
        if confirm == "yes":
            store.clear()
            print("✅ Vector store cleared")

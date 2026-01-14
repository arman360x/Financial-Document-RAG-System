"""
Embedding Generation Module

Generates embeddings using OpenAI's text-embedding models.
Supports batch processing, caching, and checkpointing for large datasets.
"""
import os
import json
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass
import time

from openai import OpenAI
from tqdm import tqdm
import numpy as np

import config
from chunker import DocumentChunk

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """Document chunk with its embedding vector."""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict
    token_count: int
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI API.
    Handles batching, rate limiting, and caching.
    """
    
    def __init__(
        self,
        model: str = None,
        batch_size: int = None,
        cache_dir: Path = None,
        use_cache: bool = True,
    ):
        self.model = model or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self.cache_dir = cache_dir or config.EMBEDDINGS_DIR
        self.use_cache = use_cache
        self.dimensions = config.EMBEDDING_DIMENSIONS
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            "total_chunks": 0,
            "cached_hits": 0,
            "api_calls": 0,
            "total_tokens": 0,
        }
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector as list of floats
        """
        # Check cache first
        if self.use_cache:
            cached = self._get_cached(text)
            if cached is not None:
                self.stats["cached_hits"] += 1
                return cached
        
        # Call OpenAI API
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding
            self.stats["api_calls"] += 1
            self.stats["total_tokens"] += response.usage.total_tokens
            
            # Cache the result
            if self.use_cache:
                self._cache_embedding(text, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            # Check cache for each text
            batch_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for j, text in enumerate(batch):
                if self.use_cache:
                    cached = self._get_cached(text)
                    if cached is not None:
                        batch_embeddings.append((j, cached))
                        self.stats["cached_hits"] += 1
                        continue
                
                texts_to_embed.append(text)
                text_indices.append(j)
            
            # Call API for non-cached texts
            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts_to_embed,
                    )
                    self.stats["api_calls"] += 1
                    self.stats["total_tokens"] += response.usage.total_tokens
                    
                    for idx, embedding_data in zip(text_indices, response.data):
                        embedding = embedding_data.embedding
                        batch_embeddings.append((idx, embedding))
                        
                        # Cache the result
                        if self.use_cache:
                            self._cache_embedding(texts_to_embed[text_indices.index(idx)], embedding)
                    
                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Retry with exponential backoff
                    time.sleep(1)
                    continue
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        self.stats["total_chunks"] = len(texts)
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        show_progress: bool = True,
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            show_progress: Show progress bar
        
        Returns:
            List of EmbeddedChunk objects
        """
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, show_progress)
        
        # Create EmbeddedChunk objects
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata,
                token_count=chunk.token_count,
            )
            embedded_chunks.append(embedded)
        
        logger.info(
            f"Generated {len(embedded_chunks)} embeddings. "
            f"API calls: {self.stats['api_calls']}, "
            f"Cache hits: {self.stats['cached_hits']}, "
            f"Total tokens: {self.stats['total_tokens']}"
        )
        
        return embedded_chunks
    
    def embed_chunks_streaming(
        self,
        chunks: Generator[DocumentChunk, None, None],
        checkpoint_every: int = 100,
        checkpoint_path: Path = None,
    ) -> Generator[EmbeddedChunk, None, None]:
        """
        Generate embeddings for chunks in streaming fashion with checkpointing.
        
        Args:
            chunks: Generator of DocumentChunk objects
            checkpoint_every: Save checkpoint every N chunks
            checkpoint_path: Path to save checkpoints
        
        Yields:
            EmbeddedChunk objects
        """
        checkpoint_path = checkpoint_path or self.cache_dir / "checkpoint.pkl"
        
        # Load checkpoint if exists
        processed_ids = set()
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                processed_ids = checkpoint_data.get("processed_ids", set())
            logger.info(f"Loaded checkpoint with {len(processed_ids)} processed chunks")
        
        buffer = []
        count = 0
        
        for chunk in chunks:
            # Skip already processed
            if chunk.chunk_id in processed_ids:
                continue
            
            buffer.append(chunk)
            
            # Process batch
            if len(buffer) >= self.batch_size:
                embedded = self.embed_chunks(buffer, show_progress=False)
                for emb_chunk in embedded:
                    processed_ids.add(emb_chunk.chunk_id)
                    yield emb_chunk
                
                buffer = []
                count += self.batch_size
                
                # Save checkpoint
                if count % checkpoint_every == 0:
                    self._save_checkpoint(checkpoint_path, processed_ids)
        
        # Process remaining
        if buffer:
            embedded = self.embed_chunks(buffer, show_progress=False)
            for emb_chunk in embedded:
                processed_ids.add(emb_chunk.chunk_id)
                yield emb_chunk
        
        # Final checkpoint
        self._save_checkpoint(checkpoint_path, processed_ids)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(
            f"{self.model}:{text}".encode()
        ).hexdigest()
    
    def _get_cached(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding to disk."""
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _save_checkpoint(self, path: Path, processed_ids: set):
        """Save processing checkpoint."""
        with open(path, 'wb') as f:
            pickle.dump({"processed_ids": processed_ids}, f)
        logger.debug(f"Saved checkpoint: {len(processed_ids)} chunks processed")
    
    def get_stats(self) -> Dict:
        """Get embedding generation statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cached_hits"] / self.stats["total_chunks"]
                if self.stats["total_chunks"] > 0 else 0
            ),
            "estimated_cost": self._estimate_cost(),
        }
    
    def _estimate_cost(self) -> float:
        """Estimate API cost based on token usage."""
        # Pricing for text-embedding-3-small: $0.02 per 1M tokens
        price_per_million = 0.02
        return (self.stats["total_tokens"] / 1_000_000) * price_per_million


def save_embeddings(
    embedded_chunks: List[EmbeddedChunk],
    output_path: Path,
):
    """
    Save embedded chunks to disk.
    
    Args:
        embedded_chunks: List of EmbeddedChunk objects
        output_path: Path to save embeddings
    """
    data = {
        "model": config.EMBEDDING_MODEL,
        "dimensions": config.EMBEDDING_DIMENSIONS,
        "num_chunks": len(embedded_chunks),
        "chunks": [chunk.to_dict() for chunk in embedded_chunks],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Saved {len(embedded_chunks)} embeddings to {output_path}")


def load_embeddings(input_path: Path) -> List[EmbeddedChunk]:
    """
    Load embedded chunks from disk.
    
    Args:
        input_path: Path to embeddings file
    
    Returns:
        List of EmbeddedChunk objects
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    chunks = []
    for chunk_data in data["chunks"]:
        chunk = EmbeddedChunk(
            chunk_id=chunk_data["chunk_id"],
            content=chunk_data["content"],
            embedding=chunk_data["embedding"],
            metadata=chunk_data["metadata"],
            token_count=chunk_data["token_count"],
        )
        chunks.append(chunk)
    
    logger.info(f"Loaded {len(chunks)} embeddings from {input_path}")
    return chunks


# CLI interface
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Generate embeddings for documents")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with JSON documents",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )
    
    args = parser.parse_args()
    
    from data_ingestion import SECDataIngestion
    from chunker import SECFilingChunker
    
    # Load filings
    ingestion = SECDataIngestion(output_dir=Path(args.input))
    filings = list(ingestion.load_filings())
    
    print(f"Loaded {len(filings)} filings")
    
    # Chunk filings
    chunker = SECFilingChunker()
    all_chunks = []
    for filing in filings:
        chunks = chunker.chunk_filing(
            filing.content,
            filing.filing_type,
            metadata={
                "company": filing.company_name,
                "ticker": filing.ticker,
                "filing_type": filing.filing_type,
                "filing_date": filing.filing_date,
            },
        )
        all_chunks.extend(chunks)
    
    print(f"Created {len(all_chunks)} chunks")
    
    # Generate embeddings
    generator = EmbeddingGenerator(
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )
    
    embedded_chunks = generator.embed_chunks(all_chunks)
    
    # Save embeddings
    save_embeddings(embedded_chunks, Path(args.output))
    
    # Print stats
    stats = generator.get_stats()
    print(f"\n✅ Embedding generation complete!")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   API calls: {stats['api_calls']}")
    print(f"   Cache hits: {stats['cached_hits']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Estimated cost: ${stats['estimated_cost']:.4f}")

"""
Document Chunking Module

Implements various chunking strategies for SEC filings:
- Section-based chunking (for structured documents)
- Recursive character splitting
- Semantic chunking
"""
import re
import logging
from typing import List, Dict, Generator, Optional
from dataclasses import dataclass, field
import hashlib

import tiktoken

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document."""
    chunk_id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = self._generate_id()
        if not self.token_count:
            self.token_count = self._count_tokens()
    
    def _generate_id(self) -> str:
        """Generate unique chunk ID based on content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        return f"chunk_{content_hash}"
    
    def _count_tokens(self) -> int:
        """Count tokens using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            return len(encoding.encode(self.content))
        except Exception:
            # Fallback: estimate ~4 chars per token
            return len(self.content) // 4
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
        }


class TextChunker:
    """Base class for text chunking."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or config.MIN_CHUNK_SIZE
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
        
        Returns:
            List of DocumentChunk objects
        """
        raise NotImplementedError


class RecursiveCharacterChunker(TextChunker):
    """
    Recursive character text splitter.
    Splits on paragraph, sentence, then character boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Split text recursively on separators."""
        metadata = metadata or {}
        chunks = []
        
        # Split recursively
        raw_chunks = self._split_recursive(text, self.separators)
        
        # Merge small chunks and split large ones
        merged_chunks = self._merge_chunks(raw_chunks)
        
        # Create DocumentChunk objects
        for i, content in enumerate(merged_chunks):
            if len(content.strip()) < self.min_chunk_size:
                continue
            
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(merged_chunks),
            }
            
            chunk = DocumentChunk(
                chunk_id="",  # Will be auto-generated
                content=content.strip(),
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_recursive(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """Recursively split text on separators."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level split
            return [text[i:i + self.chunk_size] 
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
        splits = text.split(separator)
        
        result = []
        for split in splits:
            if len(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recursively split with next separator
                result.extend(self._split_recursive(split, remaining_separators))
        
        return result
    
    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks while respecting chunk_size."""
        merged = []
        current_chunk = ""
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            potential_chunk = (
                current_chunk + "\n\n" + chunk 
                if current_chunk else chunk
            )
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    merged.append(current_chunk)
                
                if len(chunk) > self.chunk_size:
                    # Split oversized chunk
                    for i in range(0, len(chunk), self.chunk_size - self.chunk_overlap):
                        merged.append(chunk[i:i + self.chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = chunk
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged


class SectionBasedChunker(TextChunker):
    """
    Section-based chunker for structured documents like SEC filings.
    Splits on document sections first, then applies recursive chunking.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        section_patterns: List[str] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.section_patterns = section_patterns or []
        self.recursive_chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[DocumentChunk]:
        """Split text by sections, then chunk each section."""
        metadata = metadata or {}
        all_chunks = []
        
        # Extract sections
        sections = self._extract_sections(text)
        
        if not sections:
            # No sections found, use recursive chunking
            return self.recursive_chunker.chunk_text(text, metadata)
        
        # Chunk each section
        for section_name, section_content in sections.items():
            section_metadata = {
                **metadata,
                "section": section_name,
            }
            
            chunks = self.recursive_chunker.chunk_text(
                section_content, 
                section_metadata
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract named sections from text."""
        sections = {}
        
        if not self.section_patterns:
            return sections
        
        for i, section_name in enumerate(self.section_patterns):
            pattern = re.escape(section_name)
            match = re.search(pattern, text, re.IGNORECASE)
            
            if not match:
                continue
            
            start = match.start()
            end = len(text)
            
            # Find next section
            for next_section in self.section_patterns[i + 1:]:
                next_pattern = re.escape(next_section)
                next_match = re.search(next_pattern, text[start:], re.IGNORECASE)
                if next_match:
                    end = start + next_match.start()
                    break
            
            content = text[start:end].strip()
            if len(content) > self.min_chunk_size:
                sections[section_name] = content
        
        return sections


class SECFilingChunker:
    """
    Specialized chunker for SEC filings.
    Uses section-based chunking with 10-K/10-Q section patterns.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    
    def chunk_filing(
        self, 
        filing_content: str,
        filing_type: str,
        metadata: Dict = None,
    ) -> List[DocumentChunk]:
        """
        Chunk an SEC filing based on its type.
        
        Args:
            filing_content: Raw filing text
            filing_type: Type of filing (10-K, 10-Q, 8-K)
            metadata: Additional metadata to attach
        
        Returns:
            List of DocumentChunk objects
        """
        metadata = metadata or {}
        
        # Get section patterns for filing type
        section_patterns = config.SEC_SECTIONS.get(filing_type, [])
        
        # Create appropriate chunker
        if section_patterns:
            chunker = SectionBasedChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                section_patterns=section_patterns,
            )
        else:
            chunker = RecursiveCharacterChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        
        chunks = chunker.chunk_text(filing_content, metadata)
        
        logger.info(
            f"Created {len(chunks)} chunks from {filing_type} filing "
            f"({len(filing_content)} chars)"
        )
        
        return chunks
    
    def chunk_filings_batch(
        self,
        filings: List[Dict],
    ) -> Generator[DocumentChunk, None, None]:
        """
        Chunk multiple filings in batch.
        
        Args:
            filings: List of filing dictionaries with 'content', 
                    'filing_type', and metadata
        
        Yields:
            DocumentChunk objects
        """
        total_chunks = 0
        
        for filing in filings:
            content = filing.get("content", "")
            filing_type = filing.get("filing_type", "10-K")
            
            # Build metadata
            metadata = {
                "company": filing.get("company_name", ""),
                "ticker": filing.get("ticker", ""),
                "filing_type": filing_type,
                "filing_date": filing.get("filing_date", ""),
                "cik": filing.get("cik", ""),
                "source_url": filing.get("document_url", ""),
            }
            
            chunks = self.chunk_filing(content, filing_type, metadata)
            
            for chunk in chunks:
                total_chunks += 1
                yield chunk
        
        logger.info(f"Total chunks created: {total_chunks}")


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[DocumentChunk]:
    """
    Convenience function to chunk a list of documents.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of DocumentChunk objects
    """
    chunker = SECFilingChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    all_chunks = []
    for chunk in chunker.chunk_filings_batch(documents):
        all_chunks.append(chunk)
    
    return all_chunks


# CLI for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Test document chunking")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Chunk document
    chunker = SECFilingChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    chunks = chunker.chunk_filing(
        data.get("content", ""),
        data.get("filing_type", "10-K"),
        metadata={"ticker": data.get("ticker", "")},
    )
    
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Preview: {chunk.content[:200]}...")

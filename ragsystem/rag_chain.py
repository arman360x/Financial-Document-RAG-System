"""
RAG Chain Module

Implements the complete RAG pipeline:
1. Query processing
2. Context retrieval
3. Prompt construction
4. LLM generation
5. Source citation
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json

from openai import OpenAI

import config
from retriever import Retriever, RetrievalResult, get_retriever
from vector_store import get_vector_store

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[Dict] = field(default_factory=list)
    query: str = ""
    context_used: str = ""
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        source_list = "\n".join(
            f"  - {s['source']}" for s in self.sources
        )
        return f"""Answer: {self.answer}

Sources:
{source_list}

Confidence: {self.confidence:.0%}"""


class RAGChain:
    """
    Complete RAG pipeline for question answering.
    """
    
    def __init__(
        self,
        retriever: Retriever = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.retriever = retriever or get_retriever(
            vector_store=get_vector_store()
        )
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def query(
        self,
        question: str,
        filters: Dict = None,
        include_sources: bool = True,
        max_context_tokens: int = 4000,
    ) -> RAGResponse:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User question
            filters: Metadata filters for retrieval
            include_sources: Include source citations in response
            max_context_tokens: Maximum tokens for context
        
        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Retrieve relevant context
        context, results = self.retriever.retrieve_with_context(
            query=question,
            max_tokens=max_context_tokens,
            filters=filters,
        )
        
        if not context:
            return RAGResponse(
                answer="I couldn't find relevant information in the SEC filings to answer your question.",
                sources=[],
                query=question,
                confidence=0.0,
            )
        
        # Step 2: Construct prompt
        prompt = self._construct_prompt(question, context, include_sources)
        
        # Step 3: Generate response
        answer, confidence = self._generate_response(prompt)
        
        # Step 4: Extract sources
        sources = self._extract_sources(results) if include_sources else []
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
            confidence=confidence,
            metadata={
                "model": self.model,
                "num_sources": len(results),
                "context_tokens": len(context) // 4,
            }
        )
    
    def _construct_prompt(
        self,
        question: str,
        context: str,
        include_sources: bool,
    ) -> List[Dict]:
        """Construct chat messages for the LLM."""
        
        system_message = config.SYSTEM_PROMPT.format(context=context)
        
        source_instruction = ""
        if include_sources:
            source_instruction = """

When citing information, reference the source in brackets like [Company Name, Filing Type, Date].
"""
        
        user_message = f"""{config.RAG_PROMPT_TEMPLATE.format(question=question)}
{source_instruction}"""
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    
    def _generate_response(self, messages: List[Dict]) -> Tuple[str, float]:
        """Generate response using LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Estimate confidence based on response characteristics
            confidence = self._estimate_confidence(answer)
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I encountered an error while generating the response.", 0.0
    
    def _estimate_confidence(self, answer: str) -> float:
        """
        Estimate confidence based on response characteristics.
        
        Factors:
        - Presence of hedging language
        - Specificity of answer
        - Citation presence
        """
        confidence = 0.8  # Base confidence
        
        # Check for hedging language
        hedging_phrases = [
            "i'm not sure",
            "it's unclear",
            "may not",
            "might not",
            "couldn't find",
            "no information",
            "unable to",
        ]
        
        answer_lower = answer.lower()
        for phrase in hedging_phrases:
            if phrase in answer_lower:
                confidence -= 0.15
        
        # Check for specific citations
        if "[" in answer and "]" in answer:
            confidence += 0.1
        
        # Check for numbers (specificity)
        if any(char.isdigit() for char in answer):
            confidence += 0.05
        
        return max(min(confidence, 1.0), 0.0)
    
    def _extract_sources(self, results: List[RetrievalResult]) -> List[Dict]:
        """Extract source information from retrieval results."""
        sources = []
        seen_sources = set()
        
        for result in results:
            metadata = result.chunk.metadata
            
            # Create unique source identifier
            source_id = (
                metadata.get("ticker", ""),
                metadata.get("filing_type", ""),
                metadata.get("filing_date", ""),
            )
            
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            
            source_info = {
                "source": self._format_source(metadata),
                "ticker": metadata.get("ticker", ""),
                "filing_type": metadata.get("filing_type", ""),
                "filing_date": metadata.get("filing_date", ""),
                "section": metadata.get("section", ""),
                "url": metadata.get("source_url", ""),
                "relevance_score": result.final_score,
            }
            sources.append(source_info)
        
        return sources
    
    def _format_source(self, metadata: Dict) -> str:
        """Format metadata as readable source string."""
        parts = []
        
        if metadata.get("company"):
            parts.append(metadata["company"])
        elif metadata.get("ticker"):
            parts.append(metadata["ticker"])
        
        if metadata.get("filing_type"):
            parts.append(metadata["filing_type"])
        
        if metadata.get("filing_date"):
            parts.append(metadata["filing_date"])
        
        return ", ".join(parts) if parts else "Unknown source"
    
    def query_with_followup(
        self,
        question: str,
        conversation_history: List[Dict] = None,
        **kwargs,
    ) -> RAGResponse:
        """
        Query with conversation history for follow-up questions.
        
        Args:
            question: Current question
            conversation_history: List of previous Q&A pairs
            **kwargs: Additional arguments for query()
        
        Returns:
            RAGResponse
        """
        # Reformulate question with context
        if conversation_history:
            question = self._reformulate_question(question, conversation_history)
        
        return self.query(question, **kwargs)
    
    def _reformulate_question(
        self,
        question: str,
        history: List[Dict],
    ) -> str:
        """Reformulate question to be standalone using conversation history."""
        history_text = "\n".join(
            f"Q: {h['question']}\nA: {h['answer'][:200]}..."
            for h in history[-3:]  # Last 3 exchanges
        )
        
        prompt = f"""Given the conversation history, reformulate the follow-up question to be a standalone question.

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Question reformulation failed: {e}")
            return question


class StreamingRAGChain(RAGChain):
    """
    RAG chain with streaming response support.
    """
    
    def query_stream(
        self,
        question: str,
        filters: Dict = None,
        max_context_tokens: int = 4000,
    ):
        """
        Process query with streaming response.
        
        Yields:
            Chunks of the response as they're generated
        """
        # Retrieve context
        context, results = self.retriever.retrieve_with_context(
            query=question,
            max_tokens=max_context_tokens,
            filters=filters,
        )
        
        if not context:
            yield "I couldn't find relevant information to answer your question."
            return
        
        # Construct prompt
        messages = self._construct_prompt(question, context, include_sources=True)
        
        # Stream response
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\nError: {str(e)}"


class GroundedRAGChain(RAGChain):
    """
    RAG chain with enhanced grounding and hallucination detection.
    """
    
    def query(self, question: str, **kwargs) -> RAGResponse:
        """Query with hallucination check."""
        # Get initial response
        response = super().query(question, **kwargs)
        
        # Verify claims in the response
        verified_answer, confidence_adjustment = self._verify_claims(
            response.answer,
            response.context_used,
        )
        
        response.answer = verified_answer
        response.confidence = max(0, response.confidence + confidence_adjustment)
        
        return response
    
    def _verify_claims(
        self,
        answer: str,
        context: str,
    ) -> Tuple[str, float]:
        """
        Verify claims in the answer against the context.
        
        Returns:
            Tuple of (verified_answer, confidence_adjustment)
        """
        prompt = f"""Analyze the following answer and verify each claim against the provided context.
If any claims are not supported by the context, flag them.

Context:
{context[:3000]}

Answer to verify:
{answer}

For each unsupported claim, wrap it in [UNVERIFIED: claim].
Return the modified answer, or the original if all claims are supported.

Verified answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=0,
            )
            
            verified = response.choices[0].message.content.strip()
            
            # Calculate confidence adjustment
            unverified_count = verified.count("[UNVERIFIED:")
            confidence_adjustment = -0.1 * unverified_count
            
            # Clean up the answer
            verified = verified.replace("[UNVERIFIED:", "[Note: Could not verify -")
            
            return verified, confidence_adjustment
            
        except Exception as e:
            logger.warning(f"Claim verification failed: {e}")
            return answer, 0.0


# Factory function
def get_rag_chain(
    chain_type: str = "standard",
    **kwargs,
) -> RAGChain:
    """
    Get RAG chain instance.
    
    Args:
        chain_type: "standard", "streaming", or "grounded"
    
    Returns:
        RAGChain instance
    """
    if chain_type == "streaming":
        return StreamingRAGChain(**kwargs)
    elif chain_type == "grounded":
        return GroundedRAGChain(**kwargs)
    return RAGChain(**kwargs)


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG chain")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chain", choices=["standard", "streaming", "grounded"])
    parser.add_argument("--ticker", type=str, help="Filter by ticker")
    parser.add_argument("--filing-type", type=str, help="Filter by filing type")
    
    args = parser.parse_args()
    
    # Build filters
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.filing_type:
        filters["filing_type"] = args.filing_type
    
    # Initialize chain
    chain = get_rag_chain(chain_type=args.chain or "standard")
    
    # Query
    if args.chain == "streaming":
        print(f"\n🔍 Query: {args.query}\n")
        print("📝 Answer: ", end="", flush=True)
        for chunk in chain.query_stream(
            args.query,
            filters=filters if filters else None,
        ):
            print(chunk, end="", flush=True)
        print()
    else:
        response = chain.query(
            args.query,
            filters=filters if filters else None,
        )
        
        print(f"\n🔍 Query: {args.query}")
        print(f"\n📝 Answer:\n{response.answer}")
        print(f"\n📊 Confidence: {response.confidence:.0%}")
        print(f"\n📚 Sources:")
        for source in response.sources:
            print(f"   - {source['source']} (relevance: {source['relevance_score']:.2f})")

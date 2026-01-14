"""
Evaluation Module for RAG System

Implements metrics for evaluating RAG performance:
- Retrieval accuracy
- Answer relevance
- Faithfulness (groundedness)
- Hallucination detection
- Latency measurement
"""
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from rag_chain import RAGChain, RAGResponse, get_rag_chain
from retriever import get_retriever
from vector_store import get_vector_store
from embedder import EmbeddingGenerator

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    """Evaluation question with expected answer components."""
    question: str
    expected_topics: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    company: str = ""
    filing_type: str = ""


@dataclass
class EvalResult:
    """Result from evaluating a single question."""
    question: str
    answer: str
    retrieval_score: float
    relevance_score: float
    faithfulness_score: float
    hallucination_score: float
    latency_ms: float
    sources_found: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "retrieval": 0.25,
            "relevance": 0.30,
            "faithfulness": 0.30,
            "hallucination": 0.15,
        }
        return (
            weights["retrieval"] * self.retrieval_score +
            weights["relevance"] * self.relevance_score +
            weights["faithfulness"] * self.faithfulness_score +
            weights["hallucination"] * (1 - self.hallucination_score)  # Inverse
        )
    
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "scores": {
                "retrieval": self.retrieval_score,
                "relevance": self.relevance_score,
                "faithfulness": self.faithfulness_score,
                "hallucination": self.hallucination_score,
                "overall": self.overall_score,
            },
            "latency_ms": self.latency_ms,
            "sources_found": self.sources_found,
            "details": self.details,
        }


class RAGEvaluator:
    """
    Evaluates RAG system performance on various metrics.
    """
    
    def __init__(self, rag_chain: RAGChain = None):
        self.rag_chain = rag_chain or get_rag_chain()
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_generator = EmbeddingGenerator()
    
    def evaluate_question(
        self,
        eval_question: EvalQuestion,
    ) -> EvalResult:
        """
        Evaluate a single question.
        
        Args:
            eval_question: EvalQuestion object
        
        Returns:
            EvalResult with all metrics
        """
        # Build filters
        filters = {}
        if eval_question.company:
            filters["ticker"] = eval_question.company
        if eval_question.filing_type:
            filters["filing_type"] = eval_question.filing_type
        
        # Measure latency
        start_time = time.time()
        
        try:
            response = self.rag_chain.query(
                eval_question.question,
                filters=filters if filters else None,
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return EvalResult(
                question=eval_question.question,
                answer=f"Error: {str(e)}",
                retrieval_score=0,
                relevance_score=0,
                faithfulness_score=0,
                hallucination_score=1,
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate metrics
        retrieval_score = self._evaluate_retrieval(
            eval_question.expected_sources,
            [s["filing_type"] for s in response.sources],
        )
        
        relevance_score = self._evaluate_relevance(
            eval_question.question,
            response.answer,
        )
        
        faithfulness_score = self._evaluate_faithfulness(
            response.answer,
            response.context_used,
        )
        
        hallucination_score = self._detect_hallucinations(
            response.answer,
            response.context_used,
        )
        
        return EvalResult(
            question=eval_question.question,
            answer=response.answer,
            retrieval_score=retrieval_score,
            relevance_score=relevance_score,
            faithfulness_score=faithfulness_score,
            hallucination_score=hallucination_score,
            latency_ms=latency_ms,
            sources_found=[s["source"] for s in response.sources],
            details={
                "confidence": response.confidence,
                "num_sources": len(response.sources),
            },
        )
    
    def _evaluate_retrieval(
        self,
        expected_sources: List[str],
        retrieved_sources: List[str],
    ) -> float:
        """
        Evaluate retrieval accuracy.
        
        Measures if expected source types were retrieved.
        """
        if not expected_sources:
            return 1.0 if retrieved_sources else 0.0
        
        matches = sum(
            1 for exp in expected_sources
            if any(exp.lower() in ret.lower() for ret in retrieved_sources)
        )
        
        return matches / len(expected_sources)
    
    def _evaluate_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Evaluate answer relevance using LLM.
        """
        prompt = f"""Rate the relevance of the answer to the question on a scale of 0-10.
Consider:
- Does the answer address the question directly?
- Is the information provided useful for answering the question?
- Does the answer stay on topic?

Only respond with a single number.

Question: {question}

Answer: {answer}

Relevance score (0-10):"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            
            score = float(response.choices[0].message.content.strip())
            return min(max(score / 10, 0), 1)
            
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return 0.5
    
    def _evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Evaluate faithfulness/groundedness of answer to context.
        
        Measures if claims in answer are supported by context.
        """
        if not context:
            return 0.0
        
        prompt = f"""Analyze the answer and determine what percentage of claims are supported by the context.

Context:
{context[:3000]}

Answer:
{answer}

For each claim in the answer, determine if it's:
1. Directly supported by the context
2. Implied by the context
3. Not supported by the context

Respond with only a number from 0-100 representing the percentage of claims that are supported (directly or implied).

Support percentage:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            
            score = float(response.choices[0].message.content.strip().replace("%", ""))
            return min(max(score / 100, 0), 1)
            
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return 0.5
    
    def _detect_hallucinations(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Detect hallucinations in the answer.
        
        Returns a score from 0 (no hallucinations) to 1 (severe hallucinations).
        """
        if not context:
            return 0.5
        
        prompt = f"""Identify any factual claims in the answer that are NOT supported by the context.

Context:
{context[:3000]}

Answer:
{answer}

List each unsupported claim, or say "NONE" if all claims are supported.
Then rate the severity of hallucinations from 0-10 (0 = no hallucinations, 10 = severe hallucinations).

Format:
Unsupported claims: [list or NONE]
Hallucination severity: [0-10]"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract severity score
            if "severity:" in content.lower():
                severity_part = content.lower().split("severity:")[-1].strip()
                score = float(severity_part.split()[0])
                return min(max(score / 10, 0), 1)
            
            # If "NONE" in response, low hallucination
            if "none" in content.lower():
                return 0.1
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Hallucination detection failed: {e}")
            return 0.5
    
    def evaluate_batch(
        self,
        questions: List[EvalQuestion],
        show_progress: bool = True,
    ) -> List[EvalResult]:
        """
        Evaluate a batch of questions.
        
        Args:
            questions: List of EvalQuestion objects
            show_progress: Show progress indicator
        
        Returns:
            List of EvalResult objects
        """
        results = []
        
        for i, question in enumerate(questions):
            if show_progress:
                logger.info(f"Evaluating {i + 1}/{len(questions)}: {question.question[:50]}...")
            
            result = self.evaluate_question(question)
            results.append(result)
        
        return results
    
    def generate_report(
        self,
        results: List[EvalResult],
        output_path: Path = None,
    ) -> Dict:
        """
        Generate evaluation report.
        
        Args:
            results: List of EvalResult objects
            output_path: Optional path to save report
        
        Returns:
            Report dictionary
        """
        # Calculate aggregate metrics
        num_questions = len(results)
        
        avg_metrics = {
            "retrieval": np.mean([r.retrieval_score for r in results]),
            "relevance": np.mean([r.relevance_score for r in results]),
            "faithfulness": np.mean([r.faithfulness_score for r in results]),
            "hallucination": np.mean([r.hallucination_score for r in results]),
            "overall": np.mean([r.overall_score for r in results]),
            "latency_ms": np.mean([r.latency_ms for r in results]),
        }
        
        # Identify issues
        low_performers = [
            r for r in results if r.overall_score < 0.5
        ]
        
        high_hallucination = [
            r for r in results if r.hallucination_score > 0.5
        ]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": num_questions,
            "average_metrics": avg_metrics,
            "metric_thresholds": {
                "retrieval_target": 0.85,
                "relevance_target": 0.80,
                "faithfulness_target": 0.90,
                "hallucination_max": 0.05,
                "latency_max_ms": 3000,
            },
            "pass_fail": {
                "retrieval_pass": avg_metrics["retrieval"] >= 0.85,
                "relevance_pass": avg_metrics["relevance"] >= 0.80,
                "faithfulness_pass": avg_metrics["faithfulness"] >= 0.90,
                "hallucination_pass": avg_metrics["hallucination"] <= 0.05,
                "latency_pass": avg_metrics["latency_ms"] <= 3000,
            },
            "issues": {
                "low_performers": len(low_performers),
                "high_hallucination": len(high_hallucination),
            },
            "detailed_results": [r.to_dict() for r in results],
        }
        
        # Calculate overall pass
        passes = list(report["pass_fail"].values())
        report["overall_pass"] = all(passes)
        report["pass_rate"] = sum(passes) / len(passes)
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report


def get_default_eval_questions() -> List[EvalQuestion]:
    """Get default evaluation questions."""
    return [
        EvalQuestion(
            question="What are Apple's main risk factors?",
            expected_topics=["risk", "competition", "supply chain"],
            expected_sources=["10-K"],
            company="AAPL",
        ),
        EvalQuestion(
            question="What was Microsoft's total revenue in the latest fiscal year?",
            expected_topics=["revenue", "financial"],
            expected_sources=["10-K"],
            company="MSFT",
        ),
        EvalQuestion(
            question="What legal proceedings is Tesla currently involved in?",
            expected_topics=["legal", "litigation", "proceedings"],
            expected_sources=["10-K", "10-Q"],
            company="TSLA",
        ),
        EvalQuestion(
            question="Describe Amazon's business segments.",
            expected_topics=["business", "segments", "operations"],
            expected_sources=["10-K"],
            company="AMZN",
        ),
        EvalQuestion(
            question="What are NVIDIA's competitive advantages?",
            expected_topics=["competition", "advantage", "market"],
            expected_sources=["10-K"],
            company="NVDA",
        ),
    ]


def print_report(report: Dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("📊 RAG EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n📅 Timestamp: {report['timestamp']}")
    print(f"📝 Questions evaluated: {report['num_questions']}")
    
    print("\n📈 AVERAGE METRICS:")
    print("-" * 40)
    metrics = report["average_metrics"]
    print(f"  Retrieval Accuracy:  {metrics['retrieval']:.1%}")
    print(f"  Answer Relevance:    {metrics['relevance']:.1%}")
    print(f"  Faithfulness:        {metrics['faithfulness']:.1%}")
    print(f"  Hallucination Rate:  {metrics['hallucination']:.1%}")
    print(f"  Overall Score:       {metrics['overall']:.1%}")
    print(f"  Avg Latency:         {metrics['latency_ms']:.0f}ms")
    
    print("\n✅ PASS/FAIL STATUS:")
    print("-" * 40)
    for metric, passed in report["pass_fail"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {metric}: {status}")
    
    print("\n🎯 OVERALL:")
    print("-" * 40)
    overall_status = "✅ PASS" if report["overall_pass"] else "❌ FAIL"
    print(f"  Status: {overall_status}")
    print(f"  Pass Rate: {report['pass_rate']:.1%}")
    
    if report["issues"]["low_performers"] > 0:
        print(f"\n⚠️  {report['issues']['low_performers']} questions had low scores")
    if report["issues"]["high_hallucination"] > 0:
        print(f"⚠️  {report['issues']['high_hallucination']} questions had high hallucination")
    
    print("\n" + "=" * 60)


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--test-set",
        type=str,
        help="Path to JSON file with test questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Output path for report",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with fewer questions",
    )
    
    args = parser.parse_args()
    
    # Load questions
    if args.test_set:
        with open(args.test_set, 'r') as f:
            data = json.load(f)
        questions = [EvalQuestion(**q) for q in data]
    else:
        questions = get_default_eval_questions()
    
    if args.quick:
        questions = questions[:3]
    
    print(f"\n🔬 Running evaluation on {len(questions)} questions...")
    
    # Run evaluation
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_batch(questions)
    
    # Generate report
    report = evaluator.generate_report(
        results,
        output_path=Path(args.output),
    )
    
    # Print results
    print_report(report)

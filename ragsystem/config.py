"""
Configuration settings for SEC RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Path Configuration
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================================
# OpenAI Configuration
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"  # Cost-effective, 1536 dimensions
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 100  # Process 100 chunks per API call

# LLM settings
LLM_MODEL = "gpt-4o"  # Or "gpt-4o-mini" for cost savings
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.1  # Low temperature for factual responses

# ============================================================================
# Chunking Configuration
# ============================================================================
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

# Section-based chunking for SEC filings
SEC_SECTIONS = {
    "10-K": [
        "Item 1. Business",
        "Item 1A. Risk Factors",
        "Item 1B. Unresolved Staff Comments",
        "Item 2. Properties",
        "Item 3. Legal Proceedings",
        "Item 4. Mine Safety Disclosures",
        "Item 5. Market for Registrant",
        "Item 6. Selected Financial Data",
        "Item 7. Management's Discussion and Analysis",
        "Item 7A. Quantitative and Qualitative Disclosures",
        "Item 8. Financial Statements",
        "Item 9. Changes in and Disagreements",
        "Item 9A. Controls and Procedures",
        "Item 9B. Other Information",
        "Item 10. Directors, Executive Officers",
        "Item 11. Executive Compensation",
        "Item 12. Security Ownership",
        "Item 13. Certain Relationships",
        "Item 14. Principal Accountant Fees",
        "Item 15. Exhibits",
    ],
    "10-Q": [
        "Part I - Financial Information",
        "Part II - Other Information",
    ],
}

# ============================================================================
# Vector Database Configuration
# ============================================================================
# PostgreSQL with pgvector
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/sec_rag"
)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_DB = os.getenv("POSTGRES_DB", "sec_rag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# Vector index settings
VECTOR_INDEX_TYPE = "hnsw"  # Options: "hnsw", "ivfflat"
HNSW_M = 16  # Max connections per node
HNSW_EF_CONSTRUCTION = 64  # Size of dynamic candidate list

# Pinecone (alternative)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "sec-filings")

# ============================================================================
# Retrieval Configuration
# ============================================================================
TOP_K = 5  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score
RERANK_ENABLED = True  # Enable reranking of results
HYBRID_SEARCH_ALPHA = 0.5  # Balance between semantic (1.0) and keyword (0.0)

# ============================================================================
# SEC EDGAR Configuration
# ============================================================================
SEC_API_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar"
SEC_USER_AGENT = os.getenv(
    "SEC_API_USER_AGENT", 
    "SEC-RAG-Demo contact@example.com"
)

# Rate limiting (SEC requires 10 requests/second max)
SEC_RATE_LIMIT = 10  # requests per second
SEC_REQUEST_DELAY = 0.1  # seconds between requests

# Default companies to download (by ticker)
DEFAULT_COMPANIES = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "TSLA",   # Tesla
    "META",   # Meta (Facebook)
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan Chase
    "V",      # Visa
    "WMT",    # Walmart
]

# CIK mappings for common companies
TICKER_TO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "TSLA": "0001318605",
    "META": "0001326801",
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "V": "0001403161",
    "WMT": "0000104169",
    "BRK-A": "0001067983",
    "JNJ": "0000200406",
    "UNH": "0000731766",
    "XOM": "0000034088",
    "PG": "0000080424",
}

# Filing types to download
FILING_TYPES = ["10-K", "10-Q", "8-K"]

# Years of data to download
DEFAULT_YEARS = 5

# ============================================================================
# RAG Chain Configuration
# ============================================================================
# System prompt for grounded responses
SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions based on SEC filings.

IMPORTANT RULES:
1. Only use information from the provided context (SEC filings)
2. Always cite your sources with the filing name, company, and date
3. If the context doesn't contain enough information, say so clearly
4. Never make up or hallucinate information
5. Be specific with numbers, dates, and facts from the filings

Context from SEC filings:
{context}
"""

RAG_PROMPT_TEMPLATE = """Based on the SEC filings provided, please answer the following question.

Question: {question}

Instructions:
- Provide a comprehensive answer based only on the context provided
- Include specific quotes and references where relevant
- If the information is not in the context, clearly state that
- Format your response clearly with key points highlighted
"""

# ============================================================================
# Evaluation Configuration
# ============================================================================
EVAL_METRICS = [
    "retrieval_accuracy",
    "answer_relevance",
    "faithfulness",  # Groundedness
    "hallucination_rate",
    "latency",
]

# Test questions for evaluation
EVAL_QUESTIONS = [
    {
        "question": "What are Apple's main risk factors?",
        "expected_sources": ["10-K"],
        "company": "AAPL",
    },
    {
        "question": "What was Microsoft's total revenue in the latest fiscal year?",
        "expected_sources": ["10-K"],
        "company": "MSFT",
    },
    {
        "question": "What legal proceedings is Tesla currently involved in?",
        "expected_sources": ["10-K", "10-Q"],
        "company": "TSLA",
    },
]

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# Streamlit Configuration
# ============================================================================
STREAMLIT_PAGE_TITLE = "SEC Filings RAG Intelligence"
STREAMLIT_PAGE_ICON = "📊"
STREAMLIT_LAYOUT = "wide"

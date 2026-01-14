"""
SEC EDGAR Data Ingestion Module

Downloads and parses SEC filings (10-K, 10-Q, 8-K) from the EDGAR database.
Handles rate limiting, caching, and HTML/XML parsing.
"""
import os
import re
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import html2text
from tqdm import tqdm

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """Represents a single SEC filing."""
    company_name: str
    ticker: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    document_url: str
    content: str = ""
    sections: Dict[str, str] = None
    
    def __post_init__(self):
        if self.sections is None:
            self.sections = {}
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SECEdgarClient:
    """Client for interacting with SEC EDGAR API."""
    
    def __init__(self, user_agent: str = None):
        self.base_url = config.SEC_API_BASE_URL
        self.archives_url = config.SEC_ARCHIVES_URL
        self.user_agent = user_agent or config.SEC_USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.last_request_time = 0
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
    
    def _rate_limit(self):
        """Enforce SEC rate limit of 10 requests/second."""
        elapsed = time.time() - self.last_request_time
        if elapsed < config.SEC_REQUEST_DELAY:
            time.sleep(config.SEC_REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make rate-limited request to SEC."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_company_filings(
        self, 
        cik: str, 
        filing_types: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[Dict]:
        """
        Get list of filings for a company.
        
        Args:
            cik: Company CIK number
            filing_types: List of filing types (e.g., ['10-K', '10-Q'])
            start_date: Filter filings after this date (YYYY-MM-DD)
            end_date: Filter filings before this date (YYYY-MM-DD)
        
        Returns:
            List of filing metadata dictionaries
        """
        filing_types = filing_types or config.FILING_TYPES
        
        # Normalize CIK (remove leading zeros for API, pad for URLs)
        cik_padded = cik.zfill(10)
        
        # Get company submissions
        url = f"{self.base_url}/submissions/CIK{cik_padded}.json"
        response = self._make_request(url)
        
        if not response:
            return []
        
        data = response.json()
        company_name = data.get("name", "Unknown")
        
        filings = []
        recent = data.get("filings", {}).get("recent", {})
        
        if not recent:
            return []
        
        # Parse filing data
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        
        for i, (form, date, accession, doc) in enumerate(
            zip(forms, dates, accessions, primary_docs)
        ):
            # Filter by filing type
            if form not in filing_types:
                continue
            
            # Filter by date
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
            
            # Build document URL
            accession_formatted = accession.replace("-", "")
            doc_url = (
                f"{self.archives_url}/data/{cik_padded}/"
                f"{accession_formatted}/{doc}"
            )
            
            filings.append({
                "company_name": company_name,
                "cik": cik_padded,
                "filing_type": form,
                "filing_date": date,
                "accession_number": accession,
                "document_url": doc_url,
            })
        
        logger.info(f"Found {len(filings)} filings for CIK {cik}")
        return filings
    
    def download_filing(self, filing_metadata: Dict) -> Optional[SECFiling]:
        """
        Download and parse a single SEC filing.
        
        Args:
            filing_metadata: Dictionary with filing metadata
        
        Returns:
            SECFiling object with parsed content
        """
        url = filing_metadata["document_url"]
        response = self._make_request(url)
        
        if not response:
            return None
        
        # Parse HTML content
        content = self._parse_filing_content(response.text)
        
        # Extract sections
        sections = self._extract_sections(
            content, 
            filing_metadata["filing_type"]
        )
        
        filing = SECFiling(
            company_name=filing_metadata["company_name"],
            ticker=self._get_ticker_for_cik(filing_metadata["cik"]),
            cik=filing_metadata["cik"],
            filing_type=filing_metadata["filing_type"],
            filing_date=filing_metadata["filing_date"],
            accession_number=filing_metadata["accession_number"],
            document_url=url,
            content=content,
            sections=sections,
        )
        
        return filing
    
    def _parse_filing_content(self, html: str) -> str:
        """Parse HTML filing to clean text."""
        # Remove XBRL tags
        html = re.sub(r'<ix:[^>]+>', '', html)
        html = re.sub(r'</ix:[^>]+>', '', html)
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'meta', 'link']):
            element.decompose()
        
        # Convert to markdown-like text
        text = self.html_converter.handle(str(soup))
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def _extract_sections(
        self, 
        content: str, 
        filing_type: str
    ) -> Dict[str, str]:
        """Extract named sections from filing."""
        sections = {}
        section_patterns = config.SEC_SECTIONS.get(filing_type, [])
        
        for i, section_name in enumerate(section_patterns):
            # Create regex pattern for section
            pattern = re.escape(section_name)
            
            # Find section start
            match = re.search(pattern, content, re.IGNORECASE)
            if not match:
                continue
            
            start = match.start()
            
            # Find section end (start of next section or end of document)
            end = len(content)
            for next_section in section_patterns[i + 1:]:
                next_pattern = re.escape(next_section)
                next_match = re.search(next_pattern, content[start:], re.IGNORECASE)
                if next_match:
                    end = start + next_match.start()
                    break
            
            section_content = content[start:end].strip()
            if len(section_content) > config.MIN_CHUNK_SIZE:
                sections[section_name] = section_content
        
        return sections
    
    def _get_ticker_for_cik(self, cik: str) -> str:
        """Get ticker symbol for CIK."""
        cik_normalized = cik.lstrip("0")
        for ticker, ticker_cik in config.TICKER_TO_CIK.items():
            if ticker_cik.lstrip("0") == cik_normalized:
                return ticker
        return "UNKNOWN"


class SECDataIngestion:
    """Main class for SEC data ingestion."""
    
    def __init__(self, output_dir: Path = None):
        self.client = SECEdgarClient()
        self.output_dir = output_dir or config.DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_company_filings(
        self,
        ticker: str,
        years: int = None,
        filing_types: List[str] = None,
    ) -> List[SECFiling]:
        """
        Download all filings for a company.
        
        Args:
            ticker: Company ticker symbol
            years: Number of years of data to download
            filing_types: Types of filings to download
        
        Returns:
            List of SECFiling objects
        """
        years = years or config.DEFAULT_YEARS
        filing_types = filing_types or config.FILING_TYPES
        
        # Get CIK from ticker
        cik = config.TICKER_TO_CIK.get(ticker.upper())
        if not cik:
            logger.error(f"Unknown ticker: {ticker}")
            return []
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (
            datetime.now() - timedelta(days=years * 365)
        ).strftime("%Y-%m-%d")
        
        logger.info(f"Downloading {ticker} filings from {start_date} to {end_date}")
        
        # Get filing list
        filings_metadata = self.client.get_company_filings(
            cik=cik,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Download each filing
        filings = []
        for metadata in tqdm(filings_metadata, desc=f"Downloading {ticker}"):
            filing = self.client.download_filing(metadata)
            if filing:
                filings.append(filing)
                self._save_filing(filing)
        
        logger.info(f"Downloaded {len(filings)} filings for {ticker}")
        return filings
    
    def download_multiple_companies(
        self,
        tickers: List[str] = None,
        years: int = None,
    ) -> Dict[str, List[SECFiling]]:
        """
        Download filings for multiple companies.
        
        Args:
            tickers: List of ticker symbols
            years: Number of years of data
        
        Returns:
            Dictionary mapping ticker to list of filings
        """
        tickers = tickers or config.DEFAULT_COMPANIES
        results = {}
        
        for ticker in tickers:
            filings = self.download_company_filings(ticker, years)
            results[ticker] = filings
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_filing(self, filing: SECFiling) -> Path:
        """Save filing to disk."""
        # Create company directory
        company_dir = self.output_dir / filing.ticker
        company_dir.mkdir(exist_ok=True)
        
        # Create filename
        filename = (
            f"{filing.filing_type}_{filing.filing_date}_"
            f"{filing.accession_number}.json"
        )
        filepath = company_dir / filename
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filing.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_summary(self, results: Dict[str, List[SECFiling]]) -> Path:
        """Save download summary."""
        summary = {
            "download_date": datetime.now().isoformat(),
            "companies": {},
        }
        
        total_filings = 0
        total_size = 0
        
        for ticker, filings in results.items():
            company_size = sum(len(f.content) for f in filings)
            summary["companies"][ticker] = {
                "num_filings": len(filings),
                "filing_types": list(set(f.filing_type for f in filings)),
                "date_range": {
                    "earliest": min(f.filing_date for f in filings) if filings else None,
                    "latest": max(f.filing_date for f in filings) if filings else None,
                },
                "total_size_bytes": company_size,
            }
            total_filings += len(filings)
            total_size += company_size
        
        summary["total_filings"] = total_filings
        summary["total_size_bytes"] = total_size
        summary["total_size_gb"] = round(total_size / (1024**3), 2)
        
        filepath = self.output_dir / "download_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(
            f"Downloaded {total_filings} filings, "
            f"total size: {summary['total_size_gb']} GB"
        )
        
        return filepath
    
    def load_filings(self, ticker: str = None) -> Generator[SECFiling, None, None]:
        """
        Load saved filings from disk.
        
        Args:
            ticker: Optional ticker to filter by
        
        Yields:
            SECFiling objects
        """
        if ticker:
            search_dirs = [self.output_dir / ticker]
        else:
            search_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        for company_dir in search_dirs:
            if not company_dir.exists():
                continue
            
            for filepath in company_dir.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    yield SECFiling(**data)
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")


def main():
    """Command-line interface for data ingestion."""
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR filings"
    )
    parser.add_argument(
        "--companies",
        type=str,
        default=None,
        help="Comma-separated list of tickers (e.g., AAPL,MSFT,GOOGL)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=config.DEFAULT_YEARS,
        help="Number of years of data to download"
    )
    parser.add_argument(
        "--filing-types",
        type=str,
        default=None,
        help="Comma-separated filing types (e.g., 10-K,10-Q)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for downloaded filings"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    tickers = (
        args.companies.split(",") if args.companies 
        else config.DEFAULT_COMPANIES
    )
    filing_types = (
        args.filing_types.split(",") if args.filing_types 
        else config.FILING_TYPES
    )
    output_dir = Path(args.output_dir) if args.output_dir else config.DATA_DIR
    
    # Run ingestion
    ingestion = SECDataIngestion(output_dir=output_dir)
    results = ingestion.download_multiple_companies(
        tickers=tickers,
        years=args.years,
    )
    
    print(f"\n✅ Download complete!")
    print(f"   Companies: {len(results)}")
    print(f"   Total filings: {sum(len(f) for f in results.values())}")
    print(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()

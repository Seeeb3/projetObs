import os
import re
from typing import Dict, Final, List, Optional, Pattern

import requests
from pydantic import BaseModel, Field

from tools.local_env import load_local_env_file


ADS_API_BASE_URL: Final[str] = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_UI_BASE_URL: Final[str] = "https://ui.adsabs.harvard.edu"
ADS_FULLTEXT_SOURCE_PRIORITY: Final[List[str]] = [
    "PUB_PDF",
    "EPRINT_PDF",
    "PUB_HTML",
    "EPRINT_HTML",
]
DOI_PREFIXES: Final[tuple[str, ...]] = (
    "https://doi.org/",
    "http://doi.org/",
    "doi:",
)
ARXIV_PREFIX_PATTERN: Final[Pattern[str]] = re.compile(r"^arxiv:(?P<arxiv_id>.+)$", re.IGNORECASE)
ARXIV_NEW_ID_PATTERN: Final[Pattern[str]] = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
ARXIV_OLD_ID_PATTERN: Final[Pattern[str]] = re.compile(
    r"^[a-z-]+(?:\.[A-Za-z]{2})?/\d{7}(?:v\d+)?$",
    re.IGNORECASE,
)


def normalize_doi_text(raw_doi: str | None) -> str:
    """Normalize a DOI-like string to a canonical lowercase value.

    Args:
        raw_doi: Raw DOI value when available.

    Returns:
        Canonical DOI string without URL prefix, or an empty string when missing.
    """

    if raw_doi is None:
        return ""

    normalized_doi = raw_doi.strip()
    if not normalized_doi:
        return ""

    lowered_doi = normalized_doi.lower()
    for prefix in DOI_PREFIXES:
        if lowered_doi.startswith(prefix):
            normalized_doi = normalized_doi[len(prefix) :]
            break

    return normalized_doi.strip().lower()


class ADSDoc(BaseModel):
    """Represents a document returned by NASA ADS API.
    
    Attributes:
        bibcode: The entry's bibcode.
        doi: List of DOIs associated with the bibcode.
    """
    bibcode: str
    doi: Optional[List[str]] = Field(default_factory=list)

class ADSResponse(BaseModel):
    """Represents the response structure from NASA ADS API."""
    docs: List[ADSDoc]


class ADSEsourcesDoc(BaseModel):
    """Represents a bibcode and the available ADS electronic sources.

    Attributes:
        bibcode: The entry's bibcode.
        esources: Available electronic source types for the record.
    """

    bibcode: str
    esources: List[str] = Field(default_factory=list)


class ADSEsourcesResponse(BaseModel):
    """Represents the response structure for esource lookups."""

    docs: List[ADSEsourcesDoc]


class ADSSourcesDoc(BaseModel):
    """Represents ADS source identifiers and electronic sources for one record.

    Attributes:
        bibcode: The entry's bibcode.
        esources: Available electronic source types for the record.
        identifier: Alternate identifiers such as DOI, bibcode, and arXiv id.
    """

    bibcode: str
    esources: List[str] = Field(default_factory=list)
    identifier: List[str] = Field(default_factory=list)


class ADSSourcesResponse(BaseModel):
    """Represents the ADS response structure for source enrichment lookups.

    Attributes:
        docs: Source-enrichment documents returned by ADS.
    """

    docs: List[ADSSourcesDoc]


class ADSAbstractDoc(BaseModel):
    """Represents a bibcode and abstract returned by NASA ADS API.

    Attributes:
        bibcode: The entry's bibcode.
        abstract: The abstract text when available.
    """

    bibcode: str
    abstract: Optional[str] = None


class ADSAbstractResponse(BaseModel):
    """Represents the response structure for abstract lookups."""

    docs: List[ADSAbstractDoc]


class ADSMetadataDoc(BaseModel):
    """Represents metadata fields returned by NASA ADS API.

    Attributes:
        bibcode: The entry's bibcode.
        title: List of titles when available.
        keyword: List of keywords when available.
    """

    bibcode: str
    title: List[str] = Field(default_factory=list)
    keyword: List[str] = Field(default_factory=list)


class ADSMetadataResponse(BaseModel):
    """Represents the response structure for metadata lookups."""

    docs: List[ADSMetadataDoc]


class ADSTitleAbstractDoc(BaseModel):
    """Represents title and abstract fields returned by NASA ADS API.

    Attributes:
        bibcode: The entry's bibcode.
        title: List of titles when available.
        abstract: Abstract text when available.
    """

    bibcode: str
    title: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None


class ADSTitleAbstractResponse(BaseModel):
    """Represents the response structure for title+abstract lookups."""

    docs: List[ADSTitleAbstractDoc]


class ADSAuthorDoc(BaseModel):
    """Represents author metadata returned by ADS.

    Attributes:
        bibcode: The entry's bibcode.
        author: Ordered author names when available.
    """

    bibcode: str
    author: List[str] = Field(default_factory=list)


class ADSAuthorResponse(BaseModel):
    """Represents the response structure for author lookups.

    Attributes:
        docs: Author metadata documents returned by ADS.
    """

    docs: List[ADSAuthorDoc]


class ADSArticleEnrichmentDoc(BaseModel):
    """Represents the ADS enrichment bundle returned for one bibcode.

    Attributes:
        bibcode: The entry's bibcode.
        doi: Ordered DOI candidates returned by ADS.
        abstract: Abstract text when available.
        author: Ordered author names when available.
        esources: Available electronic source types for the record.
        identifier: Alternate identifiers such as DOI and arXiv ids.
    """

    bibcode: str
    doi: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    author: List[str] = Field(default_factory=list)
    esources: List[str] = Field(default_factory=list)
    identifier: List[str] = Field(default_factory=list)


class ADSArticleEnrichmentResponse(BaseModel):
    """Represents the response structure for ADS enrichment bundle lookups.

    Attributes:
        docs: Enrichment documents returned by ADS.
    """

    docs: List[ADSArticleEnrichmentDoc]


class ADSArticleEnrichmentRecord(BaseModel):
    """Normalized ADS enrichment bundle used by the 124-row helio pipeline.

    Attributes:
        doi: First DOI string when available.
        abstract: Abstract text when available.
        authors: Ordered author names when available.
        available_esources: Normalized ADS electronic source types.
        arxiv_ids: Ordered arXiv identifiers discovered in ADS identifiers.
    """

    doi: Optional[str] = None
    abstract: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    available_esources: List[str] = Field(default_factory=list)
    arxiv_ids: List[str] = Field(default_factory=list)


class ADSCorpusEnrichmentDoc(BaseModel):
    """Represents the ADS enrichment bundle returned for one corpus bibcode.

    Attributes:
        bibcode: The entry's bibcode.
        title: Ordered title candidates returned by ADS.
        abstract: Abstract text when available.
        keyword: Ordered keywords when available.
        doi: Ordered DOI candidates returned by ADS.
        author: Ordered author names when available.
        identifier: Alternate identifiers such as DOI and arXiv ids.
    """

    bibcode: str
    title: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    keyword: List[str] = Field(default_factory=list)
    doi: List[str] = Field(default_factory=list)
    author: List[str] = Field(default_factory=list)
    identifier: List[str] = Field(default_factory=list)


class ADSCorpusEnrichmentResponse(BaseModel):
    """Represents the response structure for corpus enrichment lookups."""

    docs: List[ADSCorpusEnrichmentDoc]


class ADSCorpusEnrichmentRecord(BaseModel):
    """Normalized ADS corpus enrichment bundle used by the WIESP corpus pipeline.

    Attributes:
        title: First title string when available.
        keywords: Ordered keywords when available.
        abstract: Abstract text when available.
        doi: First DOI string when available.
        authors: Ordered author names when available.
        arxiv_ids: Ordered arXiv identifiers discovered from ADS identifiers.
    """

    title: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    arxiv_ids: List[str] = Field(default_factory=list)


class ADSDoiResolutionDoc(BaseModel):
    """Represents the ADS DOI-resolution bundle returned for one record.

    Attributes:
        bibcode: The entry's bibcode.
        doi: Ordered DOI candidates returned by ADS.
        title: Ordered title candidates returned by ADS.
        abstract: Abstract text when available.
        keyword: Ordered keywords when available.
        author: Ordered author names when available.
        identifier: Alternate identifiers such as DOI and arXiv ids.
    """

    bibcode: str
    doi: List[str] = Field(default_factory=list)
    title: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    keyword: List[str] = Field(default_factory=list)
    author: List[str] = Field(default_factory=list)
    identifier: List[str] = Field(default_factory=list)


class ADSDoiResolutionResponse(BaseModel):
    """Represents the response structure for DOI-resolution lookups."""

    docs: List[ADSDoiResolutionDoc]


class ADSDoiResolutionRecord(BaseModel):
    """Normalized ADS DOI-resolution record used by the Bapt DOI pipeline.

    Attributes:
        bibcode: Resolved ADS bibcode when available.
        doi: Preferred DOI returned by ADS when available.
        title: First title string when available.
        keywords: Ordered keywords when available.
        authors: Ordered author names when available.
        abstract: Abstract text when available.
        arxiv_ids: Ordered arXiv identifiers discovered from ADS identifiers.
    """

    bibcode: Optional[str] = None
    doi: Optional[str] = None
    title: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    arxiv_ids: List[str] = Field(default_factory=list)


class ADSFullMetadataDoc(BaseModel):
    """Represents title, abstract, and keyword fields returned by NASA ADS API.

    Attributes:
        bibcode: The entry's bibcode.
        title: List of titles when available.
        abstract: Abstract text when available.
        keyword: List of keywords when available.
    """

    bibcode: str
    title: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    keyword: List[str] = Field(default_factory=list)


class ADSFullMetadataResponse(BaseModel):
    """Represents the response structure for full metadata lookups."""

    docs: List[ADSFullMetadataDoc]


class ADSFullMetadataRecord(BaseModel):
    """Normalized ADS metadata used by downstream scripts.

    Attributes:
        title: First title string when available.
        abstract: Abstract text when available.
        keywords: Keyword list when available.
    """

    title: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class ADSGatewayLinksRecord(BaseModel):
    """Normalized ADS source links for one bibcode.

    Attributes:
        ads_abstract_url: Canonical ADS abstract page URL.
        available_esources: Electronic source types reported by ADS.
        gateway_urls: ADS gateway URLs keyed by esource type.
        pub_pdf_url: Gateway URL for the publisher PDF when available.
        eprint_pdf_url: Gateway URL for the e-print PDF when available.
        pub_html_url: Gateway URL for the publisher HTML page when available.
        eprint_html_url: Gateway URL for the e-print HTML page when available.
        best_fulltext_url: Best available full-text URL based on priority order.
        best_fulltext_source: Source type associated with ``best_fulltext_url``.
        arxiv_ids: arXiv identifiers discovered from ADS identifiers.
        arxiv_abs_urls: arXiv abstract URLs for discovered arXiv identifiers.
        arxiv_pdf_urls: arXiv PDF URLs for discovered arXiv identifiers.
        arxiv_eprint_urls: arXiv source-package URLs for discovered arXiv identifiers.
    """

    ads_abstract_url: str
    available_esources: List[str] = Field(default_factory=list)
    gateway_urls: Dict[str, str] = Field(default_factory=dict)
    pub_pdf_url: Optional[str] = None
    eprint_pdf_url: Optional[str] = None
    pub_html_url: Optional[str] = None
    eprint_html_url: Optional[str] = None
    best_fulltext_url: Optional[str] = None
    best_fulltext_source: Optional[str] = None
    arxiv_ids: List[str] = Field(default_factory=list)
    arxiv_abs_urls: List[str] = Field(default_factory=list)
    arxiv_pdf_urls: List[str] = Field(default_factory=list)
    arxiv_eprint_urls: List[str] = Field(default_factory=list)


def extract_arxiv_ids_from_identifiers(identifiers: List[str]) -> List[str]:
    """Extract normalized arXiv identifiers from ADS identifier strings.

    Args:
        identifiers: Alternate identifiers returned by ADS.

    Returns:
        Ordered, de-duplicated arXiv identifiers.
    """

    arxiv_ids: List[str] = []
    seen_arxiv_ids: set[str] = set()

    for raw_identifier in identifiers:
        normalized_identifier = raw_identifier.strip()
        if not normalized_identifier:
            continue

        prefix_match = ARXIV_PREFIX_PATTERN.match(normalized_identifier)
        candidate_identifier = (
            prefix_match.group("arxiv_id").strip()
            if prefix_match is not None
            else normalized_identifier
        )

        if not (
            ARXIV_NEW_ID_PATTERN.fullmatch(candidate_identifier)
            or ARXIV_OLD_ID_PATTERN.fullmatch(candidate_identifier)
        ):
            continue

        if candidate_identifier in seen_arxiv_ids:
            continue

        seen_arxiv_ids.add(candidate_identifier)
        arxiv_ids.append(candidate_identifier)

    return arxiv_ids


def build_article_enrichment_record(
    doc: ADSArticleEnrichmentDoc,
) -> ADSArticleEnrichmentRecord:
    """Normalize one ADS enrichment document into the slim pipeline record.

    Args:
        doc: Raw enrichment document returned by ADS.

    Returns:
        Normalized enrichment record.
    """

    doi = doc.doi[0].strip() if doc.doi and doc.doi[0].strip() else None
    abstract = doc.abstract.strip() if doc.abstract and doc.abstract.strip() else None
    authors = [author.strip() for author in doc.author if author and author.strip()]

    return ADSArticleEnrichmentRecord(
        doi=doi,
        abstract=abstract,
        authors=authors,
        available_esources=ADSClient._normalize_esources(doc.esources),
        arxiv_ids=extract_arxiv_ids_from_identifiers(doc.identifier),
    )


def build_doi_resolution_record(
    doc: ADSDoiResolutionDoc,
) -> ADSDoiResolutionRecord:
    """Normalize one ADS DOI-resolution document into a pipeline record.

    Args:
        doc: Raw DOI-resolution document returned by ADS.

    Returns:
        Normalized DOI-resolution record.
    """

    resolved_doi = next(
        (candidate for candidate in (normalize_doi_text(value) for value in doc.doi) if candidate),
        None,
    )
    resolved_title = next((title.strip() for title in doc.title if title and title.strip()), None)
    resolved_keywords = [keyword.strip() for keyword in doc.keyword if keyword and keyword.strip()]
    resolved_authors = [author.strip() for author in doc.author if author and author.strip()]
    resolved_abstract = doc.abstract.strip() if doc.abstract and doc.abstract.strip() else None

    return ADSDoiResolutionRecord(
        bibcode=doc.bibcode.strip() or None,
        doi=resolved_doi,
        title=resolved_title,
        keywords=resolved_keywords,
        authors=resolved_authors,
        abstract=resolved_abstract,
        arxiv_ids=extract_arxiv_ids_from_identifiers(doc.identifier),
    )


def build_corpus_enrichment_record(
    doc: ADSCorpusEnrichmentDoc,
) -> ADSCorpusEnrichmentRecord:
    """Normalize one ADS corpus-enrichment document into a pipeline record.

    Args:
        doc: Raw corpus-enrichment document returned by ADS.

    Returns:
        Normalized corpus-enrichment record.
    """

    resolved_title = next((title.strip() for title in doc.title if title and title.strip()), None)
    resolved_keywords = [keyword.strip() for keyword in doc.keyword if keyword and keyword.strip()]
    resolved_abstract = doc.abstract.strip() if doc.abstract and doc.abstract.strip() else None
    resolved_doi = next((doi.strip() for doi in doc.doi if doi and doi.strip()), None)
    resolved_authors = [author.strip() for author in doc.author if author and author.strip()]

    return ADSCorpusEnrichmentRecord(
        title=resolved_title,
        keywords=resolved_keywords,
        abstract=resolved_abstract,
        doi=resolved_doi,
        authors=resolved_authors,
        arxiv_ids=extract_arxiv_ids_from_identifiers(doc.identifier),
    )


class ADSClient:
    """Client for interacting with the NASA ADS API."""

    def __init__(self) -> None:
        """Initialize the client by loading the API token from ``.env``."""
        self.token = self._resolve_token()
        self.base_url = ADS_API_BASE_URL

    def _resolve_token(self) -> str:
        """Resolve the ADS token from the root ``.env`` file."""
        load_local_env_file()
        return os.environ.get("ADS_TOKEN", "").strip()

    def _get_headers(self) -> Dict[str, str]:
        """Builds authenticated request headers.

        Returns:
            Headers for ADS API requests.
        """
        if not self.token or self.token.startswith("#"):
            raise ValueError("ADS API token is missing or invalid in .env (ADS_TOKEN).")

        return {"Authorization": f"Bearer {self.token}"}

    def _run_search_query(self, query: str, rows: int, fields: str) -> Dict[str, object]:
        """Run one ADS search query.

        Args:
            query: ADS query string.
            rows: Number of rows requested from ADS.
            fields: Comma-separated ADS fields to retrieve.

        Returns:
            Raw JSON payload returned by ADS.
        """
        headers = self._get_headers()
        params = {
            "q": query,
            "fl": fields,
            "rows": rows,
        }

        response = requests.get(
            self.base_url,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _run_query(self, bibcodes: List[str], fields: str) -> Dict[str, object]:
        """Run a batched ADS search query for a list of bibcodes.

        Args:
            bibcodes: Bibcodes to query.
            fields: Comma-separated ADS fields to retrieve.

        Returns:
            Raw JSON payload returned by ADS.
        """

        query_str = " OR ".join(f'"{bc}"' for bc in bibcodes)
        return self._run_search_query(
            query=f"identifier:({query_str})",
            rows=len(bibcodes),
            fields=fields,
        )

    def _run_doi_query(self, dois: List[str], fields: str) -> Dict[str, object]:
        """Run a batched ADS search query for a list of normalized DOIs.

        Args:
            dois: Normalized DOI strings to query.
            fields: Comma-separated ADS fields to retrieve.

        Returns:
            Raw JSON payload returned by ADS.
        """

        query_str = " OR ".join(f'doi:"{doi}"' for doi in dois)
        return self._run_search_query(
            query=query_str,
            rows=len(dois),
            fields=fields,
        )

    def get_dois_from_bibcodes(self, bibcodes: List[str]) -> Dict[str, Optional[str]]:
        """Fetches DOIs for a list of bibcodes using search query endpoint (GET).
        
        Args:
            bibcodes: List of bibcodes to resolve.
            
        Returns:
            Dictionary mapping bibcode to its first DOI if found.
        """
        data = self._run_query(bibcodes=bibcodes, fields="bibcode,doi")
        if "response" not in data:
            return {b: None for b in bibcodes}
            
        ads_res = ADSResponse(docs=data["response"]["docs"])
        
        mapping = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = doc.doi[0] if doc.doi else None
            
        # Ensure all requested bibcodes are in the result
        for bc in bibcodes:
            if bc not in mapping:
                mapping[bc] = None
                
        return mapping

    def get_esources_from_bibcodes(self, bibcodes: List[str]) -> Dict[str, List[str]]:
        """Fetch electronic source types for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to the list of ADS ``esources`` values.
        """

        data = self._run_query(bibcodes=bibcodes, fields="bibcode,esources")
        if "response" not in data:
            return {bibcode: [] for bibcode in bibcodes}

        ads_res = ADSEsourcesResponse(docs=data["response"]["docs"])

        mapping: Dict[str, List[str]] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = self._normalize_esources(doc.esources)

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = []

        return mapping

    def get_abstracts_from_bibcodes(self, bibcodes: List[str]) -> Dict[str, Optional[str]]:
        """Fetches abstracts for a list of bibcodes using the ADS search endpoint.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping bibcode to abstract text when available.
        """
        data = self._run_query(bibcodes=bibcodes, fields="bibcode,abstract")
        if "response" not in data:
            return {b: None for b in bibcodes}

        ads_res = ADSAbstractResponse(docs=data["response"]["docs"])

        mapping: Dict[str, Optional[str]] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = doc.abstract

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = None

        return mapping

    def get_metadata_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Fetches title and keyword metadata for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping bibcode to metadata lists for title and keyword.
        """
        data = self._run_query(bibcodes=bibcodes, fields="bibcode,title,keyword")
        if "response" not in data:
            return {
                bibcode: {"title": [], "keyword": []}
                for bibcode in bibcodes
            }

        ads_res = ADSMetadataResponse(docs=data["response"]["docs"])

        mapping: Dict[str, Dict[str, List[str]]] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = {
                "title": doc.title,
                "keyword": doc.keyword,
            }

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = {"title": [], "keyword": []}

        return mapping

    def get_titles_and_abstracts_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Fetches title and abstract metadata for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping bibcode to title and abstract strings.
        """
        data = self._run_query(bibcodes=bibcodes, fields="bibcode,title,abstract")
        if "response" not in data:
            return {
                bibcode: {"title": None, "abstract": None}
                for bibcode in bibcodes
            }

        ads_res = ADSTitleAbstractResponse(docs=data["response"]["docs"])

        mapping: Dict[str, Dict[str, Optional[str]]] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = {
                "title": doc.title[0] if doc.title else None,
                "abstract": doc.abstract,
            }

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = {"title": None, "abstract": None}

        return mapping

    def get_authors_from_bibcodes(self, bibcodes: List[str]) -> Dict[str, List[str]]:
        """Fetch ordered ADS authors for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to ordered author names.
        """

        data = self._run_query(bibcodes=bibcodes, fields="bibcode,author")
        if "response" not in data:
            return {bibcode: [] for bibcode in bibcodes}

        ads_res = ADSAuthorResponse(docs=data["response"]["docs"])

        mapping: Dict[str, List[str]] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = doc.author

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = []

        return mapping

    def get_article_enrichment_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, ADSArticleEnrichmentRecord]:
        """Fetch DOI, abstract, authors, esources, and identifiers in one ADS query.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to a normalized enrichment bundle.
        """

        data = self._run_query(
            bibcodes=bibcodes,
            fields="bibcode,doi,abstract,author,esources,identifier",
        )
        if "response" not in data:
            return {
                bibcode: ADSArticleEnrichmentRecord()
                for bibcode in bibcodes
            }

        ads_res = ADSArticleEnrichmentResponse(docs=data["response"]["docs"])

        mapping: Dict[str, ADSArticleEnrichmentRecord] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = build_article_enrichment_record(doc)

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = ADSArticleEnrichmentRecord()

        return mapping

    def get_full_metadata_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, ADSFullMetadataRecord]:
        """Fetches title, abstract, and keyword metadata for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping bibcode to normalized metadata.
        """
        data = self._run_query(bibcodes=bibcodes, fields="bibcode,title,abstract,keyword")
        if "response" not in data:
            return {
                bibcode: ADSFullMetadataRecord()
                for bibcode in bibcodes
            }

        ads_res = ADSFullMetadataResponse(docs=data["response"]["docs"])

        mapping: Dict[str, ADSFullMetadataRecord] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = ADSFullMetadataRecord(
                title=doc.title[0] if doc.title else None,
                abstract=doc.abstract,
                keywords=doc.keyword,
            )

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = ADSFullMetadataRecord()

        return mapping

    def get_corpus_enrichment_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, ADSCorpusEnrichmentRecord]:
        """Fetch title, keywords, abstract, DOI, authors, and arXiv ids in one ADS query.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to a normalized corpus enrichment bundle.
        """

        data = self._run_query(
            bibcodes=bibcodes,
            fields="bibcode,title,abstract,keyword,doi,author,identifier",
        )
        if "response" not in data:
            return {
                bibcode: ADSCorpusEnrichmentRecord()
                for bibcode in bibcodes
            }

        ads_res = ADSCorpusEnrichmentResponse(docs=data["response"]["docs"])

        mapping: Dict[str, ADSCorpusEnrichmentRecord] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = build_corpus_enrichment_record(doc)

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = ADSCorpusEnrichmentRecord()

        return mapping

    def get_metadata_from_dois(
        self,
        dois: List[str],
    ) -> Dict[str, ADSDoiResolutionRecord]:
        """Fetch ADS metadata by DOI and map results back to requested DOIs.

        Args:
            dois: Normalized DOI strings to resolve.

        Returns:
            Dictionary mapping each requested DOI to a normalized ADS record.
        """

        data = self._run_doi_query(
            dois=dois,
            fields="bibcode,doi,title,abstract,keyword,author,identifier",
        )
        if "response" not in data:
            return {doi: ADSDoiResolutionRecord() for doi in dois}

        ads_res = ADSDoiResolutionResponse(docs=data["response"]["docs"])
        requested_dois = set(dois)
        mapping: Dict[str, ADSDoiResolutionRecord] = {}

        for doc in ads_res.docs:
            record = build_doi_resolution_record(doc)
            doc_dois = [
                normalized_doi
                for normalized_doi in (normalize_doi_text(value) for value in doc.doi)
                if normalized_doi
            ]
            matched_requested_dois = [
                normalized_doi
                for normalized_doi in doc_dois
                if normalized_doi in requested_dois
            ]
            for matched_doi in matched_requested_dois:
                mapping.setdefault(matched_doi, record)

        for doi in dois:
            mapping.setdefault(doi, ADSDoiResolutionRecord())

        return mapping

    def get_gateway_links_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, ADSGatewayLinksRecord]:
        """Build ADS gateway links for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to normalized ADS gateway links.
        """

        return self.get_source_links_from_bibcodes(bibcodes=bibcodes)

    def get_source_links_from_bibcodes(
        self,
        bibcodes: List[str],
    ) -> Dict[str, ADSGatewayLinksRecord]:
        """Build ADS source-link records for a list of bibcodes.

        Args:
            bibcodes: List of bibcodes to resolve.

        Returns:
            Dictionary mapping each bibcode to normalized ADS source links.
        """

        data = self._run_query(bibcodes=bibcodes, fields="bibcode,esources,identifier")
        if "response" not in data:
            return {
                bibcode: self._build_gateway_links_record(
                    bibcode=bibcode,
                    esources=[],
                    identifiers=[],
                )
                for bibcode in bibcodes
            }

        ads_res = ADSSourcesResponse(docs=data["response"]["docs"])

        mapping: Dict[str, ADSGatewayLinksRecord] = {}
        for doc in ads_res.docs:
            mapping[doc.bibcode] = self._build_gateway_links_record(
                bibcode=doc.bibcode,
                esources=doc.esources,
                identifiers=doc.identifier,
            )

        for bibcode in bibcodes:
            if bibcode not in mapping:
                mapping[bibcode] = self._build_gateway_links_record(
                    bibcode=bibcode,
                    esources=[],
                    identifiers=[],
                )

        return mapping

    def _build_gateway_links_record(
        self,
        bibcode: str,
        esources: List[str],
        identifiers: List[str] | None = None,
    ) -> ADSGatewayLinksRecord:
        """Construct normalized ADS gateway links for one bibcode.

        Args:
            bibcode: Bibcode to resolve.
            esources: Available ADS electronic source types.
            identifiers: Alternate identifiers returned by ADS.

        Returns:
            Normalized gateway-link record for the bibcode.
        """

        normalized_esources = self._normalize_esources(esources)
        arxiv_ids = extract_arxiv_ids_from_identifiers(identifiers or [])
        gateway_urls = {
            esource: self._build_ads_gateway_url(bibcode=bibcode, esource=esource)
            for esource in normalized_esources
        }
        best_fulltext_source = self._select_best_fulltext_source(normalized_esources)

        return ADSGatewayLinksRecord(
            ads_abstract_url=self._build_ads_abstract_url(bibcode=bibcode),
            available_esources=normalized_esources,
            gateway_urls=gateway_urls,
            pub_pdf_url=gateway_urls.get("PUB_PDF"),
            eprint_pdf_url=gateway_urls.get("EPRINT_PDF"),
            pub_html_url=gateway_urls.get("PUB_HTML"),
            eprint_html_url=gateway_urls.get("EPRINT_HTML"),
            best_fulltext_url=(
                gateway_urls.get(best_fulltext_source)
                if best_fulltext_source is not None
                else None
            ),
            best_fulltext_source=best_fulltext_source,
            arxiv_ids=arxiv_ids,
            arxiv_abs_urls=[self._build_arxiv_abs_url(arxiv_id=arxiv_id) for arxiv_id in arxiv_ids],
            arxiv_pdf_urls=[self._build_arxiv_pdf_url(arxiv_id=arxiv_id) for arxiv_id in arxiv_ids],
            arxiv_eprint_urls=[self._build_arxiv_eprint_url(arxiv_id=arxiv_id) for arxiv_id in arxiv_ids],
        )

    def _build_ads_abstract_url(self, bibcode: str) -> str:
        """Construct the canonical ADS abstract page URL for one bibcode.

        Args:
            bibcode: Bibcode to resolve.

        Returns:
            ADS abstract page URL.
        """

        return f"{ADS_UI_BASE_URL}/abs/{bibcode}/abstract"

    def _build_ads_gateway_url(self, bibcode: str, esource: str) -> str:
        """Construct one ADS gateway URL for a bibcode and esource type.

        Args:
            bibcode: Bibcode to resolve.
            esource: ADS esource type such as ``PUB_PDF`` or ``EPRINT_PDF``.

        Returns:
            ADS gateway URL.
        """

        return f"{ADS_UI_BASE_URL}/link_gateway/{bibcode}/{esource}"

    def _build_arxiv_abs_url(self, arxiv_id: str) -> str:
        """Construct the canonical arXiv abstract URL for one arXiv id.

        Args:
            arxiv_id: arXiv identifier.

        Returns:
            arXiv abstract URL.
        """

        return f"https://arxiv.org/abs/{arxiv_id}"

    def _build_arxiv_pdf_url(self, arxiv_id: str) -> str:
        """Construct the canonical arXiv PDF URL for one arXiv id.

        Args:
            arxiv_id: arXiv identifier.

        Returns:
            arXiv PDF URL.
        """

        return f"https://arxiv.org/pdf/{arxiv_id}"

    def _build_arxiv_eprint_url(self, arxiv_id: str) -> str:
        """Construct the canonical arXiv source package URL for one arXiv id.

        Args:
            arxiv_id: arXiv identifier.

        Returns:
            arXiv e-print source URL.
        """

        return f"https://arxiv.org/e-print/{arxiv_id}"

    def _select_best_fulltext_source(self, esources: List[str]) -> Optional[str]:
        """Pick the best available full-text source from ADS esources.

        Args:
            esources: Normalized ADS electronic source types.

        Returns:
            Preferred source type, or ``None`` when no esource is available.
        """

        for preferred_source in ADS_FULLTEXT_SOURCE_PRIORITY:
            if preferred_source in esources:
                return preferred_source

        if not esources:
            return None

        return esources[0]

    @staticmethod
    def _normalize_esources(esources: List[str]) -> List[str]:
        """Normalize ADS electronic source values and remove duplicates.

        Args:
            esources: Raw ADS electronic source types.

        Returns:
            Upper-cased, de-duplicated esource values in original order.
        """

        normalized_esources: List[str] = []
        seen_esources: set[str] = set()

        for esource in esources:
            normalized_esource = esource.strip().upper()
            if not normalized_esource or normalized_esource in seen_esources:
                continue
            seen_esources.add(normalized_esource)
            normalized_esources.append(normalized_esource)

        return normalized_esources

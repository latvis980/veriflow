# utils/tts_service.py
"""
The True Story (TTS) Service
Client for The True Story news aggregation platform APIs.

Provides three integration paths:
1. Search API (esearch) - real-time keyword search across 6-month archive
2. Stories API - fetch cluster details by cluster_id
3. Issue monitor - poll for new editions and download tar.gz archives

Search API is the primary real-time path used during fact verification.
Tar.gz polling builds the local Supabase cache for fast entity lookups.

Endpoints:
- Issue ID:  http://static.thetruestory.news/{edition}-issue-id.json
- Issue tar: http://static.thetruestory.news/{edition}-issue.tar.gz
- Search:    https://embed-search.thetruestory.news/esearch/api
- Stories:   https://thetruestory.news/api/stories/{cluster_id}
"""

import asyncio
import time
import re
import json
import tarfile
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from langsmith import traceable

from utils.logger import fact_logger


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TTSSearchHit:
    """A single search result from the esearch API"""
    doc_id: int
    title: str  # may contain <b> highlight tags
    url: str
    timestamp: str
    text: str  # article text, may contain <b> highlights
    source_title: str
    source_slug: str
    score: float
    cluster_id: Optional[str] = None
    cluster_size: Optional[int] = None
    cluster_rank: Optional[int] = None
    cluster_title: Optional[str] = None
    group: List[Dict[str, Any]] = field(default_factory=list)
    group_score: float = 0.0

    @property
    def clean_title(self) -> str:
        """Title with <b> tags stripped"""
        return re.sub(r'</?b>', '', self.title)

    @property
    def clean_text(self) -> str:
        """Text with <b> tags stripped"""
        return re.sub(r'</?b>', '', self.text)

    @property
    def source_count(self) -> int:
        """Total sources: group items + 1 (the hit itself)"""
        return len(self.group) + 1


@dataclass
class TTSSearchResult:
    """Complete search response from esearch API"""
    hits: List[TTSSearchHit]
    total: int
    took_ms: int
    edition: str
    index: str

    @property
    def has_clustered_results(self) -> bool:
        """Whether any hits have cluster grouping"""
        return any(h.cluster_id for h in self.hits)

    def best_cluster_match(self, min_cluster_size: int = 3) -> Optional[TTSSearchHit]:
        """Return highest-scoring hit with a cluster above minimum size"""
        for hit in self.hits:
            if hit.cluster_id and (hit.cluster_size or 0) >= min_cluster_size:
                return hit
        return None


@dataclass
class TTSClusterInfo:
    """Cluster details from the Stories API"""
    cluster_id: str
    timestamp: str
    titles: List[Dict[str, Any]]
    items: List[Dict[str, Any]]  # ranked articles with source info
    categories: List[str]


@dataclass
class TTSIssueInfo:
    """Issue monitor data"""
    issue_id: str
    timestamp: str
    edition: str


# ============================================================================
# TTS SERVICE
# ============================================================================

class TTSService:
    """
    Client for The True Story APIs.

    Usage:
        tts = TTSService()

        # Real-time search during verification
        results = await tts.search("Macron Ukraine negotiations", edition="en")
        if results.hits:
            best = results.best_cluster_match(min_cluster_size=5)
            if best:
                # Use best.clean_text and best.group as evidence

        # Check for new issue
        issue = await tts.check_issue("ru")
        if issue.issue_id != last_known_id:
            clusters = await tts.download_issue("ru")
    """

    SEARCH_BASE = "https://embed-search.thetruestory.news/esearch/api"
    STORIES_BASE = "https://thetruestory.news/api/stories"
    STATIC_BASE = "http://static.thetruestory.news"

    # Rate limiting: be gentle with TTS servers
    MIN_REQUEST_INTERVAL = 1.0  # seconds between requests

    def __init__(self, request_timeout: float = 30.0):
        self.request_timeout = request_timeout
        self._last_request_time = 0.0
        self._client: Optional[httpx.AsyncClient] = None

        fact_logger.logger.info("TTSService initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-init async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout),
                follow_redirects=True,
                headers={
                    "User-Agent": "VeriFlow/1.0 (fact-checking platform)",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def _rate_limit(self):
        """Enforce minimum interval between requests"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    async def close(self):
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ========================================================================
    # SEARCH API
    # ========================================================================

    @traceable(
        name="tts_search",
        run_type="tool",
        tags=["tts", "search"]
    )
    async def search(
        self,
        query: str,
        edition: str = "en",
        index: str = "*",
        grouping: bool = True,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> TTSSearchResult:
        """
        Search The True Story via esearch API.

        Args:
            query: Search keywords (use original language of the claim)
            edition: "en" for world, "ru" for Russian
            index: "3d" (3-day), "archive" (6-month), or "*" (all)
            grouping: Group related articles by cluster
            date_from: Filter from date (YYYY-MM-DD)
            date_to: Filter to date (YYYY-MM-DD)

        Returns:
            TTSSearchResult with hits, totals, and timing
        """
        await self._rate_limit()

        params = {
            "q": query,
            "edition": edition,
            "index": index,
        }
        if grouping:
            params["grouping"] = "yes"
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        try:
            client = await self._get_client()
            response = await client.get(self.SEARCH_BASE, params=params)
            response.raise_for_status()
            data = response.json().get("data", {})

            hits = []
            for h in data.get("hits", []):
                hits.append(TTSSearchHit(
                    doc_id=h.get("doc_id", 0),
                    title=h.get("title", ""),
                    url=h.get("url", ""),
                    timestamp=h.get("timestamp", ""),
                    text=h.get("text", ""),
                    source_title=h.get("source_title", ""),
                    source_slug=h.get("source_slug", ""),
                    score=h.get("score", 0.0),
                    cluster_id=h.get("cluster_id"),
                    cluster_size=h.get("cluster_size"),
                    cluster_rank=h.get("cluster_rank"),
                    cluster_title=h.get("cluster_title"),
                    group=h.get("group", []),
                    group_score=h.get("group_score", 0.0),
                ))

            total_info = data.get("total", {})
            total = total_info.get("value", 0) if isinstance(total_info, dict) else 0

            result = TTSSearchResult(
                hits=hits,
                total=total,
                took_ms=data.get("took", {}).get("elastic", 0),
                edition=edition,
                index=index,
            )

            fact_logger.logger.info(
                f"TTS search: '{query}' ({edition}) -> "
                f"{len(hits)} hits, {total} total, {result.took_ms}ms"
            )

            return result

        except httpx.HTTPStatusError as e:
            fact_logger.logger.error(
                f"TTS search HTTP error: {e.response.status_code} for query '{query}'"
            )
            return TTSSearchResult(hits=[], total=0, took_ms=0, edition=edition, index=index)

        except Exception as e:
            fact_logger.logger.error(f"TTS search error: {e}")
            return TTSSearchResult(hits=[], total=0, took_ms=0, edition=edition, index=index)

    # ========================================================================
    # STORIES API
    # ========================================================================

    @traceable(
        name="tts_get_story",
        run_type="tool",
        tags=["tts", "stories"]
    )
    async def get_story(self, cluster_id: str) -> Optional[TTSClusterInfo]:
        """
        Fetch cluster details from the Stories API.

        Args:
            cluster_id: UUID of the cluster

        Returns:
            TTSClusterInfo with titles, items, and categories
        """
        await self._rate_limit()

        url = f"{self.STORIES_BASE}/{cluster_id}"

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            raw = response.json()
            content = raw.get("content", {}).get("data", {})
            cluster_ranks = content.get("cluster_rank", [])

            if not cluster_ranks:
                return None

            cr = cluster_ranks[0]
            cluster = cr.get("cluster", {})

            items = []
            for item in cluster.get("cluster_items", []):
                ni = item.get("news_newsitem", {})
                items.append({
                    "rank": item.get("rank"),
                    "doc_id": item.get("id"),
                    "title": ni.get("title", ""),
                    "url": ni.get("url", ""),
                    "source_date": ni.get("source_date", ""),
                    "source_title": ni.get("source", {}).get("title", ""),
                    "source_slug": ni.get("source", {}).get("slug", ""),
                })

            titles = []
            for t in cluster.get("cluster_titles", []):
                src = t.get("cluster_title_source", {})
                titles.append({
                    "title": t.get("title", ""),
                    "source_title": src.get("source", {}).get("title", ""),
                    "source_slug": src.get("source", {}).get("slug", ""),
                    "url": src.get("url", ""),
                })

            result = TTSClusterInfo(
                cluster_id=cluster_id,
                timestamp=cluster.get("timestamp", ""),
                titles=titles,
                items=items,
                categories=cluster.get("cluster_categories", []),
            )

            fact_logger.logger.info(
                f"TTS story: cluster {cluster_id[:12]}... -> "
                f"{len(items)} items"
            )

            return result

        except Exception as e:
            fact_logger.logger.error(f"TTS story fetch error: {e}")
            return None

    # ========================================================================
    # ISSUE MONITOR
    # ========================================================================

    async def check_issue(self, edition: str = "ru") -> Optional[TTSIssueInfo]:
        """
        Check the current issue ID for an edition.

        Args:
            edition: "ru" or "en"

        Returns:
            TTSIssueInfo with issue_id and timestamp
        """
        await self._rate_limit()

        url = f"{self.STATIC_BASE}/{edition}-issue-id.json"

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            return TTSIssueInfo(
                issue_id=data.get("issue_id", ""),
                timestamp=data.get("timestamp", ""),
                edition=edition,
            )

        except Exception as e:
            fact_logger.logger.error(f"TTS issue check error ({edition}): {e}")
            return None

    async def download_issue(self, edition: str = "ru") -> List[Dict[str, Any]]:
        """
        Download and parse the current issue tar.gz archive.

        Returns a list of parsed cluster dictionaries from the archive.
        Each cluster contains: cluster_id, tags, summary, docs, cite_clusters,
        categories, source_countries, etc.

        Args:
            edition: "ru" or "en"

        Returns:
            List of cluster dicts (one per JSON file in the archive)
        """
        await self._rate_limit()

        url = f"{self.STATIC_BASE}/{edition}-issue.tar.gz"

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            clusters = []
            tar_bytes = io.BytesIO(response.content)

            with tarfile.open(fileobj=tar_bytes, mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".json"):
                        continue

                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    try:
                        cluster_data = json.loads(f.read().decode("utf-8"))
                        clusters.append(cluster_data)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        fact_logger.logger.warning(
                            f"TTS: failed to parse {member.name}: {e}"
                        )

            fact_logger.logger.info(
                f"TTS issue download ({edition}): "
                f"{len(clusters)} clusters parsed"
            )

            return clusters

        except Exception as e:
            fact_logger.logger.error(f"TTS issue download error ({edition}): {e}")
            return []

    # ========================================================================
    # HIGH-LEVEL: SEARCH AND BUILD EVIDENCE
    # ========================================================================

    async def find_evidence_for_claim(
        self,
        query: str,
        edition: str = "en",
        min_cluster_size: int = 3,
        max_evidence_articles: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        High-level method: search TTS for a claim and return structured evidence.

        This is the main entry point called from orchestrators during verification.

        Args:
            query: Search keywords (from TTS router)
            edition: "en" or "ru"
            min_cluster_size: Minimum sources for a cluster to count as evidence
            max_evidence_articles: Max article texts to include in evidence

        Returns:
            Evidence dict with cluster info, article texts, and source list,
            or None if no good match found.
        """
        results = await self.search(query, edition=edition, grouping=True)

        if not results.hits:
            return None

        best = results.best_cluster_match(min_cluster_size=min_cluster_size)
        if not best:
            return None

        # Collect article texts from the top hit and its group
        evidence_texts = [
            {
                "source": best.source_title,
                "text": best.clean_text,
                "url": best.url,
            }
        ]

        for g in best.group[:max_evidence_articles - 1]:
            # Group items from search have title and source but not full text.
            # The main hit's text is the most detailed.
            evidence_texts.append({
                "source": g.get("source_title", ""),
                "title": re.sub(r'</?b>', '', g.get("title", "")),
                "url": g.get("url", ""),
            })

        # Optionally fetch more detail from stories API
        story_sources = []
        if best.cluster_id:
            story = await self.get_story(best.cluster_id)
            if story:
                story_sources = [
                    {
                        "title": item["title"],
                        "source": item["source_title"],
                        "url": item["url"],
                    }
                    for item in story.items[:10]
                ]

        return {
            "matched": True,
            "cluster_id": best.cluster_id,
            "cluster_title": re.sub(r'</?b>', '', best.cluster_title or ""),
            "cluster_size": best.cluster_size,
            "search_score": best.score,
            "edition": edition,
            "evidence_texts": evidence_texts,
            "story_sources": story_sources,
            "total_search_results": results.total,
            "source_count": (best.cluster_size or 0),
        }

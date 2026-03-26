# utils/tts_service.py
"""
The True Story (TTS) Service
Client for The True Story news aggregation platform APIs.

Provides three integration paths:
1. Search API (esearch) - real-time keyword search across 6-month archive
2. Stories API - fetch cluster details by cluster_id
3. Issue monitor - poll for new editions and download tar.gz archives
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
    title: str
    url: str
    timestamp: str
    text: str
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
        return re.sub(r'</?b>', '', self.title)

    @property
    def clean_text(self) -> str:
        return re.sub(r'</?b>', '', self.text)

    @property
    def source_count(self) -> int:
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
        return any(h.cluster_id for h in self.hits)

    def best_cluster_match(self, min_cluster_size: int = 3) -> Optional[TTSSearchHit]:
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
    items: List[Dict[str, Any]]
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
    """

    SEARCH_BASE = "https://embed-search.thetruestory.news/esearch/api"
    STORIES_BASE = "https://thetruestory.news/api/stories"
    STATIC_BASE = "http://static.thetruestory.news"

    MIN_REQUEST_INTERVAL = 1.0

    def __init__(self, request_timeout: float = 30.0):
        self.request_timeout = request_timeout
        self._last_request_time = 0.0
        self._client: Optional[httpx.AsyncClient] = None

        fact_logger.logger.info("TTSService: initialized")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            fact_logger.logger.debug("TTSService: creating new httpx.AsyncClient")
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
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            wait = self.MIN_REQUEST_INTERVAL - elapsed
            fact_logger.logger.debug(f"TTSService: rate limiting, waiting {wait:.2f}s")
            await asyncio.sleep(wait)
        self._last_request_time = time.time()

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            fact_logger.logger.debug("TTSService: client closed")

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
        """Search The True Story via esearch API."""
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

        fact_logger.logger.info(
            f"TTSService.search: query='{query}', edition={edition}, index={index}"
        )

        # English edition needs specific queries (short ones return 500)
        if edition == "en" and len(query.split()) < 3:
            fact_logger.logger.warning(
                f"TTSService.search: query too short for en edition "
                f"('{query}'), skipping"
            )
            return TTSSearchResult(
                hits=[], total=0, took_ms=0, edition=edition, index=index
            )

        try:
            client = await self._get_client()
            fact_logger.logger.debug(f"TTSService.search: GET {self.SEARCH_BASE}")

            response = await client.get(self.SEARCH_BASE, params=params)
            response.raise_for_status()

            raw_data = response.json()
            data = raw_data.get("data", {})

            fact_logger.logger.debug(
                f"TTSService.search: response status={response.status_code}, "
                f"data keys={list(data.keys())}"
            )

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
                f"TTSService.search: '{query}' ({edition}) -> "
                f"{len(hits)} hits, {total} total, {result.took_ms}ms"
            )

            # Log top hit details
            if hits:
                top = hits[0]
                fact_logger.logger.info(
                    f"TTSService.search: top hit: cluster_id={top.cluster_id}, "
                    f"cluster_size={top.cluster_size}, score={top.score:.2f}, "
                    f"source={top.source_title}"
                )

            return result

        except httpx.HTTPStatusError as e:
            fact_logger.logger.error(
                f"TTSService.search: HTTP {e.response.status_code} "
                f"for query '{query}'"
            )
            return TTSSearchResult(
                hits=[], total=0, took_ms=0, edition=edition, index=index
            )

        except httpx.ConnectError as e:
            fact_logger.logger.error(
                f"TTSService.search: connection error: {e}"
            )
            return TTSSearchResult(
                hits=[], total=0, took_ms=0, edition=edition, index=index
            )

        except Exception as e:
            fact_logger.logger.error(
                f"TTSService.search: {type(e).__name__}: {e}"
            )
            import traceback
            fact_logger.logger.error(f"TTSService traceback: {traceback.format_exc()}")
            return TTSSearchResult(
                hits=[], total=0, took_ms=0, edition=edition, index=index
            )

    # ========================================================================
    # STORIES API
    # ========================================================================

    @traceable(
        name="tts_get_story",
        run_type="tool",
        tags=["tts", "stories"]
    )
    async def get_story(self, cluster_id: str) -> Optional[TTSClusterInfo]:
        """Fetch cluster details from the Stories API."""
        await self._rate_limit()

        url = f"{self.STORIES_BASE}/{cluster_id}"
        fact_logger.logger.info(f"TTSService.get_story: fetching {cluster_id[:12]}...")

        try:
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()

            raw = response.json()
            content = raw.get("content", {}).get("data", {})
            cluster_ranks = content.get("cluster_rank", [])

            if not cluster_ranks:
                fact_logger.logger.warning(
                    f"TTSService.get_story: no cluster_rank data for {cluster_id}"
                )
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
                f"TTSService.get_story: cluster {cluster_id[:12]}... -> "
                f"{len(items)} items"
            )

            return result

        except Exception as e:
            fact_logger.logger.error(
                f"TTSService.get_story: {type(e).__name__}: {e}"
            )
            return None

    # ========================================================================
    # ISSUE MONITOR
    # ========================================================================

    async def check_issue(self, edition: str = "ru") -> Optional[TTSIssueInfo]:
        """Check the current issue ID for an edition."""
        await self._rate_limit()

        url = f"{self.STATIC_BASE}/{edition}-issue-id.json"
        fact_logger.logger.debug(f"TTSService.check_issue: GET {url}")

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
            fact_logger.logger.error(f"TTSService.check_issue ({edition}): {e}")
            return None

    async def download_issue(self, edition: str = "ru") -> List[Dict[str, Any]]:
        """Download and parse the current issue tar.gz archive."""
        await self._rate_limit()

        url = f"{self.STATIC_BASE}/{edition}-issue.tar.gz"
        fact_logger.logger.info(f"TTSService.download_issue: GET {url}")

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
                            f"TTSService: failed to parse {member.name}: {e}"
                        )

            fact_logger.logger.info(
                f"TTSService.download_issue ({edition}): "
                f"{len(clusters)} clusters parsed"
            )

            return clusters

        except Exception as e:
            fact_logger.logger.error(
                f"TTSService.download_issue ({edition}): {type(e).__name__}: {e}"
            )
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
        Main entry point called from orchestrators during verification.
        """
        fact_logger.logger.info(
            f"TTSService.find_evidence: query='{query}', edition={edition}"
        )

        results = await self.search(query, edition=edition, grouping=True)

        if not results.hits:
            fact_logger.logger.info(
                f"TTSService.find_evidence: no hits for '{query}'"
            )
            return None

        best = results.best_cluster_match(min_cluster_size=min_cluster_size)
        if not best:
            fact_logger.logger.info(
                f"TTSService.find_evidence: no cluster >= {min_cluster_size} sources "
                f"for '{query}' (top hit has cluster_size={results.hits[0].cluster_size})"
            )
            return None

        fact_logger.logger.info(
            f"TTSService.find_evidence: matched cluster '{best.clean_title[:60]}' "
            f"({best.cluster_size} sources, score={best.score:.2f})"
        )

        # Collect article texts from the top hit and its group
        evidence_texts = [
            {
                "source": best.source_title,
                "text": best.clean_text,
                "url": best.url,
            }
        ]

        for g in best.group[:max_evidence_articles - 1]:
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

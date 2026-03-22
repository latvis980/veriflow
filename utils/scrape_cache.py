# utils/scrape_cache.py
"""
Shared URL Scrape Cache for Parallel Mode Execution

When comprehensive mode runs key_claims + manipulation in parallel (Phase 1),
both modes search the web and often discover overlapping URLs. Without caching,
each mode scrapes the same URL independently -- wasting browser resources,
adding latency, and risking rate limits.

ScrapeCache wraps a single BrowserlessScraper and provides:
  - URL-level caching: each URL is scraped at most once
  - In-progress deduplication: if mode A is scraping a URL that mode B also
    needs, mode B waits for mode A's result instead of scraping again
  - Same interface as BrowserlessScraper.scrape_urls_for_facts()

Lifecycle:
  - Created once per comprehensive analysis (before Stage 2)
  - Shared across all parallel mode orchestrators
  - Closed after Stage 2 completes

NOT persistent across analyses -- VeriFlow requires fresh content for every run.
"""

import asyncio
from typing import Dict, List, Optional

from utils.logger import fact_logger


class ScrapeCache:
    """
    Shared scraping cache that deduplicates URL scrapes across parallel modes.

    Drop-in replacement for BrowserlessScraper in orchestrators:
      - scrape_urls_for_facts(urls) -> Dict[str, str]
      - url_failure_reasons property
      - stats property
      - close() method
    """

    def __init__(self, config):
        self._config = config
        # URL -> scraped content (empty string = scrape returned nothing)
        self._cache: Dict[str, str] = {}
        # URLs where scrape was attempted but returned no usable content
        self._failed_urls: set = set()
        # URLs currently being scraped by another task -- waiters get the Event
        self._in_progress: Dict[str, asyncio.Event] = {}
        # Protects _cache, _failed_urls, _in_progress from concurrent modification
        self._lock = asyncio.Lock()
        # Underlying scraper (created lazily in async context)
        self._scraper = None
        # Aggregated failure reasons from all scrape calls
        self._url_failure_reasons: Dict[str, str] = {}
        # Counters
        self._cache_hits = 0
        self._cache_misses = 0
        self._wait_hits = 0

        fact_logger.logger.info("ScrapeCache initialized for shared URL caching")

    async def _get_scraper(self):
        """Lazy-init the underlying BrowserlessScraper in the async context."""
        if self._scraper is None:
            from utils.browserless_scraper import BrowserlessScraper
            self._scraper = BrowserlessScraper(self._config)
            fact_logger.logger.info("ScrapeCache: BrowserlessScraper created")
        return self._scraper

    async def scrape_urls_for_facts(self, urls: List[str]) -> Dict[str, str]:
        """
        Scrape URLs, returning cached results where available.

        Same interface as BrowserlessScraper.scrape_urls_for_facts().

        For each URL in the input list:
          1. If already cached -> return cached content (instant)
          2. If another task is currently scraping it -> wait for that result
          3. Otherwise -> scrape it and cache the result

        Args:
            urls: List of URLs to scrape

        Returns:
            Dict mapping URL -> scraped content (only URLs with content)
        """
        if not urls:
            return {}

        results: Dict[str, str] = {}
        urls_to_scrape: List[str] = []
        events_to_wait: Dict[str, asyncio.Event] = {}

        # Phase 1: Classify each URL (cached / in-progress / needs-scraping)
        async with self._lock:
            for url in urls:
                if url in self._cache:
                    # Already scraped successfully
                    results[url] = self._cache[url]
                    self._cache_hits += 1
                elif url in self._failed_urls:
                    # Previously attempted, no usable content
                    self._cache_hits += 1
                elif url in self._in_progress:
                    # Another task is scraping this right now
                    events_to_wait[url] = self._in_progress[url]
                    self._wait_hits += 1
                else:
                    # New URL -- mark as in-progress and queue for scraping
                    urls_to_scrape.append(url)
                    event = asyncio.Event()
                    self._in_progress[url] = event
                    self._cache_misses += 1

        # Phase 2: Wait for any URLs being scraped by other tasks
        if events_to_wait:
            fact_logger.logger.info(
                f"ScrapeCache: Waiting for {len(events_to_wait)} URLs "
                f"being scraped by another mode"
            )
            for url, event in events_to_wait.items():
                await event.wait()
                # After event fires, check if content is now in cache
                if url in self._cache:
                    results[url] = self._cache[url]

        # Phase 3: Scrape the genuinely new URLs
        if urls_to_scrape:
            fact_logger.logger.info(
                f"ScrapeCache: Scraping {len(urls_to_scrape)} new URLs "
                f"({self._cache_hits} cache hits, {self._wait_hits} wait hits)"
            )

            scraper = await self._get_scraper()
            scraped = await scraper.scrape_urls_for_facts(urls_to_scrape)

            # Merge failure reasons from this scrape batch
            if hasattr(scraper, 'url_failure_reasons'):
                self._url_failure_reasons.update(scraper.url_failure_reasons)

            # Store results in cache and signal waiters
            async with self._lock:
                for url in urls_to_scrape:
                    content = scraped.get(url)
                    if content:
                        self._cache[url] = content
                        results[url] = content
                    else:
                        self._failed_urls.add(url)

                    # Signal any tasks waiting on this URL
                    if url in self._in_progress:
                        self._in_progress[url].set()
                        del self._in_progress[url]

        return results

    @property
    def url_failure_reasons(self) -> Dict[str, str]:
        """Aggregated failure reasons from all scrape calls."""
        return self._url_failure_reasons

    @property
    def stats(self) -> Dict:
        """Combined stats from the underlying scraper + cache metrics."""
        base_stats = {}
        if self._scraper and hasattr(self._scraper, 'stats'):
            base_stats = dict(self._scraper.stats)

        base_stats.update({
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "wait_hits": self._wait_hits,
            "cached_urls": len(self._cache),
            "failed_urls": len(self._failed_urls),
        })
        return base_stats

    def get_cache_summary(self) -> str:
        """Human-readable cache summary for logging."""
        total_requests = self._cache_hits + self._cache_misses + self._wait_hits
        if total_requests == 0:
            return "ScrapeCache: No URLs requested"

        hit_rate = (self._cache_hits + self._wait_hits) / total_requests * 100
        return (
            f"ScrapeCache summary: {total_requests} URL requests, "
            f"{self._cache_hits} cache hits, {self._wait_hits} wait hits, "
            f"{self._cache_misses} fresh scrapes "
            f"({hit_rate:.0f}% deduplication rate)"
        )

    async def close(self):
        """Close the underlying scraper and release browser resources."""
        summary = self.get_cache_summary()
        fact_logger.logger.info(summary)

        if self._scraper:
            try:
                await self._scraper.close()
            except Exception as e:
                fact_logger.logger.debug(f"ScrapeCache scraper cleanup: {e}")
            self._scraper = None

# utils/browserless_scraper.py
"""
Enhanced Railway Browserless Scraper with Multi-Layer Anti-Bot Evasion

NEW FEATURES:
- MULTI-STRATEGY FALLBACK: Basic -> Advanced -> Cloudflare bypass
- USER-AGENT ROTATION: Modern browser UAs (Chrome 133, Firefox 124, Safari 17, Edge 133)
- SITE-SPECIFIC SELECTORS: Custom extractors for The Hill, Reuters, and other news sites
- HUMAN BEHAVIOR SIMULATION: Random mouse movements, scrolling, delays
- DOMAIN-SPECIFIC LEARNING: Remembers which strategy works for each domain
- SMART WAIT STRATEGIES: networkidle -> load -> timed fallbacks
- COOKIE PERSISTENCE: Session continuity across requests
- ENHANCED HEADERS: Realistic browser headers with client hints
- CLOUDSCRAPER BYPASS: Solves Cloudflare JS challenges for CF-protected sites

EXISTING FEATURES:
- Proper Railway Browserless connection using chromium.connect()
- Persistent browser sessions (browsers stay open during run)
- Support for Railway replicas with load distribution
- Browser pooling for connection reuse
- Fallback to local Playwright if Railway unavailable
- TIMEOUT PROTECTION: Overall 30s timeout prevents infinite hangs
- Individual operation timeouts for robustness
- PARALLEL browser initialization (saves ~30 seconds!)
- Paywall detection for early failure
- AI-POWERED CONTENT CLEANING: Removes subscription noise, device warnings, etc.
"""

import asyncio
import io
import time
import re
import os
import random
import json
from typing import Dict, List, Optional
from urllib.parse import urlparse
from enum import Enum

from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from utils.domain_strategy_service import get_domain_strategy_service

from utils.logger import fact_logger

# BeautifulSoup fallback for when Playwright is unavailable
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# NEW: Import content cleaner for AI-powered noise removal
CONTENT_CLEANER_AVAILABLE = False
ArticleContentCleaner = None  # Will be set if import succeeds

try:
    from utils.article_content_cleaner import ArticleContentCleaner as _ArticleContentCleaner
    ArticleContentCleaner = _ArticleContentCleaner
    CONTENT_CLEANER_AVAILABLE = True
except ImportError:
    fact_logger.logger.info("[LOG] ArticleContentCleaner not available, using basic cleaning only")


# NEW: ScrapingBee fallback for 401/403 blocked sites (Imperva, Reuters, etc.)
SCRAPINGBEE_AVAILABLE = False
ScrapingBeeFallback = None
try:
    from utils.scrapingbee_fallback import ScrapingBeeFallback as _ScrapingBeeFallback
    ScrapingBeeFallback = _ScrapingBeeFallback
    SCRAPINGBEE_AVAILABLE = True
except ImportError:
    fact_logger.logger.info("ScrapingBee fallback not available (missing module or httpx)")

# NEW: CloudScraper fallback for Cloudflare-protected sites
CLOUDSCRAPER_AVAILABLE = False
CloudScraperFallback = None
try:
    from utils.cloudscraper_fallback import CloudScraperFallback as _CloudScraperFallback
    CloudScraperFallback = _CloudScraperFallback
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    fact_logger.logger.info("CloudScraper fallback not available (pip install cloudscraper)")

RESIDENTIAL_PROXY_AVAILABLE = False
ResidentialProxyScraper = None
try:
    from utils.residential_proxy_scraper import ResidentialProxyScraper as _ResProxy
    ResidentialProxyScraper = _ResProxy
    RESIDENTIAL_PROXY_AVAILABLE = True
except ImportError:
    fact_logger.logger.info("Residential proxy scraper not available")

# Visual paywall detector (GPT-4o-mini vision analysis of screenshots)
VISUAL_PAYWALL_AVAILABLE = False
VisualPaywallDetector = None
try:
    from utils.visual_paywall_detector import VisualPaywallDetector as _VPD
    VisualPaywallDetector = _VPD
    VISUAL_PAYWALL_AVAILABLE = True
except ImportError:
    fact_logger.logger.info("Visual paywall detector not available (missing module or openai)")

# PDF text extraction (pure Python, no system dependencies)
PDF_EXTRACTION_AVAILABLE = False
try:
    from pypdf import PdfReader as _PdfReader
    PDF_EXTRACTION_AVAILABLE = True
    fact_logger.logger.info("PDF extraction available (pypdf)")
except ImportError:
    try:
        from pypdfium2 import PdfDocument as _PdfDocument
        PDF_EXTRACTION_AVAILABLE = True
        fact_logger.logger.info("PDF extraction available (pypdfium2)")
    except ImportError:
        fact_logger.logger.info(
            "PDF extraction not available (pip install pypdf)"
        )


class HTTPBlockedError(Exception):
    """Raised when a site returns 401/403 at the HTTP level (WAF/anti-bot block)."""
    def __init__(self, status_code: int, url: str = ""):
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} block detected: {url}")


class ScrapingStrategy(str, Enum):
    """Enumeration of scraping strategies in order of sophistication"""
    BASIC = "basic"
    ADVANCED = "advanced"


# ============================================================================
# CONFIGURATION: User Agents, Selectors, and Strategies
# ============================================================================

# Modern user agents (updated Feb 2026)
USER_AGENTS = [
    # Chrome 133 on Windows (Chromium 133, aligns with Edge 144 cadence)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    # Firefox 147 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) Gecko/20100101 Firefox/147.0",
    # Safari 18.2 on macOS Sequoia (15.2) - still the dominant macOS version
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    # Edge 144 on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/144.0.0.0",
    # Chrome 133 on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
]

# Site-specific content selectors
SITE_SELECTORS = {
    'thehill.com': [
        '[data-content]',
        '.article__content',
        '.article-content',
        '#article-content',
        '[class*="article-body"]',
        '[class*="story-content"]',
        '.submitted-content',
    ],
    'reuters.com': [
        '[data-testid="article-body"]',
        '.article-body__content',
        '[class*="ArticleBody"]',
        '[data-module="Article"]',
        '.StandardArticleBody_body',
        '[class*="article-body"]',
    ],
    'nytimes.com': [
        '.StoryBodyCompanionColumn',
        '[name="articleBody"]',
        '.story-body',
        '#story',
    ],
    'washingtonpost.com': [
        '.article-body',
        '[data-qa="article-body"]',
        '.story-body',
    ],
    'cnn.com': [
        '.article__content',
        '.pg-rail-tall__body',
        '[class*="article-body"]',
    ],
    'bbc.com': [
        '[data-component="text-block"]',
        '.article__body',
        '.story-body',
    ],
    'foxnews.com': [
        '.article-body',
        '.article-content',
        '[class*="article-text"]',
    ],
}

# Generic fallback selectors (used if no site-specific match)
GENERIC_SELECTORS = [
    'main',
    'article',
    '[role="main"]',
    '.main-content',
    '.content',
    '.article-content',
    '.post-content',
    '#content',
    '#main',
    # WordPress selectors
    '.entry-content',
    '.post-entry',
    '.site-content',
    '.page-content',
    # News site selectors
    '.story-body',
    '.article-body',
    '#article-body',
    '.post-body',
    '#primary',
    # Elementor page builder selectors (WordPress + Elementor themes)
    '.elementor-widget-container',
    '.elementor-text-editor',
    '.elementor-section',
    '.elementor-widget-text-editor',
    # Other page builders
    '.wp-block-post-content',
    '.fl-post-content',
    '.vc_column-inner',
    # Last resort: full body (JS strips nav/header/footer automatically)
    'body',
]

class BrowserlessScraper:
    """
    Enhanced Railway Browserless scraper with multi-layered anti-bot evasion.
    """

    def __init__(self, config):
        self.config = config

        # Railway Browserless configuration
        self.is_railway = os.getenv('RAILWAY_ENVIRONMENT') is not None
        self.browserless_endpoint = os.getenv('BROWSER_PLAYWRIGHT_ENDPOINT_PRIVATE') or os.getenv('BROWSER_PLAYWRIGHT_ENDPOINT')
        self.browserless_token = os.getenv('BROWSER_TOKEN')

        # Diagnostic logging for auth debugging
        if self.browserless_endpoint:
            has_token_in_url = '?token=' in (self.browserless_endpoint or '') or '&token=' in (self.browserless_endpoint or '')
            fact_logger.logger.info(
                f"Browserless config: endpoint={'SET' if self.browserless_endpoint else 'MISSING'}, "
                f"token={'SET' if self.browserless_token else 'MISSING'}, "
                f"token_in_url={has_token_in_url}"
            )
            if not self.browserless_token and not has_token_in_url:
                fact_logger.logger.warning(
                    "BROWSER_TOKEN env var is not set and endpoint has no embedded token. "
                    "Authentication will fail. Set BROWSER_TOKEN in Railway variables."
                )

        # Support for Railway replicas
        self.replica_id = os.getenv('RAILWAY_REPLICA_ID', '0')

        # Browser pool for persistent sessions
        self.max_concurrent = 10
        self.playwright = None
        self.browser_pool: List[Browser] = []
        self.context_pool: List[BrowserContext] = []  # NEW: Store contexts for cookie persistence
        self.current_browser_index = 0
        self.session_active = False
        self._session_lock = asyncio.Lock()
        self._bound_loop_id = None  # Track which event loop owns the browser pool
        self.strategy_service = get_domain_strategy_service()

        # NEW: ScrapingBee fallback for 401/403 blocked sites
        self._scrapingbee = None
        if SCRAPINGBEE_AVAILABLE and ScrapingBeeFallback is not None:
            self._scrapingbee = ScrapingBeeFallback()

        # NEW: CloudScraper fallback for Cloudflare-protected sites
        self._cloudscraper = None
        if CLOUDSCRAPER_AVAILABLE and CloudScraperFallback is not None:
            self._cloudscraper = CloudScraperFallback()

        # In BrowserlessScraper.__init__
        self._residential_proxy = ResidentialProxyScraper() if RESIDENTIAL_PROXY_AVAILABLE else None

        # Timeouts
        self.default_timeout = 5000  # 5 seconds
        self.slow_timeout = 10000     # 10 seconds
        self.browser_launch_timeout = 10000
        self.overall_scrape_timeout = 180.0  # Increased to accommodate ScrapingBee + AI cleaning

        # Domain-specific timeouts
        self.domain_timeouts = {
            'nytimes.com': 10000,
            'washingtonpost.com': 10000,
            'wsj.com': 10000,
            'forbes.com': 10000,
            'reuters.com': 12000,  # Needs more time
            'thehill.com': 10000,
        }

        # Timing
        self.load_wait_time = 2.0
        self.interaction_delay = 0.5

        # AI-powered content cleaner
        self._content_cleaner = None  # ArticleContentCleaner, initialized lazily
        self.enable_ai_cleaning = True

        # Visual paywall detector (GPT-4o-mini vision)
        self._visual_paywall_detector = None
        if VISUAL_PAYWALL_AVAILABLE and VisualPaywallDetector is not None:
            self._visual_paywall_detector = VisualPaywallDetector(
                short_content_threshold=800
            )

        # NEW: Enhanced stats tracking
        self.stats = {
            "total_scraped": 0,
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "timeout_scrapes": 0,
            "paywall_detected": 0,
            "ai_cleaned": 0,
            "ai_cleaning_failed": 0,
            "avg_scrape_time": 0.0,
            "total_processing_time": 0.0,
            "browser_reuses": 0,
            "railway_browserless": bool(self.browserless_endpoint),
            "replica_id": self.replica_id,
            # NEW: Strategy usage tracking
            "strategy_usage": {
                ScrapingStrategy.BASIC: 0,
                ScrapingStrategy.ADVANCED: 0,
            },
            "strategy_success": {
                ScrapingStrategy.BASIC: 0,
                ScrapingStrategy.ADVANCED: 0,
            },
            # Site-specific failures
            "site_failures": {},
            # ScrapingBee fallback stats
            "scrapingbee_attempts": 0,
            "scrapingbee_successes": 0,
            # CloudScraper fallback stats
            "cloudscraper_attempts": 0,
            "cloudscraper_successes": 0,
            # Visual paywall detection stats
            "visual_paywall_checks": 0,
            "visual_paywall_detected": 0,
            # PDF extraction stats
            "pdf_extractions": 0,
            "pdf_extraction_successes": 0,
            "residential_proxy_attempts": 0,
            "residential_proxy_successes": 0,
        }

        # Per-URL failure reason tracking (readable by orchestrators)
        # Values: "paywall", "http_blocked", "all_strategies_failed", "timeout"
        self.url_failure_reasons: Dict[str, str] = {}

        if self.browserless_endpoint:
            fact_logger.logger.info(f"[LOG] Railway Browserless endpoint configured: {self.browserless_endpoint[:50]}...")
            fact_logger.logger.info(f"[LOG] Running on replica: {self.replica_id}")
        else:
            fact_logger.logger.info("[LOG] Local Playwright mode")

        fact_logger.log_component_start(
            "BrowserlessScraper",
            browserless=bool(self.browserless_endpoint),
            replica_id=self.replica_id,
            browser_pool_size=self.max_concurrent,
            ai_cleaning=CONTENT_CLEANER_AVAILABLE
        )

    def _get_content_cleaner(self):
        """Lazy initialization of content cleaner"""
        if self._content_cleaner is None and CONTENT_CLEANER_AVAILABLE and ArticleContentCleaner is not None:
            try:
                self._content_cleaner = ArticleContentCleaner(self.config)
                fact_logger.logger.info("[LOG] AI content cleaner initialized")
            except Exception as e:
                fact_logger.logger.warning(f"Failed to initialize content cleaner: {e}")
        return self._content_cleaner

    def _get_random_user_agent(self) -> str:
        """Select random modern user agent"""
        return random.choice(USER_AGENTS)

    def _get_site_selectors(self, url: str) -> List[str]:
        """Get site-specific selectors + generic fallbacks"""
        domain = urlparse(url).netloc.lower()

        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]

        # Get site-specific selectors
        site_selectors = SITE_SELECTORS.get(domain, [])

        # Combine with generic selectors
        return site_selectors + GENERIC_SELECTORS

    async def scrape_urls_for_facts(self, urls: List[str], cancel_check=None) -> Dict[str, str]:
        """
        Scrape multiple URLs with persistent browser sessions and AI cleaning.

        Args:
            urls: List of URLs to scrape
            cancel_check: Optional callable that raises an exception if cancelled.
                          Called before scraping starts and after results are collected.

        Returns:
            Dict mapping URL to scraped (and cleaned) content
        """
        if not urls:
            fact_logger.logger.warning("No URLs provided for scraping")
            return {}

        # Calculate how many browsers we actually need
        num_browsers_needed = min(len(urls), 3)  # Cap at 3 for small batches

        fact_logger.logger.info(
            f"Starting scrape of {len(urls)} URLs with persistent browsers",
            extra={"url_count": len(urls), "replica_id": self.replica_id}
        )

        # Check for cancellation before starting browser initialization
        if cancel_check:
            cancel_check()

        await self._initialize_browser_pool(min_browsers=num_browsers_needed)

        try:
            # Check for cancellation after browser initialization
            if cancel_check:
                cancel_check()

            # Guard: If no browsers initialized, return empty results
            if not self.browser_pool:
                fact_logger.logger.error(
                    "No browsers available in pool. Check BROWSER_TOKEN env var "
                    "and Browserless service status in Railway."
                )
                return {url: "" for url in urls}

            # Process URLs with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._scrape_with_semaphore(semaphore, url, i % len(self.browser_pool))
                for i, url in enumerate(urls)
            ]

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for cancellation after all scrapes complete
            if cancel_check:
                cancel_check()

            # Convert to dict and handle exceptions
            results = {}
            for url, result in zip(urls, results_list):
                if isinstance(result, Exception):
                    fact_logger.logger.error(
                        f"Scraping failed for {url}: {result}",
                        extra={"url": url, "error": str(result)}
                    )
                    results[url] = ""
                else:
                    results[url] = result

            successful = len([v for v in results.values() if v])
            self.stats["browser_reuses"] += max(0, len(urls) - len(self.browser_pool))

            fact_logger.logger.info(
                f"Scraping complete: {successful}/{len(urls)} successful",
                extra={
                    "successful": successful,
                    "total": len(urls),
                    "browser_reuses": self.stats["browser_reuses"],
                    "timeouts": self.stats["timeout_scrapes"],
                    "paywalls": self.stats["paywall_detected"],
                    "ai_cleaned": self.stats["ai_cleaned"],
                    "strategy_stats": self.stats["strategy_success"]
                }
            )

            return results

        finally:
            # Keep browsers alive for next batch
            pass

    async def _initialize_browser_pool(self, min_browsers: Optional[int] = None):
        """Initialize browser pool in PARALLEL with enhanced contexts.
        
        Detects when the event loop has changed (e.g. new request thread)
        and forces re-initialization to avoid 'Event loop is closed' errors.
        """
        async with self._session_lock:
            # Detect stale browser pool from a different event loop
            current_loop_id = id(asyncio.get_running_loop())
            
            if self.session_active and self._bound_loop_id != current_loop_id:
                fact_logger.logger.warning(
                    f"Event loop changed (was {self._bound_loop_id}, now {current_loop_id}). "
                    f"Resetting browser pool to avoid stale connections."
                )
                await self._force_reset_pool()
            
            if self.session_active and len(self.browser_pool) >= (min_browsers or 1):
                return

            target_count = min_browsers or self.max_concurrent

            if len(self.browser_pool) < target_count:
                browsers_needed = target_count - len(self.browser_pool)
                fact_logger.logger.info(f"[LOG] Initializing {browsers_needed} browsers in parallel...")

                start = time.time()
                self.playwright = await async_playwright().start()

                # Create browsers in parallel
                browser_tasks = [
                    self._create_browser(i + len(self.browser_pool))
                    for i in range(browsers_needed)
                ]
                new_browsers = await asyncio.gather(*browser_tasks, return_exceptions=True)

                # Filter out failed browsers
                for result in new_browsers:
                    if isinstance(result, Browser):
                        self.browser_pool.append(result)

                elapsed = time.time() - start
                fact_logger.logger.info(
                    f"Initialized {len(self.browser_pool)} browsers in {elapsed:.1f}s"
                )

            self.session_active = True
            self._bound_loop_id = current_loop_id

    async def _force_reset_pool(self):
        """Force-reset stale browser pool without awaiting closes on dead loop.
        
        When the event loop has changed, we can't properly close the old 
        browsers (they're bound to a closed loop). We just discard references
        and let garbage collection handle cleanup.
        """
        fact_logger.logger.info("[LOG] Force-resetting stale browser pool...")
        
        # Don't try to close old browsers -- their event loop is dead.
        # Just discard references.
        self.browser_pool = []
        self.context_pool = []
        self.session_active = False
        self._bound_loop_id = None
        
        # Stop old Playwright instance if possible (may fail, that's ok)
        if self.playwright:
            try:
                await asyncio.wait_for(self.playwright.stop(), timeout=2.0)
            except Exception:
                pass
            self.playwright = None

    async def _create_browser(self, browser_index: int) -> Optional[Browser]:
        """Create a single browser with Railway Browserless or local Playwright"""
        try:
            if not self.playwright:
                fact_logger.logger.error(f"Playwright not initialized for browser {browser_index}")
                return None
            if self.browserless_endpoint:
                # Build WebSocket endpoint with authentication
                endpoint = self.browserless_endpoint
                has_token_in_url = '?token=' in endpoint or '&token=' in endpoint

                if has_token_in_url:
                    # Endpoint already contains token (e.g. from Railway reference var)
                    ws_endpoint = endpoint
                elif self.browserless_token:
                    # Append token as query parameter
                    separator = '&' if '?' in endpoint else '?'
                    ws_endpoint = f"{endpoint}{separator}token={self.browserless_token}"
                else:
                    # No token available -- warn and try anyway (will likely 401)
                    fact_logger.logger.warning(
                        "No BROWSER_TOKEN set. Connection will likely fail with 401."
                    )
                    ws_endpoint = endpoint

                browser = await self.playwright.chromium.connect(
                    ws_endpoint,
                    timeout=self.browser_launch_timeout
                )
                fact_logger.logger.debug(f"[LOG] Connected to Railway browser {browser_index}")
            else:
                # Local Playwright
                browser = await self.playwright.chromium.launch(
                    headless=True,
                    timeout=self.browser_launch_timeout,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                    ]
                )
                fact_logger.logger.debug(f"[LOG] Launched local browser {browser_index}")

            return browser

        except Exception as e:
            fact_logger.logger.error(f"[LOG] Failed to create browser {browser_index}: {e}")
            return None

    async def _scrape_with_semaphore(self, semaphore: asyncio.Semaphore, url: str, browser_index: int) -> str:
        """Scrape using persistent browser from pool"""
        async with semaphore:
            return await self._scrape_single_url(url, browser_index)

    async def _scrape_single_url(self, url: str, browser_index: int) -> str:
        """Scrape single URL with timeout protection and multi-strategy fallback"""
        start_time = time.time()
        self.stats["total_scraped"] += 1

        # --- PDF DETECTION: bypass Playwright entirely for PDF URLs ---
        if self._is_pdf_url(url):
            fact_logger.logger.info(
                f"PDF URL detected, using direct extraction: {url}"
            )
            try:
                content = await asyncio.wait_for(
                    self._extract_pdf_content(url),
                    timeout=60.0
                )
                if content:
                    processing_time = time.time() - start_time
                    self.stats["successful_scrapes"] += 1
                    self.stats["pdf_extraction_successes"] += 1
                    self.stats["total_processing_time"] += processing_time
                    fact_logger.logger.info(
                        f"PDF extracted successfully: {url} "
                        f"({len(content)} chars, {processing_time:.1f}s)"
                    )
                    return content
                else:
                    fact_logger.logger.warning(
                        f"PDF extraction failed (empty content, likely blocked or scanned): {url}"
                    )
                    return ""
            except asyncio.TimeoutError:
                fact_logger.logger.warning(
                    f"PDF extraction timed out after 60s: {url}"
                )
                return ""
            except Exception as e:
                fact_logger.logger.warning(
                    f"PDF extraction failed for {url}: {type(e).__name__}: {e}"
                )
                return ""
            # PDF handling is self-contained -- Playwright cannot render raw PDFs
        # --- END PDF DETECTION ---

        # Select browser from pool
        if browser_index >= len(self.browser_pool):
            browser_index = 0

        browser = self.browser_pool[browser_index]

        # Overall timeout to prevent infinite hangs
        try:
            return await asyncio.wait_for(
                self._scrape_url_multi_strategy(url, browser_index, browser, start_time),
                timeout=self.overall_scrape_timeout
            )
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            self.stats["failed_scrapes"] += 1
            self.stats["timeout_scrapes"] += 1
            if url not in self.url_failure_reasons:
                self.url_failure_reasons[url] = "timeout"
            fact_logger.logger.error(
                f"TIMEOUT after {processing_time:.1f}s: {url}",
                extra={"url": url, "browser_index": browser_index, "timeout": self.overall_scrape_timeout}
            )
            return ""
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["failed_scrapes"] += 1
            if url not in self.url_failure_reasons:
                self.url_failure_reasons[url] = "scrape_failed"
            fact_logger.logger.error(
                f"Scraping error for {url}: {e}",
                extra={"url": url, "duration": processing_time, "error": str(e)}
            )
            return ""

    async def _scrape_url_multi_strategy(
        self,
        url: str,
        browser_index: int,
        browser: Browser,
        start_time: float
    ) -> str:
        """
        Multi-strategy scraping with Supabase domain learning.
        Tries strategies in order:
            known -> basic -> advanced -> ScrapingBee -> ResidentialProxy -> CloudScraper

        If a 401/403 HTTP block is detected, immediately breaks to ScrapingBee fallback.
        If ScrapingBee succeeds, saves "scrapingbee" as the domain strategy so future
        requests skip Playwright entirely.
        If ScrapingBee also fails, tries ResidentialProxy (for IP-reputation-blocked sites
        like Reuters, Bloomberg, FT, AP that block datacenter IPs with no CAPTCHA).
        If ResidentialProxy also fails, tries CloudScraper (Cloudflare JS challenges).
        Each successful fallback saves its strategy to Supabase for future fast-routing.
        """
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]

        # Check Supabase for known working strategy
        known_strategy = self.strategy_service.get_strategy(domain)

        # If domain is known to need ScrapingBee, skip Playwright entirely
        if known_strategy == "scrapingbee":
            fact_logger.logger.info(
                f"[ScrapingBee] Known ScrapingBee domain: {domain} -- skipping Playwright",
                extra={"domain": domain, "source": "supabase"}
            )
            content = await self._try_scrapingbee_fallback(url, domain, start_time, browser_index, browser)
            if content:
                return content
            # If ScrapingBee also failed this time, fall through to Playwright strategies

        # If domain is known to need ResidentialProxy, skip Playwright and ScrapingBee entirely
        if known_strategy == "residential_proxy":
            fact_logger.logger.info(
                f"[ResProxy] Known residential proxy domain: {domain} -- skipping Playwright",
                extra={"domain": domain, "source": "supabase"}
            )
            content = await self._try_residential_proxy_fallback(url, domain, start_time, browser_index, browser)
            if content:
                return content
            # If residential proxy also failed this time, fall through to Playwright strategies

        # If domain is known to need CloudScraper, skip Playwright entirely
        if known_strategy == "cloudscraper":
            fact_logger.logger.info(
                f"[CloudScraper] Known CloudScraper domain: {domain} -- skipping Playwright",
                extra={"domain": domain, "source": "supabase"}
            )
            content = await self._try_cloudscraper_fallback(url, domain, start_time, browser_index, browser)
            if content:
                return content
            # If CloudScraper also failed this time, fall through to Playwright strategies

        # Try known Playwright strategy first (if it's not a fallback-only strategy)
        if known_strategy and known_strategy not in ("scrapingbee", "residential_proxy", "cloudscraper"):
            fact_logger.logger.info(
                f"Using known strategy for {domain}: {known_strategy}",
                extra={"domain": domain, "strategy": known_strategy, "source": "supabase"}
            )

            try:
                content = await self._try_strategy(
                    url, browser_index, browser, start_time, known_strategy
                )
            except HTTPBlockedError:
                content = ""
                fact_logger.logger.info(
                    f"HTTP block on known strategy for {domain} -- trying ScrapingBee"
                )
                sb_content = await self._try_scrapingbee_fallback(url, domain, start_time, browser_index, browser)
                if sb_content:
                    return sb_content

            if content:
                processing_time = time.time() - start_time
                self.strategy_service.save_strategy(domain, known_strategy)
                self.stats["strategy_success"][known_strategy] += 1
                fact_logger.logger.debug(
                    f"Known strategy succeeded for {domain}",
                    extra={
                        "domain": domain,
                        "strategy": known_strategy,
                        "duration_ms": int(processing_time * 1000),
                        "content_length": len(content)
                    }
                )
                return content
            else:
                fact_logger.logger.warning(
                    f"Known strategy '{known_strategy}' failed for {domain}, trying all strategies",
                    extra={"domain": domain, "failed_strategy": known_strategy}
                )

        # Try Playwright strategies in order: basic -> advanced
        strategies = [ScrapingStrategy.BASIC, ScrapingStrategy.ADVANCED]

        http_blocked = False

        for strategy in strategies:
            fact_logger.logger.info(
                f"Trying {strategy} strategy for {url}",
                extra={"url": url, "strategy": strategy, "domain": domain}
            )

            try:
                content = await self._try_strategy(
                    url, browser_index, browser, start_time, strategy
                )

                if content:
                    processing_time = time.time() - start_time
                    self.strategy_service.save_strategy(domain, strategy)
                    self.stats["strategy_success"][strategy] += 1

                    fact_logger.logger.info(
                        f"Learned strategy: {domain} -> {strategy}",
                        extra={
                            "domain": domain,
                            "strategy": strategy,
                            "duration_ms": int(processing_time * 1000),
                            "is_new_learning": known_strategy is None,
                            "content_length": len(content)
                        }
                    )
                    return content
                else:
                    fact_logger.logger.debug(
                        f"{strategy} strategy failed for {url}",
                        extra={"url": url, "strategy": strategy, "domain": domain}
                    )

            except HTTPBlockedError:
                fact_logger.logger.info(
                    f"[ResProxy] HTTP 401 block detected for {domain} -- "
                    f"skipping other fallbacks, going directly to residential proxy"
                )
                http_blocked = True
                self.url_failure_reasons[url] = "http_blocked"
                # Go directly to residential proxy -- ScrapingBee and CloudScraper
                # will also be blocked by the same IP-reputation check, not worth trying.
                rp_content = await self._try_residential_proxy_fallback(
                    url, domain, start_time, browser_index, browser
                )
                if rp_content:
                    self.url_failure_reasons.pop(url, None)
                    return rp_content
                # Residential proxy exhausted its own retries -- nothing left to try.
                fact_logger.logger.warning(
                    f"[ResProxy] Residential proxy failed after retries for {domain} -- giving up"
                )
                break

        # All Playwright strategies failed (no HTTP block) -- try ScrapingBee
        if not http_blocked:
            sb_content = await self._try_scrapingbee_fallback(url, domain, start_time, browser_index, browser)
            if sb_content:
                self.url_failure_reasons.pop(url, None)  # Clear -- fallback succeeded
                return sb_content

            # ScrapingBee failed -- try ResidentialProxy (IP-reputation-blocked sites)
            rp_content = await self._try_residential_proxy_fallback(url, domain, start_time, browser_index, browser)
            if rp_content:
                self.url_failure_reasons.pop(url, None)  # Clear -- fallback succeeded
                return rp_content

            # ResidentialProxy also failed -- try CloudScraper (Cloudflare JS challenge bypass)
            cs_content = await self._try_cloudscraper_fallback(url, domain, start_time, browser_index, browser)
            if cs_content:
                self.url_failure_reasons.pop(url, None)  # Clear -- fallback succeeded
                return cs_content

        # Everything failed -- set final reason if not already set (paywall takes priority)
        if url not in self.url_failure_reasons:
            if http_blocked:
                self.url_failure_reasons[url] = "http_blocked"
            else:
                self.url_failure_reasons[url] = "all_strategies_failed"

        fact_logger.logger.error(
            f"All strategies failed for {url}" + (" (HTTP blocked)" if http_blocked else ""),
            extra={
                "url": url,
                "domain": domain,
                "strategies_tried": [s for s in strategies],
                "known_strategy": known_strategy,
                "http_blocked": http_blocked
            }
        )

        # Track site-specific failure
        if domain not in self.stats["site_failures"]:
            self.stats["site_failures"][domain] = 0
        self.stats["site_failures"][domain] += 1

        return ""

    async def _extract_and_clean_content(self, page: Page, url: str) -> str:
        """
        Shared content extraction + cleaning pipeline.
        Used by both Playwright strategies and ScrapingBee fallback.

        Pipeline:
        1. Extract structured content using site-specific JS selectors
        2. Basic regex cleaning
        3. AI-powered cleaning (if available)

        Returns cleaned content string, or empty string on failure.
        """
        # Step 1: Extract structured content via JS selectors
        raw_content = await asyncio.wait_for(
            self._extract_structured_content(page, url),
            timeout=10.0
        )

        if not raw_content or len(raw_content.strip()) <= 100:
            fact_logger.logger.debug(
                f"Insufficient content extracted from {url}",
                extra={"content_length": len(raw_content) if raw_content else 0}
            )
            return ""

        # Step 2: Basic regex cleaning
        content = self._clean_content(raw_content)

        # Step 3: AI-powered cleaning
        if self.enable_ai_cleaning and CONTENT_CLEANER_AVAILABLE:
            try:
                cleaner = self._get_content_cleaner()
                if cleaner is not None:
                    cleaning_result = await asyncio.wait_for(
                        cleaner.clean(url, content),
                        timeout=60.0
                    )

                    if (cleaning_result.success
                            and cleaning_result.cleaned is not None
                            and cleaning_result.cleaned.body):

                        original_len = len(content)
                        content = cleaning_result.cleaned.body
                        self.stats["ai_cleaned"] += 1

                        fact_logger.logger.info(
                            f"AI cleaned: {original_len} -> {len(content)} chars "
                            f"({cleaning_result.reduction_percent:.0f}% noise removed)"
                        )
                    else:
                        self.stats["ai_cleaning_failed"] += 1
            except asyncio.TimeoutError:
                self.stats["ai_cleaning_failed"] += 1
                fact_logger.logger.warning(
                    f"AI cleaning timed out (60s) for {url}, using regex-cleaned text"
                )
            except Exception as e:
                self.stats["ai_cleaning_failed"] += 1
                fact_logger.logger.warning(f"AI cleaning failed: {type(e).__name__}: {e}")

        return content

    def _extract_with_beautifulsoup(self, raw_html: str, url: str) -> str:
        """
        Fallback content extraction using BeautifulSoup when Playwright is unavailable.
        
        This handles the case where CloudScraper or ScrapingBee successfully fetch 
        raw HTML, but Playwright can't process it (e.g. event loop issues).
        
        Uses the same site-specific selector logic as the Playwright path.
        """
        if not BS4_AVAILABLE or not BeautifulSoup:
            return ""

        try:
            soup = BeautifulSoup(raw_html, 'lxml')  # type: ignore[possibly-undefined]
            
            # Remove unwanted elements (same as Playwright JS extraction)
            for tag in soup.find_all(['script', 'style', 'noscript', 'nav', 'footer', 
                                       'header', 'aside', 'iframe']):
                tag.decompose()
            
            # Remove elements by class/role patterns
            unwanted_patterns = [
                'sidebar', 'navigation', 'nav', 'menu', 'cookie', 'popup', 
                'modal', 'advertisement', 'ad', 'social-share', 'comments', 
                'related-articles', 'newsletter', 'subscription'
            ]
            for pattern in unwanted_patterns:
                for el in soup.find_all(class_=lambda c: c and pattern in str(c).lower()):  # type: ignore[arg-type]
                    el.decompose()
            for el in soup.find_all(attrs={'role': ['navigation', 'banner', 'contentinfo']}):
                el.decompose()
            
            # Try site-specific selectors first
            selectors = self._get_site_selectors(url) if url else GENERIC_SELECTORS
            best_content = ""
            best_score = 0
            
            for selector in selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        text = element.get_text(separator='\n', strip=True)
                        # Score: paragraph count * average length
                        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 20]
                        score = len(paragraphs) * (sum(len(p) for p in paragraphs) / max(len(paragraphs), 1))
                        
                        if score > best_score:
                            best_score = score
                            best_content = text
                except Exception:
                    continue
            
            if best_content and len(best_content.strip()) > 100:
                # Apply basic regex cleaning
                content = self._clean_content(best_content)
                fact_logger.logger.info(
                    f"[BS4 Fallback] Extracted {len(content)} chars from {url}",
                    extra={"url": url, "method": "beautifulsoup"}
                )
                return content
            
            # Last resort: try article tag or main content area
            for tag_name in ['article', 'main', '[role="main"]']:
                element = soup.select_one(tag_name) if '[' in tag_name else soup.find(tag_name)
                if element:
                    text = element.get_text(separator='\n', strip=True)
                    if len(text.strip()) > 200:
                        content = self._clean_content(text)
                        fact_logger.logger.info(
                            f"[BS4 Fallback] Extracted {len(content)} chars via <{tag_name}> from {url}",
                            extra={"url": url, "method": "beautifulsoup_generic"}
                        )
                        return content
            
            return ""
            
        except Exception as e:
            fact_logger.logger.warning(f"[BS4 Fallback] Failed for {url}: {e}")
            return ""

    async def _try_strategy(
        self,
        url: str,
        browser_index: int,
        browser: Browser,
        start_time: float,
        strategy: str
    ) -> str:
        """Try a specific Playwright scraping strategy"""
        self.stats["strategy_usage"][strategy] += 1

        page = None
        context = None

        try:
            # Create context based on strategy
            context = await self._create_context(browser, strategy)

            # Create page
            page = await asyncio.wait_for(context.new_page(), timeout=10.0)

            # Configure page based on strategy
            await self._configure_page(page, strategy)

            # Navigate with strategy-specific wait and capture HTTP status
            http_status = await self._navigate_with_strategy(page, url, strategy)

            # Detect HTTP-level blocks (401/403) -- raise to skip remaining strategies
            if http_status in (401, 403):
                raise HTTPBlockedError(http_status, url)

            # Early paywall detection
            if await self._detect_paywall(page):
                self.stats["failed_scrapes"] += 1
                self.stats["paywall_detected"] += 1
                self.url_failure_reasons[url] = "paywall"
                fact_logger.logger.warning(f"Paywall detected, skipping: {url}")
                return ""

            # Apply human-like behaviors for advanced strategies
            if strategy == ScrapingStrategy.ADVANCED:
                await self._simulate_human_behavior(page)

            # Shared extraction + cleaning pipeline
            content = await self._extract_and_clean_content(page, url)

            if content:
                # --- VISUAL PAYWALL CHECK ---
                # If content is suspiciously short, take a full-page screenshot
                # and ask GPT-4o-mini vision to check for paywalls
                if (self._visual_paywall_detector
                        and self._visual_paywall_detector.should_check(len(content))):

                    self.stats["visual_paywall_checks"] += 1
                    fact_logger.logger.info(
                        f"Content seems short ({len(content)} chars), "
                        f"running visual paywall check for {url}"
                    )

                    try:
                        vp_result = await asyncio.wait_for(
                            self._visual_paywall_detector.detect(
                                page, url, len(content)
                            ),
                            timeout=20.0
                        )

                        if vp_result.is_paywalled and vp_result.confidence >= 0.7:
                            self.stats["failed_scrapes"] += 1
                            self.stats["paywall_detected"] += 1
                            self.stats["visual_paywall_detected"] += 1
                            self.url_failure_reasons[url] = "paywall"
                            fact_logger.logger.warning(
                                f"VISUAL PAYWALL detected for {url}: "
                                f"{vp_result.paywall_type} "
                                f"(confidence={vp_result.confidence:.0%}) "
                                f"-- {vp_result.description}"
                            )
                            return ""
                        else:
                            fact_logger.logger.info(
                                f"Visual check passed -- no paywall "
                                f"(confidence={vp_result.confidence:.0%})"
                            )

                    except asyncio.TimeoutError:
                        fact_logger.logger.warning(
                            "Visual paywall check timed out (20s), "
                            "proceeding with extracted content"
                        )
                    except Exception as e:
                        fact_logger.logger.warning(
                            f"Visual paywall check failed: {e}, "
                            f"proceeding with extracted content"
                        )
                # --- END VISUAL PAYWALL CHECK ---

                processing_time = time.time() - start_time
                self.stats["successful_scrapes"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_scrape_time"] = (
                    self.stats["total_processing_time"] / self.stats["total_scraped"]
                )

                fact_logger.logger.info(
                    f"Successfully scraped with {strategy}: {url}",
                    extra={
                        "url": url,
                        "duration": processing_time,
                        "content_length": len(content),
                        "strategy": strategy
                    }
                )
                return content
            else:
                return ""

        except HTTPBlockedError:
            raise  # Re-raise so _scrape_url_multi_strategy can catch it
        except Exception as e:
            fact_logger.logger.debug(f"Strategy {strategy} failed for {url}: {e}")
            return ""
        finally:
            # Close page
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=3.0)
                except Exception:
                    pass

            # Close context
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=3.0)
                except Exception:
                    pass

    async def _try_scrapingbee_fallback(
        self, url: str, domain: str, start_time: float,
        browser_index: int, browser: Browser
    ) -> str:
        """
        Try ScrapingBee API as a last resort for HTTP-blocked sites.

        ScrapingBee is used ONLY as an HTTP transport (bypasses WAF/anti-bot).
        Content extraction and cleaning use the same Playwright-based pipeline
        as all other strategies: HTML is loaded into a Playwright page, then
        processed through _extract_and_clean_content().

        If successful, saves "scrapingbee" as the domain strategy in Supabase
        so future requests skip Playwright navigation entirely.
        """
        if not self._scrapingbee or not self._scrapingbee.enabled:
            fact_logger.logger.debug(
                f"ScrapingBee fallback not available for {domain}"
            )
            return ""

        self.stats["scrapingbee_attempts"] += 1

        # Step 1: Fetch raw HTML via ScrapingBee (WAF bypass)
        try:
            raw_html = await self._scrapingbee.fetch_raw_html(url)
        except Exception as e:
            fact_logger.logger.error(f"[ScrapingBee] Exception fetching HTML: {e}")
            raw_html = None

        if not raw_html:
            return ""

        # Step 2: Load HTML into Playwright page for standard extraction
        page = None
        context = None

        try:
            context = await self._create_context(browser, ScrapingStrategy.BASIC)
            page = await asyncio.wait_for(context.new_page(), timeout=10.0)
            await page.set_content(raw_html, wait_until='domcontentloaded')

            # Step 3: Same extraction + cleaning pipeline as Playwright path
            content = await self._extract_and_clean_content(page, url)

            if content:
                processing_time = time.time() - start_time
                self.stats["successful_scrapes"] += 1
                self.stats["scrapingbee_successes"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_scrape_time"] = (
                    self.stats["total_processing_time"] / max(self.stats["total_scraped"], 1)
                )

                # Save "scrapingbee" as strategy -- next time skip Playwright navigation
                self.strategy_service.save_strategy(domain, "scrapingbee")

                fact_logger.logger.info(
                    f"[ScrapingBee] Successfully scraped {url} ({len(content)} chars, {processing_time:.1f}s)",
                    extra={"url": url, "domain": domain, "strategy": "scrapingbee", "content_length": len(content)}
                )
                return content

        except Exception as e:
            fact_logger.logger.warning(f"[ScrapingBee] Playwright extraction failed for {url}: {e}")
            
            # Fallback: try BeautifulSoup extraction if we have the raw HTML
            if raw_html and BS4_AVAILABLE:
                fact_logger.logger.info(f"[ScrapingBee] Trying BeautifulSoup fallback for {url}")
                content = self._extract_with_beautifulsoup(raw_html, url)
                if content:
                    processing_time = time.time() - start_time
                    self.stats["successful_scrapes"] += 1
                    self.stats["scrapingbee_successes"] += 1
                    self.strategy_service.save_strategy(domain, "scrapingbee")
                    fact_logger.logger.info(
                        f"[ScrapingBee+BS4] Successfully extracted {len(content)} chars from {url}",
                        extra={"url": url, "domain": domain, "strategy": "scrapingbee_bs4"}
                    )
                    return content

        finally:
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=3.0)
                except Exception:
                    pass
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=3.0)
                except Exception:
                    pass

        return ""

    async def _try_residential_proxy_fallback(
        self, url: str, domain: str, start_time: float,
        browser_index: int, browser: Browser
    ) -> str:
        """
        Try residential proxy as a fallback for IP-reputation-blocked sites.

        Sites like Reuters, Bloomberg, FT, and AP News reject requests from
        datacenter IP ranges at the network edge -- no CAPTCHA, just a silent
        block or 403. A residential proxy routes the request through a real
        ISP-assigned IP, making it indistinguishable from a regular user visit.

        Uses the same two-step pattern as ScrapingBee and CloudScraper:
          1. Fetch raw HTML via residential proxy (bypasses IP reputation block)
          2. Load HTML into Playwright for standard content extraction + cleaning

        If successful, saves "residential_proxy" to Supabase so future requests
        for this domain skip all prior steps and go directly here.
        """
        if not self._residential_proxy or not self._residential_proxy.enabled:
            fact_logger.logger.debug(
                f"[ResProxy] Residential proxy not available for {domain}"
            )
            return ""

        self.stats["residential_proxy_attempts"] += 1

        # Step 1: Fetch raw HTML via residential proxy
        try:
            raw_html = await self._residential_proxy.fetch_raw_html(url)
        except Exception as e:
            fact_logger.logger.error(f"[ResProxy] Exception fetching HTML for {url}: {e}")
            raw_html = None

        if not raw_html:
            return ""

        # Step 2: Load HTML into Playwright page for standard extraction
        page = None
        context = None

        try:
            context = await self._create_context(browser, ScrapingStrategy.BASIC)
            page = await asyncio.wait_for(context.new_page(), timeout=10.0)
            await page.set_content(raw_html, wait_until='domcontentloaded')

            content = await self._extract_and_clean_content(page, url)

            if content:
                processing_time = time.time() - start_time
                self.stats["successful_scrapes"] += 1
                self.stats["residential_proxy_successes"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_scrape_time"] = (                                          # missing in your version
                    self.stats["total_processing_time"] / max(self.stats["total_scraped"], 1)
                )

                self.strategy_service.save_strategy(domain, "residential_proxy")

                fact_logger.logger.info(
                    f"[ResProxy] Successfully scraped {url} ({len(content)} chars, {processing_time:.1f}s)",
                    extra={
                        "url": url,
                        "domain": domain,
                        "strategy": "residential_proxy",
                        "content_length": len(content)
                    }
                )
                return content

        except Exception as e:
            fact_logger.logger.warning(f"[ResProxy] Playwright extraction failed for {url}: {e}")

            # BS4 fallback if Playwright can't process the HTML                            # missing in your version
            if raw_html and BS4_AVAILABLE:
                fact_logger.logger.info(f"[ResProxy] Trying BeautifulSoup fallback for {url}")
                content = self._extract_with_beautifulsoup(raw_html, url)
                if content:
                    processing_time = time.time() - start_time
                    self.stats["successful_scrapes"] += 1
                    self.stats["residential_proxy_successes"] += 1
                    self.stats["total_processing_time"] += processing_time
                    self.strategy_service.save_strategy(domain, "residential_proxy")
                    fact_logger.logger.info(
                        f"[ResProxy+BS4] Successfully extracted {len(content)} chars from {url}",
                        extra={"url": url, "domain": domain, "strategy": "residential_proxy_bs4"}
                    )
                    return content

        finally:
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=3.0)
                except Exception:
                    pass
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=3.0)
                except Exception:
                    pass

        return ""

    async def _try_cloudscraper_fallback(
        self, url: str, domain: str, start_time: float,
        browser_index: int, browser: Browser
    ) -> str:
        """
        Try CloudScraper as a fallback for Cloudflare-protected sites.

        CloudScraper solves Cloudflare JavaScript challenges locally
        (no API key, no per-request cost). Used ONLY as an HTTP transport.
        Content extraction and cleaning use the same Playwright-based pipeline
        as all other strategies.

        If successful, saves "cloudscraper" as the domain strategy in Supabase
        so future requests skip Playwright navigation entirely.
        """
        if not self._cloudscraper or not self._cloudscraper.enabled:
            fact_logger.logger.debug(
                f"CloudScraper fallback not available for {domain}"
            )
            return ""

        self.stats["cloudscraper_attempts"] += 1

        # Step 1: Fetch raw HTML via CloudScraper (Cloudflare bypass)
        try:
            raw_html = await self._cloudscraper.fetch_raw_html(url)
        except Exception as e:
            fact_logger.logger.error(f"[CloudScraper] Exception fetching HTML: {e}")
            raw_html = None

        if not raw_html:
            return ""

        # Step 2: Load HTML into Playwright page for standard extraction
        page = None
        context = None

        try:
            context = await self._create_context(browser, ScrapingStrategy.BASIC)
            page = await asyncio.wait_for(context.new_page(), timeout=10.0)
            await page.set_content(raw_html, wait_until='domcontentloaded')

            # Step 3: Same extraction + cleaning pipeline as Playwright path
            content = await self._extract_and_clean_content(page, url)

            if content:
                processing_time = time.time() - start_time
                self.stats["successful_scrapes"] += 1
                self.stats["cloudscraper_successes"] += 1
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_scrape_time"] = (
                    self.stats["total_processing_time"] / max(self.stats["total_scraped"], 1)
                )

                # Save "cloudscraper" as strategy -- next time skip Playwright
                self.strategy_service.save_strategy(domain, "cloudscraper")

                fact_logger.logger.info(
                    f"[CloudScraper] Successfully scraped {url} ({len(content)} chars, {processing_time:.1f}s)",
                    extra={"url": url, "domain": domain, "strategy": "cloudscraper", "content_length": len(content)}
                )
                return content

        except Exception as e:
            fact_logger.logger.warning(f"[CloudScraper] Playwright extraction failed for {url}: {e}")
            
            # Fallback: try BeautifulSoup extraction if we have the raw HTML
            if raw_html and BS4_AVAILABLE:
                fact_logger.logger.info(f"[CloudScraper] Trying BeautifulSoup fallback for {url}")
                content = self._extract_with_beautifulsoup(raw_html, url)
                if content:
                    processing_time = time.time() - start_time
                    self.stats["successful_scrapes"] += 1
                    self.stats["cloudscraper_successes"] += 1
                    self.stats["total_processing_time"] += processing_time
                    self.strategy_service.save_strategy(domain, "cloudscraper")
                    fact_logger.logger.info(
                        f"[CloudScraper+BS4] Successfully extracted {len(content)} chars from {url}",
                        extra={"url": url, "domain": domain, "strategy": "cloudscraper_bs4"}
                    )
                    return content

        finally:
            if page:
                try:
                    await asyncio.wait_for(page.close(), timeout=3.0)
                except Exception:
                    pass
            if context:
                try:
                    await asyncio.wait_for(context.close(), timeout=3.0)
                except Exception:
                    pass

        return ""

    async def _create_context(self, browser: Browser, strategy: str) -> BrowserContext:
        """Create browser context with strategy-specific configuration"""
        user_agent = self._get_random_user_agent()

        # Base context options
        context_options = {
            'user_agent': user_agent,
            'viewport': {'width': 1920, 'height': 1080},
            'locale': 'en-US',
            'timezone_id': 'America/New_York',
        }

        # Enhanced options for advanced strategies
        if strategy == ScrapingStrategy.ADVANCED:
            context_options.update({
                'permissions': [],
                'geolocation': {'latitude': 40.7128, 'longitude': -74.0060},  # NYC
                'color_scheme': 'light',
                'has_touch': False,
                'is_mobile': False,
                'java_script_enabled': True,
                'bypass_csp': True,
                'ignore_https_errors': True,
            })

        context = await browser.new_context(**context_options)

        # Set extra HTTP headers for better mimicry
        if strategy == ScrapingStrategy.ADVANCED:
            await context.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            })

        return context

    async def _configure_page(self, page: Page, strategy: str):
        """Configure page with strategy-specific optimizations"""
        try:
            # Block unnecessary resources for all strategies
            await page.route("**/*", self._block_resources)

            # Add init scripts for advanced strategies
            if strategy == ScrapingStrategy.ADVANCED:
                await page.add_init_script("""
                    // Remove webdriver flag
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });

                    // Disable image loading
                    Object.defineProperty(HTMLImageElement.prototype, 'src', {
                        set: function() { /* blocked */ },
                        get: function() { return ''; }
                    });

                    // Speed up animations
                    document.addEventListener('DOMContentLoaded', function() {
                        const style = document.createElement('style');
                        style.textContent = `
                            *, *::before, *::after {
                                animation-duration: 0.01ms !important;
                                animation-delay: -0.01ms !important;
                                transition-duration: 0.01ms !important;
                                transition-delay: -0.01ms !important;
                            }
                        `;
                        document.head.appendChild(style);
                    });

                    // Disable popups
                    window.alert = () => {};
                    window.confirm = () => true;
                    window.prompt = () => '';
                """)
        except Exception as e:
            fact_logger.logger.warning(f"Page configuration partially failed: {e}")

    async def _navigate_with_strategy(self, page: Page, url: str, strategy: str) -> int:
        """Navigate to URL with strategy-specific wait conditions.

        Returns:
            HTTP status code from navigation response (0 if unknown).
        """
        domain = urlparse(url).netloc.lower()
        base_timeout = self.domain_timeouts.get(domain, self.default_timeout)

        response = None

        if strategy == ScrapingStrategy.BASIC:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=base_timeout)
            await asyncio.sleep(random.uniform(0.5, 1.0))

        elif strategy == ScrapingStrategy.ADVANCED:
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=base_timeout)
            except Exception:
                try:
                    response = await page.goto(url, wait_until="load", timeout=base_timeout)
                except Exception:
                    response = await page.goto(url, wait_until="domcontentloaded", timeout=base_timeout)
            await asyncio.sleep(random.uniform(2.0, 4.0))

        status = response.status if response else 0
        if status in (401, 403):
            fact_logger.logger.warning(
                f"HTTP {status} detected for {url} -- site blocks at HTTP level"
            )
        return status

    async def _simulate_human_behavior(self, page: Page):
        """Simulate human-like behaviors: mouse movement, scrolling"""
        try:
            # Random mouse movement
            await page.mouse.move(
                random.randint(100, 500),
                random.randint(100, 500),
                steps=random.randint(5, 15)
            )

            # Random scroll
            scroll_amount = random.randint(100, 500)
            await page.mouse.wheel(0, scroll_amount)

            # Small delay
            await asyncio.sleep(random.uniform(0.3, 0.8))

            # Another random scroll
            await page.mouse.wheel(0, random.randint(50, 200))

        except Exception as e:
            fact_logger.logger.debug(f"Human behavior simulation error: {e}")

    async def _block_resources(self, route):
        """Block unnecessary resources for faster loading"""
        resource_type = route.request.resource_type
        url = route.request.url

        # Block resource types
        if resource_type in ['image', 'media', 'font']:
            await route.abort()
        # Block tracking and ads
        elif any(blocked in url for blocked in [
            'analytics', 'tracking', 'advertisement', 'ads',
            'facebook.com/tr', 'google-analytics', 'doubleclick'
        ]):
            await route.abort()
        else:
            await route.continue_()

    # ========================================================================
    # PDF EXTRACTION
    # ========================================================================

    def _is_pdf_url(self, url: str) -> bool:
        """
        Check if a URL points to a PDF file.

        Checks the URL path extension. For ambiguous URLs (no extension),
        we let Playwright handle it normally -- the PDF check is a fast path
        for obvious cases like .gov reports, academic papers, etc.
        """
        parsed = urlparse(url)
        path = parsed.path.lower().rstrip('/')
        return path.endswith('.pdf')

    async def _extract_pdf_content(self, url: str) -> str:
        """
        Download a PDF from a URL and extract its text content.

        Uses httpx (async) to download the PDF bytes, then extracts
        text using pypdf (pure Python, no system dependencies).

        Falls back to pypdfium2 if pypdf is not available.

        Args:
            url: Direct URL to a PDF file

        Returns:
            Extracted text content, or empty string on failure
        """
        if not PDF_EXTRACTION_AVAILABLE:
            fact_logger.logger.warning("PDF extraction not available")
            return ""

        self.stats["pdf_extractions"] += 1

        # Step 1: Download PDF bytes
        pdf_bytes = await self._download_pdf_bytes(url)
        if not pdf_bytes:
            return ""

        # Step 2: Extract text from PDF bytes
        try:
            text = self._extract_text_from_pdf_bytes(pdf_bytes)

            if text and len(text.strip()) > 50:
                fact_logger.logger.info(
                    f"PDF text extracted: {len(text)} chars "
                    f"from {len(pdf_bytes) / 1024:.1f} KB file"
                )

                # Step 3: AI-powered cleaning (same as HTML pipeline)
                if self.enable_ai_cleaning and CONTENT_CLEANER_AVAILABLE:
                    try:
                        cleaner = self._get_content_cleaner()
                        if cleaner is not None:
                            cleaning_result = await asyncio.wait_for(
                                cleaner.clean(url, text),
                                timeout=60.0
                            )
                            if (cleaning_result.success
                                    and cleaning_result.cleaned is not None
                                    and cleaning_result.cleaned.body):
                                original_len = len(text)
                                text = cleaning_result.cleaned.body
                                self.stats["ai_cleaned"] += 1
                                fact_logger.logger.info(
                                    f"AI cleaned PDF text: "
                                    f"{original_len} -> {len(text)} chars"
                                )
                    except Exception as e:
                        fact_logger.logger.debug(
                            f"AI cleaning skipped for PDF: {e}"
                        )

                return text
            else:
                fact_logger.logger.warning(
                    f"PDF text extraction returned insufficient content "
                    f"({len(text.strip()) if text else 0} chars) -- "
                    f"PDF may be scanned/image-based"
                )
                return ""

        except Exception as e:
            fact_logger.logger.error(
                f"PDF text extraction failed: {type(e).__name__}: {e}"
            )
            return ""

    async def _download_pdf_bytes(self, url: str) -> Optional[bytes]:
        """
        Download PDF file bytes from a URL.

        Strategy:
          1. Strip tracking query params (utm_source, etc.) from the URL
          2. Direct httpx download (fast, no proxy)
          3. Residential proxy download (bypasses datacenter IP blocks)

        Returns raw PDF bytes or None on failure.
        """
        import httpx

        # Strip tracking query parameters -- ChatGPT/Perplexity append
        # ?utm_source=chatgpt.com to citations; some servers reject these.
        parsed = urlparse(url)
        clean_url = parsed._replace(query="", fragment="").geturl()
        if clean_url != url:
            fact_logger.logger.debug(f"PDF URL cleaned: {url} -> {clean_url}")
            url = clean_url

        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "application/pdf,*/*",
        }

        def _is_valid_pdf(data: bytes, source: str) -> bool:
            if data[:5] == b"%PDF-":
                return True
            fact_logger.logger.warning(
                f"[PDF] {source} returned non-PDF data (magic: {data[:8]})"
            )
            return False

        # --- Attempt 1: direct httpx (fastest, works for most open servers) ---
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=20.0,
                headers=headers,
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "").lower()
                    if "pdf" in content_type or _is_valid_pdf(response.content, "direct"):
                        fact_logger.logger.info(
                            f"[PDF] Downloaded directly: "
                            f"{len(response.content) / 1024:.1f} KB from {url}"
                        )
                        return response.content
                    else:
                        fact_logger.logger.warning(
                            f"[PDF] Direct: unexpected Content-Type '{content_type}'"
                        )
                        return None
                else:
                    fact_logger.logger.warning(
                        f"[PDF] Direct download failed: HTTP {response.status_code} for {url}"
                    )

        except Exception as e:
            fact_logger.logger.warning(
                f"[PDF] Direct download failed ({type(e).__name__}): {e}"
            )

        # --- Attempt 2: residential proxy (bypasses datacenter IP blocks) ---
        if self._residential_proxy and self._residential_proxy.enabled:
            try:
                async with httpx.AsyncClient(
                    proxy=self._residential_proxy.proxy_url,
                    follow_redirects=True,
                    timeout=25.0,
                    verify=False,
                    headers=headers,
                ) as client:
                    response = await client.get(url)

                    if response.status_code == 200:
                        content_type = response.headers.get("content-type", "").lower()
                        if "pdf" in content_type or _is_valid_pdf(response.content, "residential proxy"):
                            fact_logger.logger.info(
                                f"[PDF] Downloaded via residential proxy: "
                                f"{len(response.content) / 1024:.1f} KB from {url}"
                            )
                            return response.content
                    else:
                        fact_logger.logger.warning(
                            f"[PDF] Residential proxy: HTTP {response.status_code} for {url}"
                        )

            except Exception as e:
                fact_logger.logger.warning(
                    f"[PDF] Residential proxy download failed ({type(e).__name__}): {e}"
                )
        else:
            fact_logger.logger.debug("[PDF] Residential proxy not available, skipping")

        fact_logger.logger.warning(f"[PDF] All download attempts failed for {url}")
        return None

    def _extract_text_from_pdf_bytes(self, pdf_bytes: bytes) -> str:
        """
        Extract text from raw PDF bytes using pypdf or pypdfium2.

        Args:
            pdf_bytes: Raw PDF file content

        Returns:
            Extracted text string
        """
        # Try pypdf first (most reliable for text PDFs)
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages_text = []

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())

            if pages_text:
                text = "\n\n".join(pages_text)
                fact_logger.logger.info(
                    f"pypdf extracted text from "
                    f"{len(reader.pages)} pages"
                )
                return text

        except ImportError:
            pass
        except Exception as e:
            fact_logger.logger.debug(
                f"pypdf extraction failed: {e}, trying pypdfium2"
            )

        # Fallback: pypdfium2 (faster but less common)
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(pdf_bytes)
            pages_text = []

            for page in doc:
                textpage = page.get_textpage()
                page_text = textpage.get_text_range()
                if page_text and page_text.strip():
                    pages_text.append(page_text.strip())
                textpage.close()
                page.close()
            doc.close()

            if pages_text:
                text = "\n\n".join(pages_text)
                fact_logger.logger.info(
                    f"pypdfium2 extracted text from "
                    f"{len(pages_text)} pages"
                )
                return text

        except ImportError:
            pass
        except Exception as e:
            fact_logger.logger.debug(
                f"pypdfium2 extraction failed: {e}"
            )

        return ""

    async def _detect_paywall(self, page: Page) -> bool:
        """Quick paywall detection to fail fast"""
        try:
            # Common paywall indicators
            paywall_selectors = [
                '[class*="paywall"]',
                '[class*="subscription"]',
                '[id*="paywall"]',
                '[data-testid*="paywall"]',
                '.gateway-content',
                '.meteredContent',
                '#paywall-container',
                '.piano-offer',
                '.tp-modal',
            ]

            for selector in paywall_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        is_visible = await element.is_visible()
                        if is_visible:
                            fact_logger.logger.warning(f"[LOG] Paywall detected: {selector}")
                            return True
                except Exception:
                    continue

            # Check for very short content
            try:
                body_text = await page.inner_text('body')
                if body_text and len(body_text.strip()) < 500:
                    paywall_keywords = ['subscribe', 'subscription', 'sign in to read', 'become a member', 'premium content']
                    body_lower = body_text.lower()
                    for keyword in paywall_keywords:
                        if keyword in body_lower:
                            fact_logger.logger.warning(f"[LOG] Likely paywall (short content with '{keyword}')")
                            return True
            except Exception:
                pass

            return False

        except Exception as e:
            fact_logger.logger.debug(f"Paywall detection error: {e}")
            return False

    async def _extract_structured_content(self, page: Page, url: str = "") -> str:
        """Extract main content using site-specific and generic selectors"""
        try:
            fact_logger.logger.debug("Extracting structured content")

            # Get site-specific selectors
            selectors = self._get_site_selectors(url) if url else GENERIC_SELECTORS

            # JavaScript function to extract content
            content_data = await page.evaluate(f"""
                () => {{
                    function htmlToStructuredText(element, level = 0) {{
                        if (!element) return '';
                        if (level > 10) return '';

                        let result = '';

                        if (element.nodeType === Node.TEXT_NODE) {{
                            const text = element.textContent.trim();
                            if (text && text.length > 0) {{
                                result += text + ' ';
                            }}
                        }} else if (element.nodeType === Node.ELEMENT_NODE) {{
                            const tagName = element.tagName.toLowerCase();

                            if (['script', 'style', 'noscript', 'head', 'meta', 'link'].includes(tagName)) {{
                                return '';
                            }}

                            switch (tagName) {{
                                case 'h1':
                                case 'h2':
                                case 'h3':
                                case 'h4':
                                case 'h5':
                                case 'h6':
                                    const headingText = element.textContent.trim();
                                    if (headingText) {{
                                        const headingLevel = parseInt(tagName[1]);
                                        const prefix = '#'.repeat(headingLevel);
                                        result += '\\n\\n' + prefix + ' ' + headingText + '\\n';
                                    }}
                                    break;

                                case 'p':
                                    const pText = element.textContent.trim();
                                    if (pText && pText.length > 20) {{
                                        result += '\\n' + pText + '\\n';
                                    }}
                                    break;

                                case 'div':
                                case 'section':
                                case 'article':
                                    for (let child of element.childNodes) {{
                                        result += htmlToStructuredText(child, level + 1);
                                    }}
                                    break;

                                case 'li':
                                    const liText = element.textContent.trim();
                                    if (liText && liText.length > 10) {{
                                        result += '\\n- ' + liText + '\\n';
                                    }}
                                    break;

                                default:
                                    for (let child of element.childNodes) {{
                                        result += htmlToStructuredText(child, level + 1);
                                    }}
                                    break;
                            }}
                        }}

                        return result;
                    }}

                    // Try selectors in order
                    const selectors = {json.dumps(selectors)};

                    let bestContent = '';
                    let bestScore = 0;

                    for (const selector of selectors) {{
                        const element = document.querySelector(selector);
                        if (element) {{
                            const clone = element.cloneNode(true);

                            // Remove unwanted elements
                            const unwantedElements = clone.querySelectorAll(`
                                nav, footer, header, aside,
                                .sidebar, .navigation, .nav, .menu,
                                .cookie, .popup, .modal, .advertisement, .ad,
                                .social-share, .comments, .related-articles,
                                [role="navigation"], [role="banner"], [role="contentinfo"],
                                script, style, noscript
                            `);
                            unwantedElements.forEach(el => el.remove());

                            const structuredText = htmlToStructuredText(clone);

                            // Score based on content quality
                            const headingCount = (structuredText.match(/^#+\\s/gm) || []).length;
                            const paragraphCount = (structuredText.match(/\\n[^\\n#\\-].+\\n/g) || []).length;
                            const wordCount = structuredText.split(/\\s+/).filter(w => w.length > 0).length;

                            let score = 0;
                            score += headingCount * 10;
                            score += paragraphCount * 5;
                            score += Math.min(wordCount, 500);

                            if (score > bestScore && structuredText.length > 100) {{
                                bestScore = score;
                                bestContent = structuredText;
                            }}
                        }}
                    }}

                    return bestContent;
                }}
            """)

            content = content_data if content_data else ""

            fact_logger.logger.debug(
                "Structure extraction complete",
                extra={"content_length": len(content)}
            )

            return content

        except Exception as e:
            fact_logger.logger.error(f"[LOG] Structure extraction error: {e}")
            return ""

    def _clean_content(self, content: str) -> str:
        """Basic regex cleaning of extracted content"""
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Remove common noise patterns
        noise_patterns = [
            r'Cookie.*?(?=\n|$)',
            r'Privacy Policy.*?(?=\n|$)',
            r'Terms.*?Service.*?(?=\n|$)',
            r'Subscribe.*?newsletter.*?(?=\n|$)',
            r'Follow us.*?(?=\n|$)',
            r'Download.*?app.*?(?=\n|$)',
            r'Advertisement\n',
            r'Skip to.*?content',
            r'Accept.*?cookies.*?(?=\n|$)',
            r'Back to top.*?(?=\n|$)',
        ]

        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Final cleanup
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        content = content.strip()

        return content

    def get_stats(self) -> Dict:
        """Return scraping statistics with Supabase domain learning data"""
        stats = self.stats.copy()

        # Add domain strategy information from Supabase
        try:
            all_strategies = self.strategy_service.get_all_strategies()
            stats['learned_domains_count'] = len(all_strategies)
            stats['learned_strategies'] = all_strategies

            fact_logger.logger.debug(
                f"Retrieved {len(all_strategies)} learned strategies from Supabase",
                extra={"strategy_count": len(all_strategies)}
            )
        except Exception as e:
            fact_logger.logger.warning(
                f"Could not retrieve Supabase strategy stats: {e}",
                extra={"error": str(e)}
            )
            stats['learned_domains_count'] = 0
            stats['learned_strategies'] = {}

        return stats

    async def close(self):
        """Properly close browser pool and cleanup"""
        fact_logger.logger.info("[LOG] Shutting down scraper...")

        # Close all contexts
        for i, context in enumerate(self.context_pool):
            try:
                await context.close()
                fact_logger.logger.debug(f"[LOG] Closed context {i}")
            except Exception as e:
                fact_logger.logger.debug(f"Context close error (non-critical): {e}")

        # Close all browsers in pool
        for i, browser in enumerate(self.browser_pool):
            try:
                await browser.close()
                fact_logger.logger.debug(f"[LOG] Closed browser {i}")
            except Exception as e:
                fact_logger.logger.debug(f"Browser close error (non-critical): {e}")

        # Stop Playwright
        if self.playwright:
            try:
                await self.playwright.stop()
                fact_logger.logger.debug("[LOG] Playwright stopped")
            except Exception as e:
                fact_logger.logger.debug(f"Playwright stop error (non-critical): {e}")

        self.session_active = False
        self.browser_pool = []
        self.context_pool = []

        # Print stats
        if self.stats["total_scraped"] > 0:
            success_rate = (self.stats["successful_scrapes"] / self.stats["total_scraped"]) * 100
            fact_logger.logger.info(
                f"Scraping stats: {self.stats['successful_scrapes']}/{self.stats['total_scraped']} "
                f"successful ({success_rate:.1f}%), {self.stats['browser_reuses']} browser reuses"
            )

            # Strategy stats
            fact_logger.logger.info("[LOG] Strategy performance:")
            for strategy, usage in self.stats["strategy_usage"].items():
                if usage > 0:
                    success = self.stats["strategy_success"].get(strategy, 0)
                    rate = (success / usage * 100) if usage > 0 else 0
                    fact_logger.logger.info(f"  {strategy}: {success}/{usage} ({rate:.1f}%)")

            # Site failures
            if self.stats["site_failures"]:
                fact_logger.logger.info("Sites with failures:")
                for domain, count in sorted(
                    self.stats["site_failures"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]:
                    fact_logger.logger.info(f"  {domain}: {count} failures")

        # Log Supabase learning stats on shutdown
        try:
            all_strategies = self.strategy_service.get_all_strategies()
            if all_strategies:
                fact_logger.logger.info(
                    f"Domain learning summary: {len(all_strategies)} strategies saved in Supabase"
                )

                # Count by strategy type
                strategy_counts = {}
                for domain, strategy in all_strategies.items():
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                    fact_logger.logger.info(f"  {strategy}: {count} domains")
        except Exception as e:
            fact_logger.logger.debug(f"Could not retrieve Supabase stats on shutdown: {e}")

        fact_logger.logger.info("[LOG] Scraper shutdown complete")
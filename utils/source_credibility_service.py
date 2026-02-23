# utils/source_credibility_service.py
"""
Source Credibility Service
Unified service for checking publication credibility using:
1. Supabase cache (fastest)
2. MBFC lookup (if not cached)
3. AI-based tier assignment

Integrates with existing PublicationBiasDetector and SupabaseService
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from datetime import datetime
import asyncio

from utils.logger import fact_logger


class CredibilityCheckResult(BaseModel):
    """Result of a credibility check for a publication"""

    # Source identification
    url: str
    domain: str
    publication_name: Optional[str] = None

    # Credibility scores
    credibility_tier: int = 3  # 1 (most credible) to 5 (least credible)
    credibility_rating: Optional[str] = None  # HIGH, MEDIUM, LOW, etc.

    # Bias information
    bias_rating: Optional[str] = None  # LEFT, LEFT-CENTER, CENTER, RIGHT-CENTER, RIGHT
    bias_score: Optional[float] = None  # Numeric bias score

    # Factual reporting
    factual_reporting: Optional[str] = None  # HIGH, MOSTLY FACTUAL, MIXED, LOW, VERY LOW
    factual_score: Optional[float] = None

    # Additional context
    country: Optional[str] = None
    country_freedom_rating: Optional[str] = None
    media_type: Optional[str] = None
    ownership: Optional[str] = None

    # Special flags
    is_propaganda: bool = False
    special_tags: list = Field(default_factory=list)
    failed_fact_checks: list = Field(default_factory=list)

    # Metadata
    source: str = "unknown"  # "supabase", "mbfc", "ai_only", "fallback"
    mbfc_url: Optional[str] = None
    tier_reasoning: Optional[str] = None
    checked_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SourceCredibilityService:
    """
    Unified service for checking source credibility

    Flow:
    1. Extract domain from URL
    2. Check Supabase for cached credibility data
    3. If not found, check propaganda list
    4. If still not found, run MBFC lookup
    5. Save results to Supabase for future use
    6. Return comprehensive CredibilityCheckResult
    """

    def __init__(self, config=None, brave_searcher=None, scraper=None):
        """
        Initialize the service with dependencies

        Args:
            config: Configuration object with API keys
            brave_searcher: BraveSearcher instance for MBFC lookup
            scraper: BrowserlessScraper for scraping MBFC pages
        """
        self.config = config
        self.brave_searcher = brave_searcher
        self.scraper = scraper

        # Initialize Supabase service
        try:
            from utils.supabase_service import get_supabase_service
            self.supabase = get_supabase_service(config)
            self.supabase_enabled = self.supabase.enabled
            if self.supabase_enabled:
                fact_logger.logger.info("Supabase enabled for credibility service")
        except Exception as e:
            fact_logger.logger.warning(f"Supabase not available: {e}")
            self.supabase = None
            self.supabase_enabled = False

        # Initialize MBFC detector
        try:
            from agents.publication_bias_detector import PublicationBiasDetector
            self.mbfc_detector = PublicationBiasDetector(
                config=config,
                brave_searcher=brave_searcher,
                scraper=scraper
            )
            self.mbfc_enabled = brave_searcher is not None and scraper is not None
            if self.mbfc_enabled:
                fact_logger.logger.info("MBFC lookup enabled for credibility service")
        except Exception as e:
            fact_logger.logger.warning(f"MBFC detector not available: {e}")
            self.mbfc_detector = None
            self.mbfc_enabled = False

        # Cache for this session
        self.cache: Dict[str, CredibilityCheckResult] = {}

        fact_logger.logger.info(
            " SourceCredibilityService initialized",
            extra={
                "supabase_enabled": self.supabase_enabled,
                "mbfc_enabled": self.mbfc_enabled
            }
        )

    def _extract_domain(self, url: str) -> str:
        """Extract clean root domain from URL (strips subdomains like 'edition.' or 'www.').

        Uses tldextract if available, otherwise falls back to a pure-Python heuristic
        that handles common cases including country-code TLDs like .co.uk.
        """
        # Try tldextract first (offline snapshot - no network calls)
        try:
            import tldextract
            no_fetch = tldextract.TLDExtract(suffix_list_urls=())
            extracted = no_fetch(url)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}".lower()
        except ImportError:
            pass  # Not installed - use pure-Python fallback below
        except Exception as e:
            fact_logger.logger.debug(f"tldextract error for {url}: {e}")

        # Pure-Python fallback - no external libraries needed
        # Handles: edition.cnn.com -> cnn.com, forums.bbc.co.uk -> bbc.co.uk
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()

            # Strip port if present (e.g. example.com:8080)
            if ':' in host:
                host = host.split(':')[0]

            parts = host.split('.')

            if len(parts) <= 2:
                # Already a root domain (e.g. cnn.com) - nothing to strip
                return host

            # Known second-level TLDs that need 3 parts to form the root domain
            # e.g. bbc.co.uk needs parts [-3:] not just [-2:]
            TWO_LEVEL_TLDS = {
                'co', 'com', 'net', 'org', 'gov', 'edu', 'ac', 'or', 'ne', 'me'
            }

            # If second-to-last part is a known 2-level TLD indicator AND
            # last part is a short country code (2 chars), keep 3 parts
            if len(parts[-1]) == 2 and parts[-2] in TWO_LEVEL_TLDS:
                return '.'.join(parts[-3:])

            # Otherwise keep just the last 2 parts
            return '.'.join(parts[-2:])

        except Exception:
            return ""

    async def check_credibility(
        self, 
        url: str,
        use_cache: bool = True,
        run_mbfc_if_missing: bool = True
    ) -> CredibilityCheckResult:
        """
        Check credibility of a source

        Args:
            url: URL of the source to check
            use_cache: Whether to use in-memory cache
            run_mbfc_if_missing: Whether to run MBFC lookup if not in database

        Returns:
            CredibilityCheckResult with comprehensive credibility data
        """
        domain = self._extract_domain(url)

        if not domain:
            fact_logger.logger.warning(f"Could not extract domain from URL: {url}")
            return CredibilityCheckResult(
                url=url,
                domain="unknown",
                credibility_tier=3,
                source="fallback"
            )

        # Check in-memory cache first
        if use_cache and domain in self.cache:
            fact_logger.logger.debug(f"Using cached credibility for {domain}")
            cached = self.cache[domain]
            # Update URL in cached result
            cached.url = url
            return cached

        fact_logger.logger.info(f"Checking credibility for {domain}")

        result = CredibilityCheckResult(
            url=url,
            domain=domain
        )

        # Step 1: Check Supabase cache
        if self.supabase_enabled:
            db_result = await self._check_supabase(domain)
            if db_result:
                result = db_result
                result.url = url
                result.source = "supabase"
                self.cache[domain] = result
                fact_logger.logger.info(
                    f"Found {domain} in Supabase (Tier {result.credibility_tier})"
                )
                return result

        # Step 2: Check propaganda list
        if self.supabase_enabled:
            is_propaganda = await self._check_propaganda_list(domain)
            if is_propaganda:
                result.is_propaganda = True
                result.credibility_tier = 5
                result.credibility_rating = "LOW CREDIBILITY"
                result.source = "propaganda_list"
                result.tier_reasoning = "Listed in propaganda/disinformation database"
                self.cache[domain] = result
                fact_logger.logger.warning(f" {domain} is flagged as propaganda source")
                return result

        # Step 3: Run MBFC lookup if enabled and requested
        if run_mbfc_if_missing and self.mbfc_enabled:
            mbfc_result = await self._run_mbfc_lookup(domain)
            if mbfc_result:
                result = mbfc_result
                result.url = url
                self.cache[domain] = result
                fact_logger.logger.info(
                    f"MBFC lookup complete for {domain} (Tier {result.credibility_tier})"
                )
                return result

        # Step 4: Fallback - return default tier 3 (unknown)
        result.source = "fallback"
        result.tier_reasoning = "No credibility data found"
        self.cache[domain] = result

        fact_logger.logger.info(f"No credibility data for {domain}, using default tier 3")

        return result

    async def _check_supabase(self, domain: str) -> Optional[CredibilityCheckResult]:
        """
        Check Supabase for cached credibility data

        Args:
            domain: Domain to look up

        Returns:
            CredibilityCheckResult if found, None otherwise
        """
        if not self.supabase or not self.supabase_enabled:
            return None

        try:
            record = self.supabase.get_credibility_by_domain(domain)

            if not record:
                return None

            # Map database fields to result
            names = record.get('names', [])
            publication_name = names[0] if names else domain

            return CredibilityCheckResult(
                url="",  # Will be filled by caller
                domain=domain,
                publication_name=publication_name,
                credibility_tier=record.get('assigned_tier', 3),
                credibility_rating=record.get('mbfc_credibility_rating'),
                bias_rating=record.get('mbfc_bias_rating'),
                bias_score=record.get('mbfc_bias_score'),
                factual_reporting=record.get('mbfc_factual_reporting'),
                factual_score=record.get('mbfc_factual_score'),
                country=record.get('country'),
                country_freedom_rating=record.get('mbfc_country_freedom_rating'),
                media_type=record.get('media_type'),
                ownership=record.get('ownership'),
                special_tags=record.get('mbfc_special_tags', []),
                failed_fact_checks=record.get('failed_fact_checks', []),
                mbfc_url=record.get('mbfc_url'),
                tier_reasoning=record.get('tier_reasoning'),
                source="supabase"
            )

        except Exception as e:
            fact_logger.logger.warning(f"Supabase lookup failed: {e}")
            return None

    async def _check_propaganda_list(self, domain: str) -> bool:
        """
        Check if domain is in propaganda/disinformation list

        Args:
            domain: Domain to check

        Returns:
            True if flagged as propaganda, False otherwise
        """
        if not self.supabase or not self.supabase_enabled:
            return False

        try:
            return self.supabase.is_propaganda_source(domain)
        except Exception as e:
            fact_logger.logger.warning(f"Propaganda check failed: {e}")
            return False

    async def _run_mbfc_lookup(self, domain: str) -> Optional[CredibilityCheckResult]:
        """
        Run MBFC lookup for a domain

        Args:
            domain: Domain to look up

        Returns:
            CredibilityCheckResult if found, None otherwise
        """
        if not self.mbfc_detector or not self.mbfc_enabled:
            return None

        try:
            fact_logger.logger.info(f"Running MBFC lookup for {domain}")

            mbfc_result = await self.mbfc_detector.lookup_mbfc(domain)

            if not mbfc_result:
                return None

            # Convert MBFCResult to CredibilityCheckResult
            result = CredibilityCheckResult(
                url="",
                domain=domain,
                publication_name=mbfc_result.publication_name,
                credibility_rating=mbfc_result.credibility_rating,
                bias_rating=mbfc_result.bias_rating,
                bias_score=mbfc_result.bias_score,
                factual_reporting=mbfc_result.factual_reporting,
                factual_score=mbfc_result.factual_score,
                country=mbfc_result.country,
                country_freedom_rating=mbfc_result.country_freedom_rating,
                media_type=mbfc_result.media_type,
                ownership=mbfc_result.ownership,
                special_tags=mbfc_result.special_tags or [],
                failed_fact_checks=mbfc_result.failed_fact_checks or [],
                mbfc_url=mbfc_result.mbfc_url,
                source="mbfc"
            )

            # Single tier calculation -- same logic used for Supabase storage
            result.credibility_tier = self._calculate_tier(mbfc_result)
            result.tier_reasoning = self._generate_tier_reasoning(mbfc_result, result.credibility_tier)

            # Propaganda flag: only set for truly propaganda-flagged sources
            # "Questionable Source" alone means poor methodology, not necessarily propaganda
            if mbfc_result.special_tags:
                tags_lower = [t.lower() for t in mbfc_result.special_tags]
                if 'propaganda' in tags_lower:
                    result.is_propaganda = True
                elif 'conspiracy-pseudoscience' in tags_lower:
                    result.is_propaganda = True

            return result

        except Exception as e:
            fact_logger.logger.error(f"MBFC lookup failed: {e}")
            return None

    def _calculate_tier(self, mbfc_result) -> int:
        """
        Calculate credibility tier from MBFC data.
        Single source of truth for tier assignment -- matches supabase_service fallback logic.

        Tier Guidelines:
        - Tier 1: HIGH factual + HIGH credibility
        - Tier 2: MOSTLY FACTUAL or HIGH factual with non-LOW credibility
        - Tier 3: MIXED factual or unclear data
        - Tier 4: LOW factual or extreme bias or questionable source tag
        - Tier 5: VERY LOW factual, propaganda, conspiracy-pseudoscience

        Args:
            mbfc_result: MBFCResult object

        Returns:
            Tier number (1-5)
        """
        factual = (mbfc_result.factual_reporting or "").upper()
        credibility = (mbfc_result.credibility_rating or "").upper()
        tags = [t.lower() for t in (mbfc_result.special_tags or [])]

        # Tier 5: conspiracy, propaganda, very low factual reporting
        if 'conspiracy-pseudoscience' in tags or 'propaganda' in tags:
            return 5
        if factual == 'VERY LOW' or credibility == 'LOW CREDIBILITY':
            return 5

        # Tier 4: questionable source, low factual reporting
        if 'questionable source' in tags:
            return 4
        if factual == 'LOW':
            return 4

        # Tier 1: high factual + high credibility
        if factual == 'HIGH' and 'HIGH' in credibility:
            return 1

        # Tier 2: mostly factual or high factual with reasonable credibility
        if factual in ['HIGH', 'MOSTLY FACTUAL'] and 'LOW' not in credibility:
            return 2

        # Tier 3: mixed or unclear
        return 3

    def _generate_tier_reasoning(self, mbfc_result, tier: int) -> str:
        """Generate explanation for tier assignment"""
        parts = []

        if mbfc_result.factual_reporting:
            parts.append(f"Factual reporting: {mbfc_result.factual_reporting}")

        if mbfc_result.credibility_rating:
            parts.append(f"Credibility: {mbfc_result.credibility_rating}")

        if mbfc_result.bias_rating:
            parts.append(f"Bias: {mbfc_result.bias_rating}")

        if mbfc_result.special_tags:
            parts.append(f"Tags: {', '.join(mbfc_result.special_tags)}")

        if parts:
            return f"Tier {tier} based on MBFC: " + "; ".join(parts)

        return f"Tier {tier} assigned based on available data"

    async def check_credibility_batch(
        self, 
        urls: list,
        run_mbfc_if_missing: bool = False  # Disable for batch to avoid slowdown
    ) -> Dict[str, CredibilityCheckResult]:
        """
        Check credibility for multiple URLs

        Args:
            urls: List of URLs to check
            run_mbfc_if_missing: Whether to run MBFC lookup (disabled by default for speed)

        Returns:
            Dict mapping URL to CredibilityCheckResult
        """
        results = {}

        # Deduplicate by domain
        domain_url_map = {}
        for url in urls:
            domain = self._extract_domain(url)
            if domain and domain not in domain_url_map:
                domain_url_map[domain] = url

        fact_logger.logger.info(
            f"Batch credibility check for {len(domain_url_map)} unique domains"
        )

        # Check all in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(10)

        async def check_with_semaphore(url: str):
            async with semaphore:
                return url, await self.check_credibility(
                    url, 
                    run_mbfc_if_missing=run_mbfc_if_missing
                )

        tasks = [
            check_with_semaphore(url)
            for url in domain_url_map.values()
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                url, result = await coro
                domain = self._extract_domain(url)
                # Map result to all URLs with this domain
                for orig_url in urls:
                    if self._extract_domain(orig_url) == domain:
                        result_copy = result.model_copy()
                        result_copy.url = orig_url
                        results[orig_url] = result_copy
            except Exception as e:
                fact_logger.logger.error(f"Batch check error: {e}")

        return results

    def get_tier_description(self, tier: int) -> str:
        """Get human-readable description for a tier"""
        descriptions = {
            1: "Highly Credible - Official sources, major wire services, highly reputable news",
            2: "Credible - Reputable mainstream media with strong factual reporting",
            3: "Mixed - Requires verification, may have bias or mixed factual reporting",
            4: "Low Credibility - Significant bias issues or poor factual reporting",
            5: "Unreliable - Propaganda, conspiracy, or known disinformation source"
        }
        return descriptions.get(tier, "Unknown credibility level")


# Factory function
def get_credibility_service(config=None, brave_searcher=None, scraper=None) -> SourceCredibilityService:
    """Get a credibility service instance"""
    return SourceCredibilityService(
        config=config,
        brave_searcher=brave_searcher,
        scraper=scraper
    )


# Test function
if __name__ == "__main__":
    import asyncio

    print("Testing Source Credibility Service\n")

    service = SourceCredibilityService()

    test_urls = [
        "https://www.nytimes.com/article",
        "https://www.reuters.com/story",
        "https://www.infowars.com/fake-news"
    ]

    async def test():
        for url in test_urls:
            result = await service.check_credibility(url, run_mbfc_if_missing=False)
            print(f"\n{result.domain}:")
            print(f"Tier: {result.credibility_tier}")
            print(f"Source: {result.source}")
            print(f"Propaganda: {result.is_propaganda}")

    asyncio.run(test())
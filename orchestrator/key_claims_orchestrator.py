# orchestrator/key_claims_orchestrator.py
"""
Key Claims Orchestrator - WITH PARALLEL PROCESSING
Extracts and verifies ONLY the 2-3 central thesis claims from text

OPTIMIZED: Full parallel processing for all stages
   - Parallel query generation
   - Parallel web searches (paid Brave account)
   - Parallel credibility filtering
   - Parallel scraping
   - Parallel verification
   - ~60-70% faster than sequential processing

Pipeline:
1. Extract 2-3 key claims (central thesis statements)
2. Generate search queries for each key claim (PARALLEL)
3. Execute web searches via Brave (PARALLEL)
4. Filter results by source credibility (PARALLEL)
5. Scrape credible sources (PARALLEL)
6. Verify each key claim against sources (PARALLEL)
7. Generate detailed verification report
8. Save comprehensive search audit
"""

from langsmith import traceable
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple

from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.file_manager import FileManager
from utils.job_manager import job_manager
from utils.browserless_scraper import BrowserlessScraper
from utils.brave_searcher import BraveSearcher

# Import key claims extractor
from agents.key_claims_extractor import KeyClaimsExtractor, ContentLocation

# Import existing agents
from agents.fact_checker import FactChecker, FactCheckResult
from agents.query_generator import QueryGenerator
from agents.credibility_filter import CredibilityFilter
from agents.highlighter import Highlighter

# Import search audit utilities
from utils.search_audit_builder import (
    build_session_search_audit,
    build_fact_search_audit,
    build_query_audit,
    save_search_audit,
    upload_search_audit_to_r2
)


class KeyClaimsOrchestrator:
    """
    Orchestrator for key claims extraction and verification

    Extracts only 2-3 central thesis claims and verifies them thoroughly.

    OPTIMIZED: Uses parallel processing for all claim operations
    """

    def __init__(self, config):
        self.config = config

        # Initialize agents
        self.extractor = KeyClaimsExtractor(config)
        self.query_generator = QueryGenerator(config)
        self.searcher = BraveSearcher(config, max_results=5)
        self.credibility_filter = CredibilityFilter(config, min_credibility_score=0.70)
        # NOTE: Don't create scraper here - it binds asyncio.Lock to wrong event loop
        self.highlighter = Highlighter(config)
        self.checker = FactChecker(config)
        self.file_manager = FileManager()

        # Configuration
        self.max_sources_per_claim = 15  # More sources per claim since fewer claims

        # Initialize R2 uploader for audit upload
        try:
            from utils.r2_uploader import R2Uploader
            self.r2_uploader = R2Uploader()
            self.r2_enabled = True
            fact_logger.logger.info("R2 uploader initialized for search audits")
        except Exception as e:
            self.r2_enabled = False
            self.r2_uploader = None
            fact_logger.logger.warning(f"R2 not available for audits: {e}")

        fact_logger.log_component_start(
            "KeyClaimsOrchestrator",
            max_sources_per_claim=self.max_sources_per_claim,
            parallel_mode=True
        )

    def _get_credibility_label(self, avg_score: float) -> str:
        """Convert average score to credibility label for frontend"""
        if avg_score >= 0.9:
            return "High"
        elif avg_score >= 0.7:
            return "Medium"
        elif avg_score >= 0.5:
            return "Low"
        else:
            return "Very Low"

    def _check_cancellation(self, job_id: str):
        """Check if job has been cancelled"""
        if job_manager.is_cancelled(job_id):
            raise Exception("Job cancelled by user")

    @traceable(
        name="key_claims_verification",
        run_type="chain",
        tags=["key-claims", "thesis-verification", "parallel"]
    )
    async def process_with_progress(
        self,
        text_content: str,
        job_id: str,
        source_context: Optional[Dict[str, Any]] = None,
        source_credibility: Optional[Dict[str, Any]] = None,
        standalone: bool = True,
        shared_scraper=None
    ) -> dict:
        """
        Complete key claims verification pipeline with parallel processing

        OPTIMIZED: All claim operations run in parallel

        Args:
            text_content: Plain text to verify
            job_id: Job ID for progress tracking
            source_context: Optional content classification data
            source_credibility: Optional source credibility data
            standalone: If True, calls complete_job on finish
            shared_scraper: Optional shared ScrapeCache from comprehensive mode.
                           If provided, uses it instead of creating a new scraper.
                           The caller is responsible for closing it.
        """
        session_id = self.file_manager.create_session()
        # Track credibility usage
        using_credibility = source_credibility is not None
        credibility_tier = int(source_credibility.get('tier', 0)) if source_credibility else None
        start_time = time.time()

        # Initialize session search audit
        session_audit = None

        try:
            # ================================================================
            # STAGE 0: Log Source Context (NEW)
            # ================================================================
            if source_credibility:
                tier = source_credibility.get('tier', '?')
                bias = source_credibility.get('bias_rating', 'Unknown')
                factual = source_credibility.get('factual_reporting', 'Unknown')

                job_manager.add_progress(
                    job_id, 
                    f"Source context: Tier {tier} | {bias} bias | {factual} factual reporting"
                )

                if credibility_tier and int(credibility_tier) >= 4:
                    job_manager.add_progress(
                        job_id,
                        "Low credibility source - claims require extra verification"
                    )
            elif source_context and source_context.get('publication_name'):
                job_manager.add_progress(
                    job_id,
                    f"Analyzing: {source_context.get('publication_name')}"
                )


            # ================================================================
            # STAGE 1: Extract Key Claims (Sequential - single LLM call)
            # ================================================================
            job_manager.add_progress(job_id, "Extracting key claims from text...")
            self._check_cancellation(job_id)

            # Prepare parsed content for extractor
            parsed_content = {
                'text': text_content,
                'links': [],
                'format': 'plain_text'
            }

            claims, all_sources, content_location, broad_context, media_sources, query_instructions = await self.extractor.extract(parsed_content)

            # NEW: Check if no claims were extracted
            if not claims or len(claims) == 0:
                processing_time = time.time() - start_time

                # Get the reasoning from broad_context if available
                reason = "No verifiable factual claims found in this content."
                if broad_context and hasattr(broad_context, 'reasoning') and broad_context.reasoning:
                    reason = broad_context.reasoning

                job_manager.add_progress(job_id, f"{reason}")

                result = {
                    "success": True,  # Not an error, just no claims
                    "session_id": session_id,
                    "key_claims": [],
                    "summary": {
                        "total_claims": 0,
                        "verified": 0,
                        "partially_verified": 0,
                        "unverified": 0,
                        "overall_credibility": "N/A",
                        "average_score": 0,
                        "message": reason
                    },
                    "processing_time": processing_time,
                    "methodology": "key_claims_verification",
                    "source_context": source_context,
                    "source_credibility": source_credibility,
                    "content_location": {
                        "country": content_location.country if content_location else "international",
                        "language": content_location.language if content_location else "english"
                    },
                    "no_claims_found": True  # Flag for frontend
                }

                if standalone:
                    job_manager.complete_job(job_id, result)
                return result

            # Log content analysis results
            fact_logger.logger.info(
                "Content Analysis:",
                extra={
                    "content_type": broad_context.content_type,
                    "credibility": broad_context.credibility_assessment,
                    "num_media_sources": len(media_sources),
                    "primary_strategy": query_instructions.primary_strategy
                }
            )

            if not claims:
                job_manager.add_progress(job_id, "No key claims found")
                return {
                    "success": True,
                    "session_id": session_id,
                    "claims": [],
                    "summary": {"message": "No key claims identified"},
                    "processing_time": time.time() - start_time
                }

            job_manager.add_progress(job_id, f"Found {len(claims)} key claims")

            # Initialize session audit with content location
            session_audit = build_session_search_audit(
                session_id=session_id,
                pipeline_type="key_claims",
                content_country=content_location.country if content_location else "international",
                content_language=content_location.language if content_location else "english"
            )

            # ================================================================
            # STAGE 2: Generate Search Queries (PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Generating search queries in parallel...")
            self._check_cancellation(job_id)

            query_gen_start = time.time()

            # Create query generation tasks for ALL claims
            async def generate_queries_for_claim(claim):
                """Generate queries for a single claim"""
                fact_like = type('Fact', (), {
                    'id': claim.id,
                    'statement': claim.statement
                })()

                queries = await self.query_generator.generate_queries(
                    fact=fact_like,
                    context="",
                    content_location=content_location,
                    publication_date=None,
                    broad_context=broad_context,
                    media_sources=media_sources,
                    query_instructions=query_instructions
                )
                return (claim.id, queries)

            query_tasks = [generate_queries_for_claim(claim) for claim in claims]
            query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

            # Process query results
            all_queries_by_claim = {}
            freshness_by_claim = {}

            for result in query_results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Query generation error: {result}")
                    continue
                claim_id, queries = result
                all_queries_by_claim[claim_id] = queries
                freshness_by_claim[claim_id] = queries.recommended_freshness

            query_gen_duration = time.time() - query_gen_start
            total_queries = sum(len(q.all_queries) for q in all_queries_by_claim.values())
            job_manager.add_progress(
                job_id, 
                f"Generated {total_queries} queries in {query_gen_duration:.1f}s"
            )

            # ================================================================
            # STAGE 3: Execute Web Searches (PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Searching the web in parallel...")
            self._check_cancellation(job_id)

            search_start = time.time()

            # Create search tasks for ALL claims
            async def search_for_claim(claim):
                """Execute all searches for a single claim"""
                queries = all_queries_by_claim.get(claim.id)
                if not queries:
                    return (claim.id, {}, [])

                search_results = await self.searcher.search_multiple(
                    queries=queries.all_queries,
                    search_depth="advanced",
                    max_concurrent=3,  # Aggressive with paid Brave
                    freshness=None
                )

                # Build query audits
                query_audits = []
                for query, brave_results in search_results.items():
                    qa = build_query_audit(
                        query=query,
                        brave_results=brave_results,
                        query_type="search",
                        language=queries.local_language_used or "en"
                    )
                    query_audits.append(qa)

                return (claim.id, search_results, query_audits)

            search_tasks = [search_for_claim(claim) for claim in claims]
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process search results
            search_results_by_claim = {}
            query_audits_by_claim = {}
            total_results = 0

            for result in search_results_list:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Search error: {result}")
                    continue
                claim_id, search_results, query_audits = result
                search_results_by_claim[claim_id] = search_results
                query_audits_by_claim[claim_id] = query_audits
                for brave_results in search_results.values():
                    total_results += len(brave_results.results)

            search_duration = time.time() - search_start
            job_manager.add_progress(
                job_id, 
                f"Found {total_results} potential sources in {search_duration:.1f}s"
            )

            # ================================================================
            # STAGE 4: Filter by Credibility (PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Filtering sources by credibility in parallel...")
            self._check_cancellation(job_id)

            filter_start = time.time()

            # Create credibility filter tasks for ALL claims
            async def filter_sources_for_claim(claim):
                """Filter sources for a single claim"""
                search_results = search_results_by_claim.get(claim.id, {})

                all_results_for_claim = []
                for query, results in search_results.items():
                    all_results_for_claim.extend(results.results)

                if not all_results_for_claim:
                    return (claim.id, [], {}, None)

                fact_like = type('Fact', (), {
                    'id': claim.id,
                    'statement': claim.statement
                })()

                credibility_results = await self.credibility_filter.evaluate_sources(
                    fact=fact_like,
                    search_results=all_results_for_claim
                )

                credible_sources = credibility_results.get_top_sources(self.max_sources_per_claim)
                credible_urls = [s.url for s in credible_sources]
                source_metadata = credibility_results.get_source_metadata_dict()

                return (claim.id, credible_urls, source_metadata, credibility_results)

            filter_tasks = [filter_sources_for_claim(claim) for claim in claims]
            filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)

            # Process filter results
            credible_urls_by_claim = {}
            source_metadata_by_claim = {}
            credibility_results_by_claim = {}

            for result in filter_results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Credibility filter error: {result}")
                    continue
                claim_id, credible_urls, source_metadata, cred_results = result
                credible_urls_by_claim[claim_id] = credible_urls
                source_metadata_by_claim[claim_id] = source_metadata
                credibility_results_by_claim[claim_id] = cred_results

            filter_duration = time.time() - filter_start
            total_credible = sum(len(urls) for urls in credible_urls_by_claim.values())
            job_manager.add_progress(
                job_id, 
                f"Found {total_credible} credible sources in {filter_duration:.1f}s"
            )

            # ================================================================
            # STAGE 5: Scrape Sources (PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, f"Scraping {total_credible} credible sources in parallel...")
            self._check_cancellation(job_id)

            scrape_start = time.time()

            # Use shared scraper (from comprehensive mode) or create a new one
            using_shared_scraper = shared_scraper is not None
            scraper = shared_scraper if shared_scraper else BrowserlessScraper(self.config)

            # Collect ALL URLs to scrape across all claims
            all_urls_to_scrape = []
            url_to_claim_map = {}  # Track which claim each URL belongs to

            for claim in claims:
                urls = credible_urls_by_claim.get(claim.id, [])
                for url in urls:
                    if url not in url_to_claim_map:
                        all_urls_to_scrape.append(url)
                        url_to_claim_map[url] = []
                    url_to_claim_map[url].append(claim.id)

            # Scrape all URLs at once (browser pool handles concurrency)
            all_scraped_content = await scraper.scrape_urls_for_facts(all_urls_to_scrape)

            # Organize scraped content by claim
            scraped_content_by_claim = {}
            scraped_urls_by_claim = {}
            scrape_errors_by_claim = {}

            for claim in claims:
                claim_urls = credible_urls_by_claim.get(claim.id, [])
                scraped_content_by_claim[claim.id] = {
                    url: all_scraped_content.get(url)
                    for url in claim_urls
                    if url in all_scraped_content
                }
                scraped_urls_by_claim[claim.id] = [
                    url for url in claim_urls
                    if all_scraped_content.get(url)
                ]
                scrape_errors_by_claim[claim.id] = {
                    url: "Scrape failed or empty content"
                    for url in claim_urls
                    if not all_scraped_content.get(url)
                }

            scrape_duration = time.time() - scrape_start
            successful_scrapes = len([v for v in all_scraped_content.values() if v])
            job_manager.add_progress(
                job_id, 
                f"Scraped {successful_scrapes}/{len(all_urls_to_scrape)} sources in {scrape_duration:.1f}s"
            )

            # Build claim search audits
            for claim in claims:
                claim_audit = build_fact_search_audit(
                    fact_id=claim.id,
                    fact_statement=claim.statement,
                    query_audits=query_audits_by_claim.get(claim.id, []),
                    credibility_results=credibility_results_by_claim.get(claim.id),
                    scraped_urls=scraped_urls_by_claim.get(claim.id, []),
                    scrape_errors=scrape_errors_by_claim.get(claim.id, {})
                )
                session_audit.add_fact_audit(claim_audit)

            # ================================================================
            # STAGE 6: Verify Claims (PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, f"Verifying {len(claims)} key claims in parallel...")
            self._check_cancellation(job_id)

            verify_start = time.time()

            # Create verification tasks for ALL claims
            async def verify_single_claim(claim):
                """Verify a single claim"""
                try:
                    scraped_content = scraped_content_by_claim.get(claim.id, {})
                    source_metadata = source_metadata_by_claim.get(claim.id, {})

                    if not scraped_content or not any(scraped_content.values()):
                        return FactCheckResult(
                            fact_id=claim.id,
                            statement=claim.statement,
                            match_score=0.0,
                            confidence=0.0,
                            report="Unable to verify - no credible sources found. Web search did not return any Tier 1 or Tier 2 sources for this key claim."
                        )

                    # Extract relevant excerpts
                    fact_like = type('Fact', (), {
                        'id': claim.id,
                        'statement': claim.statement
                    })()

                    excerpts = await self.highlighter.highlight(
                        fact=fact_like,
                        scraped_content=scraped_content
                    )

                    # Check the claim
                    result = await self.checker.check_fact(
                        fact=fact_like,
                        excerpts=excerpts,
                        source_metadata=source_metadata
                    )

                    # Progress update
                    job_manager.add_progress(
                        job_id,
                        f"{claim.id}: {result.match_score:.0%} verified"
                    )

                    return result

                except Exception as e:
                    fact_logger.logger.error(f"Verification error for {claim.id}: {e}")
                    return FactCheckResult(
                        fact_id=claim.id,
                        statement=claim.statement,
                        match_score=0.0,
                        confidence=0.0,
                        report=f"Verification error: {str(e)}"
                    )

            verify_tasks = [verify_single_claim(claim) for claim in claims]
            results = await asyncio.gather(*verify_tasks, return_exceptions=True)

            # Process verification results
            final_results = []
            for result in results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Verification exception: {result}")
                    continue
                final_results.append(result)

            verify_duration = time.time() - verify_start
            job_manager.add_progress(
                job_id, 
                f"Verification complete in {verify_duration:.1f}s"
            )

            # Clean up scraper only if we created it (not shared)
            if not using_shared_scraper:
                try:
                    await scraper.close()
                except Exception:
                    pass

            # ================================================================
            # STAGE 7: Generate Summary and Save Audit
            # ================================================================
            processing_time = time.time() - start_time

            # Save search audit
            job_manager.add_progress(job_id, "Saving search audit...")

            audit_file_path = save_search_audit(
                session_audit=session_audit,
                file_manager=self.file_manager,
                session_id=session_id,
                filename="search_audit.json"
            )

            # Upload audit to R2 if available
            audit_r2_url = None
            if self.r2_enabled and self.r2_uploader:
                audit_r2_url = await upload_search_audit_to_r2(
                    session_audit=session_audit,
                    session_id=session_id,
                    r2_uploader=self.r2_uploader,
                    pipeline_type="key-claims"
                )

            job_manager.add_progress(job_id, f"Complete in {processing_time:.1f}s")

            # Build summary with keys that frontend expects
            frontend_summary = {
                "total_key_claims": len(claims),
                "verified_count": len([r for r in final_results if r.match_score >= 0.9]),
                "partial_count": len([r for r in final_results if 0.7 <= r.match_score < 0.9]),
                "unverified_count": len([r for r in final_results if r.match_score < 0.7]),
                "overall_credibility": self._get_credibility_label(
                    sum(r.match_score for r in final_results) / len(final_results) if final_results else 0
                ),
                "average_score": sum(r.match_score for r in final_results) / len(final_results) if final_results else 0
            }

            # Log performance metrics
            fact_logger.logger.info(
                "Key Claims Pipeline Performance",
                extra={
                    "total_time": round(processing_time, 2),
                    "query_gen_time": round(query_gen_duration, 2),
                    "search_time": round(search_duration, 2),
                    "filter_time": round(filter_duration, 2),
                    "scrape_time": round(scrape_duration, 2),
                    "verify_time": round(verify_duration, 2),
                    "num_claims": len(claims),
                    "parallel_mode": True
                }
            )

            # BUILD RESULT DICT
            result = {
                "success": True,
                "session_id": session_id,
                "key_claims": [
                    {
                        "id": r.fact_id,
                        "statement": r.statement,
                        "match_score": r.match_score,
                        "confidence": r.confidence,
                        "report": r.report,
                        "tier_breakdown": r.tier_breakdown if hasattr(r, 'tier_breakdown') else None
                    }
                    for r in final_results
                ],
                "summary": frontend_summary,
                "processing_time": processing_time,
                "methodology": "key_claims_verification",
                "source_context": source_context,
                "source_credibility": source_credibility,
                "used_source_credibility": using_credibility,
                "content_location": {
                    "country": content_location.country,
                    "language": content_location.language
                } if content_location else None,
                "statistics": {
                    "claims_extracted": len(claims),
                    "queries_generated": total_queries,
                    "raw_results_found": total_results,
                    "credible_sources": total_credible,
                    "claims_verified": len(final_results)
                },
                "audit": {
                    "local_path": audit_file_path,
                    "r2_url": audit_r2_url,
                    "summary": {
                        "total_raw_results": session_audit.total_raw_results,
                        "total_credible": session_audit.total_credible_sources,
                        "total_filtered": session_audit.total_filtered_sources,
                        "tier_breakdown": {
                            "tier1": session_audit.total_tier1,
                            "tier2": session_audit.total_tier2,
                            "tier3_filtered": session_audit.total_tier3_filtered
                        }
                    }
                },
                "r2_upload": {
                    "success": audit_r2_url is not None,
                    "url": audit_r2_url
                },
                "performance": {
                    "query_generation": round(query_gen_duration, 2),
                    "web_search": round(search_duration, 2),
                    "credibility_filter": round(filter_duration, 2),
                    "scraping": round(scrape_duration, 2),
                    "verification": round(verify_duration, 2)
                }
            }

            # Only mark job complete in standalone mode (not when run from comprehensive)
            if standalone:
                job_manager.complete_job(job_id, result)

            return result

        except Exception as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                job_manager.add_progress(job_id, "Verification cancelled")
                if standalone:
                    job_manager.fail_job(job_id, "Cancelled by user")
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": "Cancelled by user",
                    "processing_time": time.time() - start_time
                }

            fact_logger.logger.error(f"Key claims orchestrator error: {e}")
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
            job_manager.add_progress(job_id, f"Error: {error_msg}")

            # Try to save partial audit even on error
            if session_audit and session_audit.total_facts > 0:
                try:
                    save_search_audit(
                        session_audit=session_audit,
                        file_manager=self.file_manager,
                        session_id=session_id,
                        filename="search_audit_partial.json"
                    )
                except Exception:
                    pass

            if standalone:
                job_manager.fail_job(job_id, error_msg)

            return {
                "success": False,
                "session_id": session_id,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
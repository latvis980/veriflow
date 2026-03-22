# orchestrator/web_search_orchestrator.py
"""
Web Search Orchestrator - WITH FULL PARALLEL PROCESSING
Coordinates web search-based fact verification pipeline for text without links

 OPTIMIZED: Full parallel processing for all stages
   - Parallel query generation
   - Parallel web searches (paid Brave account)
   - Parallel credibility filtering
   - Parallel scraping (batch mode)
   - Parallel verification
   - ~60-70% faster than sequential processing

Pipeline:
1. Extract facts from plain text (with country/language detection)
2. Generate search queries for each fact ( PARALLEL)
3. Execute web searches via Brave ( PARALLEL)
4. Filter results by source credibility ( PARALLEL)
5. Scrape credible sources ( PARALLEL - batch mode)
6. Verify facts against sources ( PARALLEL)
7. Save comprehensive search audit
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

# Import agents
from agents.fact_extractor import FactAnalyzer, ContentLocation
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


class WebSearchOrchestrator:
    """
    Orchestrator for web search-based fact verification

    For plain text input without provided sources
    Supports multi-language queries for non-English content

     OPTIMIZED: Uses parallel processing for all fact operations
    """

    def __init__(self, config):
        self.config = config

        # Initialize all agents
        self.analyzer = FactAnalyzer(config)
        self.query_generator = QueryGenerator(config)
        self.searcher = BraveSearcher(config, max_results=5)
        self.credibility_filter = CredibilityFilter(config, min_credibility_score=0.70)
        # NOTE: Don't create scraper here - it binds asyncio.Lock to wrong event loop
        self.highlighter = Highlighter(config)
        self.checker = FactChecker(config)
        self.file_manager = FileManager()

        # Configuration
        self.max_sources_per_fact = 10  # Maximum sources to scrape per fact

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
            "WebSearchOrchestrator",
            max_sources_per_fact=self.max_sources_per_fact,
            parallel_mode=True
        )

    def _check_cancellation(self, job_id: str):
        """Check if job has been cancelled and raise exception if so"""
        job = job_manager.get_job(job_id)
        if job and job.get('status') == 'cancelled':
            raise Exception("Job cancelled by user")

    def _create_empty_result(self, session_id: str, message: str) -> dict:
        """Create an empty result for cases with no facts"""
        return {
            "success": True,
            "session_id": session_id,
            "facts": [],
            "summary": {"message": message},
            "processing_time": 0,
            "methodology": "web_search_verification",
            "statistics": {}
        }

    def _generate_summary(self, results: list) -> dict:
        """Generate summary statistics from results"""
        if not results:
            return {"message": "No results to summarize"}

        scores = [r.match_score for r in results]
        return {
            "total_facts": len(results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "verified_count": len([r for r in results if r.match_score >= 0.9]),
            "partial_count": len([r for r in results if 0.7 <= r.match_score < 0.9]),
            "unverified_count": len([r for r in results if r.match_score < 0.7])
        }

    @traceable(
        name="web_search_fact_verification",
        run_type="chain",
        tags=["web-search", "fact-checking", "parallel"]
    )
    async def process_with_progress(
        self,
        text_content: str,
        job_id: str,
        shared_scraper=None
    ) -> dict:
        """
        Process with full parallel processing and search audit

        OPTIMIZED: All fact operations run in parallel

        Args:
            text_content: Plain text to fact-check
            job_id: Job ID for progress tracking
            shared_scraper: Optional shared ScrapeCache from comprehensive mode.
                           If provided, uses it instead of creating a new scraper.
                           The caller is responsible for closing it.
        """
        session_id = self.file_manager.create_session()
        start_time = time.time()

        # Initialize session search audit
        session_audit = None
        content_location = None

        try:
            # ================================================================
            # STAGE 1: Extract Facts (Sequential - single LLM call)
            # ================================================================
            job_manager.add_progress(job_id, "Extracting facts from text...")
            self._check_cancellation(job_id)

            parsed_input = {
                'text': text_content,
                'links': [],
                'format': 'plain_text'
            }

            facts, _, content_location = await self.analyzer.analyze(parsed_input)

            if not facts:
                job_manager.add_progress(job_id, "No verifiable facts found")
                return self._create_empty_result(session_id, "No verifiable facts found in text")

            job_manager.add_progress(job_id, f"Extracted {len(facts)} facts")

            # Initialize session audit with content location
            session_audit = build_session_search_audit(
                session_id=session_id,
                pipeline_type="web_search",
                content_country=content_location.country if content_location else "international",
                content_language=content_location.language if content_location else "english"
            )

            # Log detected location
            if content_location and content_location.country != "international":
                if content_location.language != "english":
                    job_manager.add_progress(
                        job_id, 
                        f"Detected location: {content_location.country} ({content_location.language}) - will include local language queries"
                    )
                else:
                    job_manager.add_progress(
                        job_id, 
                        f"Detected location: {content_location.country} (English)"
                    )

            # ================================================================
            # STAGE 2: Generate Search Queries ( PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Generating search queries in parallel...")
            self._check_cancellation(job_id)

            query_gen_start = time.time()

            # Create query generation tasks for ALL facts
            async def generate_queries_for_fact(fact):
                """Generate queries for a single fact"""
                queries = await self.query_generator.generate_queries(
                    fact,
                    content_location=content_location
                )
                return (fact.id, queries)

            query_tasks = [generate_queries_for_fact(fact) for fact in facts]
            query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

            # Process query results
            all_queries_by_fact = {}
            for result in query_results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Query generation error: {result}")
                    continue
                fact_id, queries = result
                all_queries_by_fact[fact_id] = queries

            query_gen_duration = time.time() - query_gen_start
            total_queries = sum(len(q.all_queries) for q in all_queries_by_fact.values())
            job_manager.add_progress(
                job_id, 
                f"Generated {total_queries} queries in {query_gen_duration:.1f}s"
            )

            # ================================================================
            # STAGE 3: Execute Web Searches ( PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Searching the web in parallel...")
            self._check_cancellation(job_id)

            search_start = time.time()

            # Create search tasks for ALL facts
            async def search_for_fact(fact):
                """Execute all searches for a single fact"""
                queries = all_queries_by_fact.get(fact.id)
                if not queries:
                    return (fact.id, {}, [])

                search_results = await self.searcher.search_multiple(
                    queries=queries.all_queries,
                    search_depth="advanced",
                    max_concurrent=3 # Aggressive with paid Brave
                )

                # Build query audits
                query_audits = []
                for query, brave_results in search_results.items():
                    # Determine query type
                    query_type = "english"
                    if queries.local_queries and query in queries.local_queries:
                        query_type = "local_language"
                    elif queries.fallback_query and query == queries.fallback_query:
                        query_type = "fallback"

                    qa = build_query_audit(
                        query=query,
                        brave_results=brave_results,
                        query_type=query_type,
                        language=content_location.language if content_location else "en"
                    )
                    query_audits.append(qa)

                return (fact.id, search_results, query_audits)

            search_tasks = [search_for_fact(fact) for fact in facts]
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Process search results
            search_results_by_fact = {}
            query_audits_by_fact = {}
            total_results = 0

            for result in search_results_list:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Search error: {result}")
                    continue
                fact_id, search_results, query_audits = result
                search_results_by_fact[fact_id] = search_results
                query_audits_by_fact[fact_id] = query_audits
                for brave_results in search_results.values():
                    total_results += len(brave_results.results)

            search_duration = time.time() - search_start
            job_manager.add_progress(
                job_id, 
                f"Found {total_results} potential sources in {search_duration:.1f}s"
            )

            # ================================================================
            # STAGE 4: Filter by Credibility ( PARALLEL)
            # ================================================================
            job_manager.add_progress(job_id, "Filtering sources by credibility in parallel...")
            self._check_cancellation(job_id)

            filter_start = time.time()

            # Create credibility filter tasks for ALL facts
            async def filter_sources_for_fact(fact):
                """Filter sources for a single fact"""
                search_results = search_results_by_fact.get(fact.id, {})

                all_results_for_fact = []
                for query, results in search_results.items():
                    all_results_for_fact.extend(results.results)

                if not all_results_for_fact:
                    return (fact.id, [], None)

                credibility_results = await self.credibility_filter.evaluate_sources(
                    fact=fact,
                    search_results=all_results_for_fact
                )

                credible_sources = credibility_results.get_top_sources(self.max_sources_per_fact)
                credible_urls = [s.url for s in credible_sources]

                return (fact.id, credible_urls, credibility_results)

            filter_tasks = [filter_sources_for_fact(fact) for fact in facts]
            filter_results = await asyncio.gather(*filter_tasks, return_exceptions=True)

            # Process filter results
            credible_urls_by_fact = {}
            credibility_results_by_fact = {}

            for result in filter_results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Credibility filter error: {result}")
                    continue
                fact_id, credible_urls, cred_results = result
                credible_urls_by_fact[fact_id] = credible_urls
                credibility_results_by_fact[fact_id] = cred_results

            filter_duration = time.time() - filter_start
            total_credible = sum(len(urls) for urls in credible_urls_by_fact.values())
            job_manager.add_progress(
                job_id, 
                f"Found {total_credible} credible sources in {filter_duration:.1f}s"
            )

            # ================================================================
            # STAGE 5: Scrape Sources ( PARALLEL - Batch Mode)
            # ================================================================
            job_manager.add_progress(job_id, f"Scraping {total_credible} sources in parallel...")
            self._check_cancellation(job_id)

            scrape_start = time.time()

            # Use shared scraper (from comprehensive mode) or create a new one
            using_shared_scraper = shared_scraper is not None
            scraper = shared_scraper if shared_scraper else BrowserlessScraper(self.config)

            # Collect ALL URLs to scrape across all facts
            all_urls_to_scrape = []
            url_to_fact_map = {}  # Track which fact each URL belongs to

            for fact in facts:
                urls = credible_urls_by_fact.get(fact.id, [])
                for url in urls:
                    if url not in url_to_fact_map:
                        all_urls_to_scrape.append(url)
                        url_to_fact_map[url] = []
                    url_to_fact_map[url].append(fact.id)

            # Scrape all URLs at once (browser pool handles concurrency)
            all_scraped_content = await scraper.scrape_urls_for_facts(all_urls_to_scrape)

            # Organize scraped content by fact
            scraped_content_by_fact = {}
            scraped_urls_by_fact = {}
            scrape_errors_by_fact = {}

            for fact in facts:
                fact_urls = credible_urls_by_fact.get(fact.id, [])
                scraped_content_by_fact[fact.id] = {
                    url: all_scraped_content.get(url)
                    for url in fact_urls
                    if url in all_scraped_content
                }
                scraped_urls_by_fact[fact.id] = [
                    url for url in fact_urls
                    if all_scraped_content.get(url)
                ]
                scrape_errors_by_fact[fact.id] = {
                    url: "Scrape failed or empty content"
                    for url in fact_urls
                    if not all_scraped_content.get(url)
                }

            scrape_duration = time.time() - scrape_start
            successful_scrapes = len([v for v in all_scraped_content.values() if v])
            job_manager.add_progress(
                job_id, 
                f"Scraped {successful_scrapes}/{len(all_urls_to_scrape)} sources in {scrape_duration:.1f}s"
            )

            # Build fact search audits
            for fact in facts:
                fact_audit = build_fact_search_audit(
                    fact_id=fact.id,
                    fact_statement=fact.statement,
                    query_audits=query_audits_by_fact.get(fact.id, []),
                    credibility_results=credibility_results_by_fact.get(fact.id),
                    scraped_urls=scraped_urls_by_fact.get(fact.id, []),
                    scrape_errors=scrape_errors_by_fact.get(fact.id, {})
                )
                session_audit.add_fact_audit(fact_audit)

            # ================================================================
            # STAGE 6: Verify Facts ( PARALLEL)
            # ================================================================
            job_manager.add_progress(
                job_id,
                f"Verifying {len(facts)} facts in parallel..."
            )
            self._check_cancellation(job_id)

            verify_start = time.time()

            # Create verification tasks for ALL facts
            async def verify_single_fact(fact):
                """Verify a single fact and return result"""
                try:
                    scraped_content = scraped_content_by_fact.get(fact.id, {})
                    cred_results = credibility_results_by_fact.get(fact.id)
                    source_metadata = cred_results.source_metadata if cred_results else {}

                    if not scraped_content or not any(scraped_content.values()):
                        return FactCheckResult(
                            fact_id=fact.id,
                            statement=fact.statement,
                            match_score=0.0,
                            confidence=0.0,
                            report="Unable to verify - no credible sources found. Web search did not yield sources that could be successfully scraped."
                        )

                    # Extract relevant excerpts
                    excerpts = await self.highlighter.highlight(
                        fact=fact,
                        scraped_content=scraped_content
                    )

                    # Verify the fact
                    result = await self.checker.check_fact(
                        fact=fact,
                        excerpts=excerpts,
                        source_metadata=source_metadata
                    )

                    # Progress update
                    score_emoji = "" if result.match_score >= 0.9 else "" if result.match_score >= 0.7 else ""
                    job_manager.add_progress(
                        job_id,
                        f"{score_emoji} {fact.id}: {result.match_score:.0%} - {result.report[:50]}..."
                    )

                    return result

                except Exception as e:
                    fact_logger.logger.error(f"Verification error for {fact.id}: {e}")
                    return FactCheckResult(
                        fact_id=fact.id,
                        statement=fact.statement,
                        match_score=0.0,
                        confidence=0.0,
                        report=f"Verification error: {str(e)}"
                    )

            verify_tasks = [verify_single_fact(fact) for fact in facts]
            results = await asyncio.gather(*verify_tasks, return_exceptions=True)

            # Process verification results
            final_results = []
            for result in results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Verification exception: {result}")
                    continue
                final_results.append(result)

            verify_duration = time.time() - verify_start
            job_manager.add_progress(job_id, f"All facts verified in {verify_duration:.1f}s")

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
            summary = self._generate_summary(final_results)

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
                    pipeline_type="web-search"
                )

            job_manager.add_progress(job_id, f"Complete in {processing_time:.1f}s")

            # Log performance metrics
            fact_logger.logger.info(
                " Web Search Pipeline Performance",
                extra={
                    "total_time": round(processing_time, 2),
                    "query_gen_time": round(query_gen_duration, 2),
                    "search_time": round(search_duration, 2),
                    "filter_time": round(filter_duration, 2),
                    "scrape_time": round(scrape_duration, 2),
                    "verify_time": round(verify_duration, 2),
                    "num_facts": len(facts),
                    "parallel_mode": True
                }
            )

            return {
                "success": True,
                "session_id": session_id,
                "facts": [
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
                "summary": summary,
                "processing_time": processing_time,
                "methodology": "web_search_verification",
                "content_location": {
                    "country": content_location.country,
                    "language": content_location.language
                } if content_location else None,
                "statistics": {
                    "facts_extracted": len(facts),
                    "queries_generated": total_queries,
                    "raw_results_found": total_results,
                    "credible_sources": total_credible,
                    "facts_verified": len(final_results)
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
                            "tier3": session_audit.total_tier3,
                            "tier4_filtered": session_audit.total_tier4_filtered,
                            "tier5_filtered": session_audit.total_tier5_filtered
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

        except Exception as e:
            error_msg = str(e)
            if "cancelled" in error_msg.lower():
                job_manager.add_progress(job_id, "Verification cancelled")
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": "Cancelled by user",
                    "processing_time": time.time() - start_time
                }

            fact_logger.logger.error(f"Web search orchestrator error: {e}")
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
                except:
                    pass

            return {
                "success": False,
                "session_id": session_id,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
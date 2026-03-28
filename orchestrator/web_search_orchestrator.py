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

# TTS (The True Story) integration - Layer 0 verification
from agents.tts_router import TTSRouter
from utils.tts_service import TTSService

# Import search audit utilities
from utils.search_audit_builder import (
    build_session_search_audit,
    build_fact_search_audit,
    build_query_audit,
    save_search_audit,
    upload_search_audit_to_r2
)

# Import TTS audit utilities
from utils.tts_audit_builder import (
    build_tts_session_audit,
    build_routing_audit,
    build_claim_audit_from_evidence,
    build_skipped_claim_audit,
    build_failed_claim_audit,
    save_tts_audit,
    upload_tts_audit_to_r2,
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

        # TTS (The True Story) - Layer 0 verification
        # Provides pre-verified multi-source evidence for news claims
        try:
            self.tts_router = TTSRouter(config)
            self.tts_service = TTSService()
            self.tts_enabled = True
            fact_logger.logger.info("TTS Layer 0 enabled")
        except Exception as e:
            self.tts_router = None
            self.tts_service = None
            self.tts_enabled = False
            fact_logger.logger.warning(f"TTS Layer 0 not available: {e}")

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

            tts_session_audit = build_tts_session_audit(
                session_id=session_id,
                pipeline_type="web_search",
                content_country=content_location.country if content_location else "international",
                content_language=content_location.language if content_location else "english",
                tts_enabled=self.tts_enabled,
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
            # STAGE 1.5: TTS Layer 0 - Check The True Story (NEW)
            # ================================================================
            tts_results = []  # FactCheckResult objects from TTS
            tts_resolved_ids = set()  # fact IDs resolved by TTS
            tts_stats = {
                "enabled": self.tts_enabled,
                "routed_to_tts": 0,
                "resolved_by_tts": 0,
                "fell_through": 0,
            }

            if self.tts_enabled:
                job_manager.add_progress(job_id, "Checking The True Story news database...")
                self._check_cancellation(job_id)

                tts_start = time.time()

                try:
                    # Prepare facts for routing
                    facts_for_router = [
                        {"id": f.id, "statement": f.statement}
                        for f in facts
                    ]

                    # Route: which facts should check TTS?
                    routing_decisions = await self.tts_router.route(
                        claims=facts_for_router,
                        content_language=content_location.language if content_location else "english",
                        content_realm=content_location.country if content_location else "unknown",
                    )

                    # Build a lookup: claim_id -> routing_decision
                    routing_lookup = {}
                    for d in routing_decisions:
                        routing_lookup[d.claim_id] = d

                    # Record skipped claims in TTS audit
                    skipped_decisions = [d for d in routing_decisions if d.route == "skip"]
                    for d in skipped_decisions:
                        claim_obj = next(
                            (f for f in facts if f.id == d.claim_id), None
                        )
                        if claim_obj:
                            tts_session_audit.add_claim_audit(
                                build_skipped_claim_audit(
                                    claim_id=d.claim_id,
                                    claim_statement=claim_obj.statement,
                                    routing_decision=d,
                                )
                            )

                    # Collect TTS-routed facts
                    tts_routed = [
                        d for d in routing_decisions if d.route == "tts"
                    ]
                    tts_stats["routed_to_tts"] = len(tts_routed)

                    if tts_routed:
                        job_manager.add_progress(
                            job_id,
                            f"Searching TTS for {len(tts_routed)}/{len(facts)} facts..."
                        )

                        # Search TTS for each routed fact (parallel)
                        async def check_tts_for_fact(decision):
                            """Search TTS for a single fact"""
                            try:
                                evidence = await self.tts_service.find_evidence_for_claim(
                                    query=decision.tts_query,
                                    edition=decision.tts_edition or "en",
                                    min_cluster_size=3,
                                    max_evidence_articles=10,
                                )
                                return (decision.claim_id, evidence)
                            except Exception as e:
                                fact_logger.logger.error(
                                    f"TTS search error for {decision.claim_id}: {e}"
                                )
                                return (decision.claim_id, None)

                        tts_tasks = [check_tts_for_fact(d) for d in tts_routed]
                        tts_search_results = await asyncio.gather(
                            *tts_tasks, return_exceptions=True
                        )

                        # Process TTS results -> FactCheckResult
                        for result in tts_search_results:
                            if isinstance(result, BaseException):
                                continue

                            fact_id, evidence = result
                            if not evidence or not evidence.get("matched"):
                                # --- TTS Audit: record fell-through (no match) ---
                                ft_claim = next((f for f in facts if f.id == fact_id), None)
                                if ft_claim:
                                    tts_session_audit.add_claim_audit(
                                        build_failed_claim_audit(
                                            claim_id=fact_id,
                                            claim_statement=ft_claim.statement,
                                            routing_decision=routing_lookup.get(fact_id),
                                            evidence=evidence,
                                            reason="No TTS match found",
                                        )
                                    )
                                continue

                            cluster_size = evidence.get("source_count", 0)
                            if cluster_size < 3:
                                # --- TTS Audit: record fell-through (cluster too small) ---
                                ft_claim = next((f for f in facts if f.id == fact_id), None)
                                if ft_claim:
                                    tts_session_audit.add_claim_audit(
                                        build_failed_claim_audit(
                                            claim_id=fact_id,
                                            claim_statement=ft_claim.statement,
                                            routing_decision=routing_lookup.get(fact_id),
                                            evidence=evidence,
                                            reason=f"Cluster too small: {cluster_size} < 3",
                                        )
                                    )
                                continue

                            # Build evidence text for the fact checker
                            evidence_texts = evidence.get("evidence_texts", [])
                            story_sources = evidence.get("story_sources", [])

                            # Construct a verification report from TTS data
                            source_list = ", ".join(
                                s.get("source", "Unknown")
                                for s in (story_sources or evidence_texts)[:5]
                            )
                            cluster_title = evidence.get("cluster_title", "")

                            report = (
                                f"Verified via The True Story news database. "
                                f"This claim matches a news cluster with "
                                f"{cluster_size} independent sources. "
                                f"Cluster: \"{cluster_title}\". "
                                f"Sources include: {source_list}."
                            )

                            # Now use the LLM fact checker with TTS evidence
                            # to get a proper match_score (not just a binary yes/no)
                            fact_obj = next(
                                (f for f in facts if f.id == fact_id), None
                            )
                            if not fact_obj:
                                continue

                            try:
                                # Build excerpts from TTS article texts
                                tts_excerpts = {}
                                for et in evidence_texts:
                                    src = et.get("source", "TTS")
                                    txt = et.get("text", "")
                                    if txt:
                                        tts_excerpts[src] = [{"quote": txt, "relevance": 1.0}]

                                if tts_excerpts:
                                    # Use existing fact_checker with TTS evidence
                                    tts_check_result = await self.checker.check_fact(
                                        fact=fact_obj,
                                        excerpts=tts_excerpts,
                                        source_metadata={},
                                    )

                                    # Capture LLM score before boost
                                    original_llm_score = tts_check_result.match_score
                                    original_llm_report = tts_check_result.report

                                    # Guard: if LLM found the cluster irrelevant, fall through
                                    # to web search rather than boosting a 0-score result.
                                    # Threshold must match MIN_RELEVANCE in apply_tts_cluster_boost.
                                    TTS_MIN_RELEVANCE = 0.30
                                    if original_llm_score < TTS_MIN_RELEVANCE:
                                        fact_logger.logger.info(
                                            f"TTS cluster mismatch for {fact_id}: "
                                            f"llm_score={original_llm_score:.2f} < {TTS_MIN_RELEVANCE}, "
                                            f"cluster='{cluster_title[:60]}' -- falling through to web search"
                                        )
                                        ft_fact = next((f for f in facts if f.id == fact_id), None)
                                        if ft_fact:
                                            tts_session_audit.add_claim_audit(
                                                build_failed_claim_audit(
                                                    claim_id=fact_id,
                                                    claim_statement=fact_obj.statement,
                                                    routing_decision=routing_lookup.get(fact_id),
                                                    evidence=evidence,
                                                    reason=f"Cluster topic mismatch (llm_score={original_llm_score:.2f}): '{cluster_title[:60]}'",
                                                )
                                            )
                                        job_manager.add_progress(
                                            job_id,
                                            f"TTS: {fact_id} cluster unrelated, routing to web search"
                                        )
                                        continue

                                    # Apply cluster-size boost
                                    from utils.tts_service import apply_tts_cluster_boost, build_tts_story_url
                                    adjusted_score, adjusted_report = apply_tts_cluster_boost(
                                        llm_score=tts_check_result.match_score,
                                        cluster_size=cluster_size,
                                        cluster_title=cluster_title,
                                        source_list=source_list,
                                        llm_report=tts_check_result.report,
                                    )
                                    tts_check_result.match_score = adjusted_score
                                    tts_check_result.report = adjusted_report

                                    # Structured TTS metadata for frontend
                                    tts_cluster_id = evidence.get("cluster_id", "")
                                    tts_edition = evidence.get("edition", "en")
                                    if tts_cluster_id:
                                        tts_check_result.tts_story_url = build_tts_story_url(
                                            cluster_id=tts_cluster_id,
                                            edition=tts_edition,
                                        )
                                    tts_check_result.tts_source_count = cluster_size
                                    tts_check_result.tts_cluster_title = cluster_title
                                    tts_check_result.tts_source_list = source_list

                                    tts_results.append(tts_check_result)
                                    tts_resolved_ids.add(fact_id)

                                    # --- TTS Audit: record resolved claim ---
                                    tts_session_audit.add_claim_audit(
                                        build_claim_audit_from_evidence(
                                            claim_id=fact_id,
                                            claim_statement=fact_obj.statement,
                                            routing_decision=routing_lookup.get(fact_id),
                                            evidence=evidence,
                                            llm_match_score=original_llm_score,
                                            llm_report=original_llm_report,
                                            adjusted_match_score=tts_check_result.match_score,
                                            adjusted_report=tts_check_result.report,
                                            resolved=True,
                                        )
                                    )

                                    score_label = (
                                        "verified" if tts_check_result.match_score >= 0.9
                                        else "partially verified" if tts_check_result.match_score >= 0.7
                                        else "low confidence"
                                    )

                                    job_manager.add_progress(
                                        job_id,
                                        f"TTS: {fact_id} {score_label} "
                                        f"({tts_check_result.match_score:.0%}, "
                                        f"{cluster_size} sources)"
                                    )
                                else:
                                    # TTS matched but no usable text - fall through
                                    pass

                            except Exception as e:
                                fact_logger.logger.error(
                                    f"TTS verification error for {fact_id}: {e}"
                                )
                                # Fall through to web search

                    tts_stats["resolved_by_tts"] = len(tts_resolved_ids)
                    tts_stats["fell_through"] = tts_stats["routed_to_tts"] - len(tts_resolved_ids)

                    tts_duration = time.time() - tts_start
                    tts_stats["duration"] = round(tts_duration, 2)
                    tts_session_audit.tts_duration_seconds = tts_duration

                    job_manager.add_progress(
                        job_id,
                        f"TTS Layer 0: {len(tts_resolved_ids)} facts verified, "
                        f"{len(facts) - len(tts_resolved_ids)} need web search "
                        f"({tts_duration:.1f}s)"
                    )

                except Exception as e:
                    fact_logger.logger.error(f"TTS Layer 0 error: {e}")
                    job_manager.add_progress(
                        job_id,
                        "TTS Layer 0 unavailable, continuing with web search"
                    )

            # Filter out TTS-resolved facts from the web search pipeline
            remaining_facts = [f for f in facts if f.id not in tts_resolved_ids]

            if not remaining_facts:
                # ALL facts resolved by TTS - skip entire web search pipeline
                job_manager.add_progress(
                    job_id,
                    "All facts verified via TTS - skipping web search"
                )

                processing_time = time.time() - start_time
                summary = self._generate_summary(tts_results)

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
                            "tier_breakdown": r.tier_breakdown if hasattr(r, 'tier_breakdown') else None,
                            "tts_story_url": getattr(r, 'tts_story_url', None),
                            "tts_source_count": getattr(r, 'tts_source_count', None),
                            "tts_cluster_title": getattr(r, 'tts_cluster_title', None),
                            "tts_source_list": getattr(r, 'tts_source_list', None),
                        }
                        for r in tts_results
                    ],
                    "summary": summary,
                    "processing_time": processing_time,
                    "methodology": "tts_verified",
                    "content_location": {
                        "country": content_location.country,
                        "language": content_location.language
                    } if content_location else None,
                    "statistics": {
                        "facts_extracted": len(facts),
                        "tts_resolved": len(tts_resolved_ids),
                        "web_search_needed": 0,
                    },
                    "tts_stats": tts_stats,
                }

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

            query_tasks = [generate_queries_for_fact(fact) for fact in remaining_facts]
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

            search_tasks = [search_for_fact(fact) for fact in remaining_facts]
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

            filter_tasks = [filter_sources_for_fact(fact) for fact in remaining_facts]
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

            for fact in remaining_facts:
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

            for fact in remaining_facts:
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
            for fact in remaining_facts:
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
                f"Verifying {len(remaining_facts)} facts in parallel..."
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
                    score_label = "OK" if result.match_score >= 0.9 else "PARTIAL" if result.match_score >= 0.7 else "LOW"
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

            verify_tasks = [verify_single_fact(fact) for fact in remaining_facts]
            results = await asyncio.gather(*verify_tasks, return_exceptions=True)

            # Process verification results
            final_results = []
            for result in results:
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f"Verification exception: {result}")
                    continue
                final_results.append(result)

            # Merge TTS-resolved results with web search results
            final_results = tts_results + final_results

            verify_duration = time.time() - verify_start
            job_manager.add_progress(job_id, f"All facts verified in {verify_duration:.1f}s")

            # Clean up scraper only if we created it (not shared)
            if not using_shared_scraper:
                try:
                    await scraper.close()
                except Exception:
                    pass

            # Close TTS service
            if self.tts_enabled and self.tts_service:
                try:
                    await self.tts_service.close()
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

            # Save TTS audit
            tts_audit_file_path = None
            tts_audit_r2_url = None

            if tts_session_audit.total_claims > 0:
                tts_audit_file_path = save_tts_audit(
                    tts_audit=tts_session_audit,
                    file_manager=self.file_manager,
                    session_id=session_id,
                    filename="tts_audit.json",
                )

                if self.r2_enabled and self.r2_uploader:
                    tts_audit_r2_url = await upload_tts_audit_to_r2(
                        tts_audit=tts_session_audit,
                        session_id=session_id,
                        r2_uploader=self.r2_uploader,
                        pipeline_type="web-search",
                    )

            job_manager.add_progress(job_id, f"Complete in {processing_time:.1f}s")

            # Log performance metrics
            fact_logger.logger.info(
                "Web Search Pipeline Performance",
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
                        "tier_breakdown": r.tier_breakdown if hasattr(r, 'tier_breakdown') else None,
                        "tts_story_url": getattr(r, 'tts_story_url', None),
                        "tts_source_count": getattr(r, 'tts_source_count', None),
                        "tts_cluster_title": getattr(r, 'tts_cluster_title', None),
                        "tts_source_list": getattr(r, 'tts_source_list', None),
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
                    "tts_resolved": len(tts_resolved_ids),
                    "queries_generated": total_queries,
                    "raw_results_found": total_results,
                    "credible_sources": total_credible,
                    "facts_verified": len(final_results)
                },
                "tts_stats": tts_stats,
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
                "tts_audit": {
                    "local_path": tts_audit_file_path,
                    "r2_url": tts_audit_r2_url,
                    "summary": tts_session_audit.to_dict()["summary"] if tts_session_audit.total_claims > 0 else None,
                },
                "r2_upload": {
                    "success": audit_r2_url is not None,
                    "url": audit_r2_url
                },
                "performance": {
                    "tts_layer": tts_stats.get("duration", 0),
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

            if tts_session_audit and tts_session_audit.total_claims > 0:
                try:
                    save_tts_audit(
                        tts_audit=tts_session_audit,
                        file_manager=self.file_manager,
                        session_id=session_id,
                        filename="tts_audit_partial.json",
                    )
                except:
                    pass

            return {
                "success": False,
                "session_id": session_id,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
# orchestrator/manipulation_orchestrator.py
"""
Opinion Manipulation Detection Orchestrator
Coordinates the full pipeline for detecting fact manipulation in articles

Pipeline:
1. Article Analysis - Detect agenda, political lean, summary
2. Fact Extraction - Extract facts with framing context
3. Web Search Verification - Verify facts via existing pipeline ( PARALLEL)
4. Manipulation Analysis - Compare verified facts to presentation ( PARALLEL)
5. Report Synthesis - Create comprehensive manipulation report
6. Save audit file to R2

Reuses existing components:
- QueryGenerator for search query creation
- BraveSearcher for web search
- CredibilityFilter for source filtering
- BrowserlessScraper for content scraping
- Highlighter for excerpt extraction
- FactChecker for verification

 OPTIMIZED: Parallel fact verification and manipulation analysis
   - All facts verified simultaneously using asyncio.gather()
   - ~60-70% faster than sequential processing
"""

from langsmith import traceable
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.file_manager import FileManager
from utils.job_manager import job_manager
from utils.browserless_scraper import BrowserlessScraper
from utils.brave_searcher import BraveSearcher

#  FIX 1: Import build_manipulation_context
try:
    from utils.credibility_context import build_manipulation_context
except ImportError:
    # Fallback if module not yet added
    def build_manipulation_context(source_credibility=None, source_info=None):
        return ""

# Import the manipulation detector agent
from agents.manipulation_detector import (
    ManipulationDetector,
    ArticleSummary,
    ExtractedFact,
    ManipulationFinding,
    ManipulationReport
)

# Import existing agents for fact verification
from agents.query_generator import QueryGenerator
from agents.credibility_filter import CredibilityFilter
from agents.highlighter import Highlighter
from agents.fact_checker import FactChecker

# Import search audit utilities
from utils.search_audit_builder import (
    build_session_search_audit,
    build_fact_search_audit,
    build_query_audit,
    save_search_audit,
    upload_search_audit_to_r2
)


class CancelledException(Exception):
    """Raised when job is cancelled"""
    pass


# Type alias for verification result tuple
VerificationResultTuple = Tuple[str, Dict[str, Any], str, List, Optional[str]]
ManipulationResultTuple = Tuple[Optional[ManipulationFinding], Optional[str]]


class ManipulationOrchestrator:
    """
    Orchestrator for opinion manipulation detection pipeline

    Coordinates:
    1. ManipulationDetector agent (4 stages)
    2. Existing fact-checking components (verification)
    3. Job management and progress streaming
    4. Audit file generation

     OPTIMIZED: Uses parallel processing for fact verification
    """

    def __init__(self, config):
        """
        Initialize the ManipulationOrchestrator

        Args:
            config: Configuration object with API keys and settings.
                   Can be a Config object (with attributes) or a dict.
        """
        self.config = config

        # Initialize the manipulation detector agent
        self.detector = ManipulationDetector(config)

        # Initialize existing fact-checking components
        self.query_generator = QueryGenerator(config)
        self.brave_searcher = BraveSearcher(config, max_results=5)
        self.credibility_filter = CredibilityFilter(config)
        # NOTE: Don't create scraper here - it binds asyncio.Lock to wrong event loop
        # self.scraper = BrowserlessScraper(config)  
        self.highlighter = Highlighter(config)
        self.fact_checker = FactChecker(config)

        # File manager for audit files
        self.file_manager = FileManager()

        # Handle config as either dict or object
        if isinstance(config, dict):
            self.max_sources_per_fact = config.get('max_sources_per_fact', 5)
            self.max_facts = config.get('max_facts', 5)
        else:
            self.max_sources_per_fact = getattr(config, 'max_sources_per_fact', 5)
            self.max_facts = getattr(config, 'max_facts', 5)

        #  NEW: Parallel processing settings (no rate limit concerns with paid Brave)
        self.max_concurrent_verifications = 5  # All facts in parallel

        # Initialize R2 uploader for audit files
        try:
            from utils.r2_uploader import R2Uploader
            self.r2_uploader = R2Uploader()
            self.r2_enabled = True
            fact_logger.logger.info(" R2 uploader initialized for manipulation audits")
        except Exception as e:
            self.r2_enabled = False
            self.r2_uploader = None
            fact_logger.logger.warning(f" R2 not available for audits: {e}")

        fact_logger.log_component_start(
            "ManipulationOrchestrator",
            max_sources_per_fact=self.max_sources_per_fact,
            max_facts=self.max_facts,
            parallel_verifications=self.max_concurrent_verifications
        )

    def _check_cancellation(self, job_id: str):
        """Check if job has been cancelled and raise exception if so"""
        if job_manager.is_cancelled(job_id):
            fact_logger.logger.info(f" Job {job_id} was cancelled")
            raise CancelledException(f"Job {job_id} was cancelled by user")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))
        return f"manip_{timestamp}_{random_suffix}"

    @traceable(
        name="manipulation_detection_pipeline",
        run_type="chain",
        tags=["manipulation-detection", "full-pipeline"]
    )
    async def process_with_progress(
        self,
        content: str,
        job_id: str,
        source_info: str = "Unknown source",
        source_credibility: Optional[Dict[str, Any]] = None,
        standalone: bool = True  # ADD THIS
    ) -> Dict[str, Any]:
        """
        Run the full manipulation detection pipeline with progress updates

         OPTIMIZED: Uses parallel processing for fact verification and manipulation analysis

        Args:
            content: Article text to analyze
            job_id: Job ID for progress tracking
            source_info: URL or source name
            source_credibility: Optional pre-fetched credibility data (NEW)

        Returns:
            Dict with manipulation report and metadata
        """
        start_time = time.time()
        session_id = self._generate_session_id()

        # Track credibility usage
        using_credibility = source_credibility is not None
        credibility_tier = source_credibility.get('tier') if source_credibility else None
        is_propaganda = source_credibility.get('is_propaganda', False) if source_credibility else False

        fact_logger.logger.info(
            " Starting manipulation detection pipeline",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "content_length": len(content),
                "parallel_mode": True,
                "has_credibility": using_credibility
            }
        )

        try:
            # ================================================================
            # STAGE 0: Log Source Credibility Context (NEW)
            # ================================================================
            if source_credibility:
                tier = source_credibility.get('tier', '?')
                bias = source_credibility.get('bias_rating', 'Unknown')
                factual = source_credibility.get('factual_reporting', 'Unknown')

                job_manager.add_progress(
                    job_id, 
                    f" Source credibility: Tier {tier} | {bias} bias | {factual} factual"
                )

                # Special warnings for concerning sources
                if is_propaganda:
                    job_manager.add_progress(
                        job_id,
                        " SOURCE FLAGGED AS PROPAGANDA - Maximum scrutiny applied"
                    )
                elif credibility_tier and credibility_tier >= 4:
                    job_manager.add_progress(
                        job_id,
                        " Low credibility source - heightened scrutiny for manipulation"
                    )
                elif credibility_tier and credibility_tier <= 2:
                    job_manager.add_progress(
                        job_id,
                        " High credibility source - focusing on subtle framing issues"
                    )

            # ================================================================
            # STAGE 1: Article Analysis
            # ================================================================
            job_manager.add_progress(job_id, "Analyzing article for agenda and bias...")
            self._check_cancellation(job_id)

            #  FIX 2: Build credibility context and call analyze_article OUTSIDE the if block
            credibility_context = ""
            if source_credibility:
                credibility_context = build_manipulation_context(source_credibility, source_info)

            #  FIX 2: Always call analyze_article (not just when source_credibility exists)
            article_summary = await self.detector.analyze_article(
                content,
                source_info,
                credibility_context=credibility_context if credibility_context else None
            )

            job_manager.add_progress(
                job_id, 
                f" Detected agenda: {article_summary.detected_agenda[:50]}..."
            )
            job_manager.add_progress(
                job_id,
                f" Political lean: {article_summary.political_lean} | Opinion ratio: {article_summary.opinion_fact_ratio:.0%}"
            )

            # ================================================================
            # STAGE 2: Fact Extraction
            # ================================================================
            job_manager.add_progress(job_id, " Extracting key facts with framing analysis...")
            self._check_cancellation(job_id)

            facts = await self.detector.extract_facts(content, article_summary)

            if not facts:
                job_manager.add_progress(job_id, " No verifiable facts found")
                #  FIX 3: Pass credibility variables to helper method
                return self._build_no_facts_result(
                    session_id=session_id,
                    article_summary=article_summary,
                    start_time=start_time,
                    source_credibility=source_credibility,
                    using_credibility=using_credibility,
                    is_propaganda=is_propaganda
                )

            # Limit facts if needed
            if len(facts) > self.max_facts:
                facts = facts[:self.max_facts]

            job_manager.add_progress(job_id, f" Extracted {len(facts)} key facts for verification")

            # Initialize session audit
            session_audit = build_session_search_audit(
                session_id=session_id,
                pipeline_type="manipulation_detection",
                content_country="international",
                content_language="english"
            )

            #  Create scraper ONCE in the async context (correct event loop)
            self.scraper = BrowserlessScraper(self.config)

            # ================================================================
            # STAGE 3: Fact Verification ( PARALLEL PROCESSING)
            # ================================================================
            job_manager.add_progress(
                job_id, 
                f" Starting parallel verification of {len(facts)} facts..."
            )
            self._check_cancellation(job_id)

            #  Create verification tasks for ALL facts
            verification_tasks = [
                self._verify_single_fact_parallel(
                    fact=fact,
                    fact_index=i,
                    total_facts=len(facts),
                    job_id=job_id,
                    article_summary=article_summary
                )
                for i, fact in enumerate(facts, 1)
            ]

            #  Execute ALL verifications in parallel
            verification_start = time.time()
            try:
                results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            except CancelledException:
                raise

            verification_duration = time.time() - verification_start
            fact_logger.logger.info(
                f" Parallel verification completed in {verification_duration:.1f}s",
                extra={"num_facts": len(facts), "duration": verification_duration}
            )

            # Process results from parallel execution
            verification_results: Dict[str, Dict[str, Any]] = {}
            source_excerpts_by_fact: Dict[str, str] = {}
            query_audits_by_fact: Dict[str, List] = {}
            verification_errors: List[str] = []

            for result in results:
                #  FIX: Check if result is an exception BEFORE unpacking
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f" Verification task exception: {result}")
                    verification_errors.append(str(result))
                    continue

                # Now safe to unpack the tuple
                fact_id, verification, excerpts, query_audits, error = result

                verification_results[fact_id] = verification
                source_excerpts_by_fact[fact_id] = excerpts
                query_audits_by_fact[fact_id] = query_audits

                if error:
                    verification_errors.append(f"{fact_id}: {error}")

                # Add to session audit
                fact_statement = next((f.statement for f in facts if f.id == fact_id), "")
                fact_audit = build_fact_search_audit(
                    fact_id=fact_id,
                    fact_statement=fact_statement,
                    query_audits=query_audits,
                    credibility_results=None,
                    scraped_urls=[],
                    scrape_errors={}
                )
                session_audit.add_fact_audit(fact_audit)

            successful_verifications = len(facts) - len(verification_errors)
            job_manager.add_progress(
                job_id, 
                f" Fact verification complete: {successful_verifications}/{len(facts)} in {verification_duration:.1f}s"
            )

            # ================================================================
            # STAGE 4: Manipulation Analysis ( PARALLEL PROCESSING)
            # ================================================================
            job_manager.add_progress(job_id, " Analyzing manipulation patterns in parallel...")
            self._check_cancellation(job_id)

            #  Create manipulation analysis tasks for ALL facts
            manipulation_tasks = [
                self._analyze_manipulation_parallel(
                    fact=fact,
                    article_summary=article_summary,
                    verification=verification_results.get(fact.id, {}),
                    excerpts=source_excerpts_by_fact.get(fact.id, "No excerpts available"),
                    job_id=job_id
                )
                for fact in facts
            ]

            #  Execute ALL manipulation analyses in parallel
            manipulation_start = time.time()
            try:
                manipulation_results = await asyncio.gather(*manipulation_tasks, return_exceptions=True)
            except CancelledException:
                raise

            manipulation_duration = time.time() - manipulation_start
            fact_logger.logger.info(
                f" Parallel manipulation analysis completed in {manipulation_duration:.1f}s",
                extra={"num_facts": len(facts), "duration": manipulation_duration}
            )

            # Process manipulation results
            manipulation_findings: List[ManipulationFinding] = []
            for result in manipulation_results:
                #  FIX: Check if result is an exception BEFORE unpacking
                if isinstance(result, BaseException):
                    fact_logger.logger.error(f" Manipulation analysis exception: {result}")
                    continue

                finding, error = result
                if finding:
                    manipulation_findings.append(finding)
                    if finding.manipulation_detected:
                        job_manager.add_progress(
                            job_id,
                            f" Manipulation detected in {finding.fact_id}: {finding.manipulation_severity} severity"
                        )

            job_manager.add_progress(job_id, f" Manipulation analysis complete in {manipulation_duration:.1f}s")

            # ================================================================
            # STAGE 5: Report Synthesis
            # ================================================================
            job_manager.add_progress(job_id, " Synthesizing final report...")
            self._check_cancellation(job_id)

            processing_time = time.time() - start_time

            report = await self.detector.synthesize_report(
                article_summary=article_summary,
                facts=facts,
                manipulation_findings=manipulation_findings,
                processing_time=processing_time
            )

            job_manager.add_progress(
                job_id,
                f" Manipulation score: {report.overall_manipulation_score:.1f}/10"
            )

            # ================================================================
            # STAGE 6: Save Audit File
            # ================================================================
            job_manager.add_progress(job_id, " Saving audit report...")

            audit_path = save_search_audit(
                session_audit=session_audit,
                file_manager=self.file_manager,
                session_id=session_id,
                filename="search_audit.json"
            )

            r2_url = None

            if audit_path and self.r2_enabled and self.r2_uploader:
                r2_url = await upload_search_audit_to_r2(
                    session_audit=session_audit,
                    session_id=session_id,
                    r2_uploader=self.r2_uploader,
                    pipeline_type="manipulation-detection"
                )
                if r2_url:
                    job_manager.add_progress(job_id, " Audit saved to cloud")

            #  FIX 4: Pass credibility to _build_result
            result = self._build_result(
                session_id=session_id,
                report=report,
                facts=facts,
                verification_results=verification_results,
                r2_url=r2_url,
                start_time=start_time,
                source_credibility=source_credibility,
                using_credibility=using_credibility,
                is_propaganda=is_propaganda
            )

            job_manager.add_progress(job_id, " Analysis complete!")

            if standalone:
                job_manager.complete_job(job_id, result)

            fact_logger.logger.info(
                " Manipulation detection pipeline complete",
                extra={
                    "session_id": session_id,
                    "manipulation_score": report.overall_manipulation_score,
                    "processing_time": round(time.time() - start_time, 2),
                    "verification_time": round(verification_duration, 2),
                    "manipulation_analysis_time": round(manipulation_duration, 2),
                    "used_credibility": using_credibility
                }
            )

            # Clean up scraper to release browser resources
            try:
                await self.scraper.close()
            except Exception as cleanup_error:
                fact_logger.logger.debug(f"Scraper cleanup: {cleanup_error}")

            return result

        except CancelledException:
            # Clean up on cancellation too
            try:
                await self.scraper.close()
            except Exception:
                pass
            job_manager.add_progress(job_id, " Analysis cancelled by user")
            raise

        except Exception as e:
            fact_logger.logger.error(f" Pipeline failed: {e}")
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
            job_manager.add_progress(job_id, f" Error: {str(e)}")
            # Clean up scraper on error
            try:
                await self.scraper.close()
            except Exception:
                pass
            raise

    # =========================================================================
    #  NEW: Parallel Verification Helper
    # =========================================================================

    async def _verify_single_fact_parallel(
        self,
        fact: ExtractedFact,
        fact_index: int,
        total_facts: int,
        job_id: str,
        article_summary: ArticleSummary
    ) -> VerificationResultTuple:
        """
        Verify a single fact - designed for parallel execution with asyncio.gather()

        Args:
            fact: The fact to verify
            fact_index: Index of this fact (1-based)
            total_facts: Total number of facts being verified
            job_id: Job ID for progress tracking
            article_summary: Article context for query generation

        Returns:
            Tuple of (fact_id, verification_result, excerpts, query_audits, error_message)
        """
        try:
            # Check cancellation before starting
            self._check_cancellation(job_id)

            # Progress update
            job_manager.add_progress(
                job_id,
                f" [{fact_index}/{total_facts}] Verifying: {fact.statement[:40]}..."
            )

            # Run actual verification
            verification, excerpts, query_audits = await self._verify_fact(
                fact=fact,
                job_id=job_id,
                article_summary=article_summary
            )

            # Log completion with score
            score = verification.get('match_score', 0.5)
            emoji = "[OK]" if score >= 0.7 else "[WARN]" if score >= 0.4 else "[FAIL]"
            job_manager.add_progress(
                job_id,
                f"{emoji} [{fact.id}] Verified: {score:.0%} confidence"
            )

            return (fact.id, verification, excerpts, query_audits, None)

        except CancelledException:
            raise  # Re-raise to stop all parallel tasks

        except Exception as e:
            fact_logger.logger.error(
                f" Parallel verification failed for {fact.id}: {e}",
                extra={"fact_id": fact.id, "error": str(e)}
            )
            # Return error result instead of crashing the entire batch
            return (
                fact.id,
                self._empty_verification(f"Verification error: {str(e)}"),
                "",
                [],
                str(e)
            )

    # =========================================================================
    #  NEW: Parallel Manipulation Analysis Helper
    # =========================================================================

    async def _analyze_manipulation_parallel(
        self,
        fact: ExtractedFact,
        article_summary: ArticleSummary,
        verification: Dict[str, Any],
        excerpts: str,
        job_id: str
    ) -> ManipulationResultTuple:
        """
        Analyze manipulation for a single fact - designed for parallel execution

        Args:
            fact: The fact to analyze
            article_summary: Article context
            verification: Verification result for this fact
            excerpts: Source excerpts for this fact
            job_id: Job ID for cancellation checks

        Returns:
            Tuple of (ManipulationFinding or None, error_message or None)
        """
        try:
            self._check_cancellation(job_id)

            finding = await self.detector.analyze_manipulation(
                fact=fact,
                article_summary=article_summary,
                verification_result=verification,
                source_excerpts=excerpts
            )

            return (finding, None)

        except CancelledException:
            raise

        except Exception as e:
            fact_logger.logger.error(
                f" Manipulation analysis failed for {fact.id}: {e}",
                extra={"fact_id": fact.id, "error": str(e)}
            )
            return (None, str(e))

    # =========================================================================
    # Existing Verification Logic (with type fixes)
    # =========================================================================

    async def _verify_fact(
        self,
        fact: ExtractedFact,
        job_id: str,
        article_summary: ArticleSummary
    ) -> Tuple[Dict[str, Any], str, List]:
        """
        Verify a single fact using the existing fact-checking pipeline

        Returns:
            Tuple of (verification_result, formatted_excerpts, query_audits)
        """
        query_audits: List = []

        try:
            # Step 1: Generate search queries
            fact_obj = type('Fact', (), {
                'id': fact.id,
                'statement': fact.statement
            })()

            queries = await self.query_generator.generate_queries(
                fact=fact_obj,
                context=f"Article agenda: {article_summary.detected_agenda}"
            )

            if not queries or not queries.primary_query:
                return self._empty_verification("Failed to generate search queries"), "", []

            # Step 2: Execute web searches
            search_results = await self.brave_searcher.search_multiple(
                queries=queries.all_queries,
                search_depth="advanced",
                max_concurrent=3  #  Can be aggressive with paid Brave account
            )

            # Build query audits
            for query, brave_results in search_results.items():
                qa = build_query_audit(
                    query=query,
                    brave_results=brave_results,
                    query_type="english",
                    language="en"
                )
                query_audits.append(qa)

            # Collect all search results for credibility filter
            all_search_results: List[Dict[str, Any]] = []
            for query, brave_results in search_results.items():
                for result in brave_results.results:
                    # Handle both dict and object format (Brave API returns dicts)
                    if isinstance(result, dict):
                        url = result.get('url', '')
                        title = result.get('title', '')
                        content = result.get('content', result.get('description', ''))
                    else:
                        url = getattr(result, 'url', '')
                        title = getattr(result, 'title', '')
                        content = getattr(result, 'content', getattr(result, 'description', ''))

                    all_search_results.append({
                        'url': url,
                        'title': title,
                        'content': content
                    })

            if not all_search_results:
                return self._empty_verification("No search results found"), "", query_audits

            # Step 3: Filter by credibility
            cred_results = await self.credibility_filter.evaluate_sources(
                fact=fact_obj,
                search_results=all_search_results
            )

            credible_urls = cred_results.get_recommended_urls(min_score=0.70) if cred_results else []

            if not credible_urls:
                return self._empty_verification("No credible sources found"), "", query_audits

            source_metadata = cred_results.source_metadata if cred_results else {}

            # Step 4: Scrape sources
            urls_to_scrape = credible_urls[:self.max_sources_per_fact]
            scraped_content = await self.scraper.scrape_urls_for_facts(urls_to_scrape)

            if not scraped_content or not any(scraped_content.values()):
                return self._empty_verification("Failed to scrape sources"), "", query_audits

            # Step 5: Extract excerpts
            all_excerpts: List[Dict[str, Any]] = []
            for url, content in scraped_content.items():
                if not content:
                    continue

                excerpts = await self.highlighter.highlight(fact_obj, {url: content})
                url_excerpts = excerpts.get(url, [])

                tier = 'unknown'
                if source_metadata and url in source_metadata:
                    metadata_obj = source_metadata[url]
                    tier = getattr(metadata_obj, 'credibility_tier', 'unknown')

                for excerpt in url_excerpts:
                    all_excerpts.append({
                        'url': url,
                        'tier': tier,
                        'quote': excerpt.get('quote', '') if isinstance(excerpt, dict) else str(excerpt),
                        'relevance': excerpt.get('relevance', 0.5) if isinstance(excerpt, dict) else 0.5
                    })

            if not all_excerpts:
                return self._empty_verification("No relevant excerpts found"), "", query_audits

            # Step 6: Verify fact
            tier_order = {'tier1': 0, 'tier2': 1, 'tier3': 2, 'unknown': 3}
            all_excerpts.sort(key=lambda x: (
                tier_order.get(x['tier'], 3),
                -x['relevance']
            ))

            formatted_excerpts = self._format_excerpts_for_checker(all_excerpts)

            # Convert to dict format for fact checker
            excerpts_by_url: Dict[str, List[Dict[str, Any]]] = {}
            for excerpt in all_excerpts:
                url = excerpt.get('url', '')
                if url not in excerpts_by_url:
                    excerpts_by_url[url] = []
                excerpts_by_url[url].append({
                    'quote': excerpt.get('quote', ''),
                    'relevance': excerpt.get('relevance', 0.5),
                    'context': excerpt.get('context', ''),
                    'tier': excerpt.get('tier', 'unknown')
                })

            # Convert source_metadata to dict format for fact_checker
            source_metadata_dict: Dict[str, Dict[str, Any]] = {}
            for url, meta in source_metadata.items():
                source_metadata_dict[url] = {
                    'name': getattr(meta, 'name', ''),
                    'source_type': getattr(meta, 'source_type', ''),
                    'credibility_score': getattr(meta, 'credibility_score', 0.0),
                    'credibility_tier': getattr(meta, 'credibility_tier', 'unknown'),
                    'tier': getattr(meta, 'credibility_tier', 'unknown')
                }

            verification_result = await self.fact_checker.check_fact(
                fact=fact_obj,
                excerpts=excerpts_by_url,
                source_metadata=source_metadata_dict
            )

            result = {
                'match_score': verification_result.match_score if verification_result else 0.5,
                'confidence': verification_result.confidence if verification_result else 0.5,
                'report': verification_result.report if verification_result else "Verification incomplete",
                'sources_used': credible_urls,
                'excerpts': formatted_excerpts
            }

            return result, formatted_excerpts, query_audits

        except Exception as e:
            fact_logger.logger.error(f" Fact verification failed: {e}")
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_verification(f"Error: {str(e)}"), "", query_audits

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _empty_verification(self, reason: str) -> Dict[str, Any]:
        """Return empty verification result"""
        return {
            'match_score': 0.5,
            'confidence': 0.0,
            'report': reason,
            'sources_used': [],
            'excerpts': ""
        }

    def _format_excerpts_for_checker(self, excerpts: List[Dict[str, Any]]) -> str:
        """Format excerpts for the fact checker"""
        lines: List[str] = []

        for excerpt in excerpts[:10]:
            tier_label = excerpt['tier'].upper() if excerpt['tier'] else 'UNKNOWN'
            quote = excerpt.get('quote', '')
            url = excerpt.get('url', 'Unknown URL')

            lines.append(f"[{tier_label}] {url}")
            if len(quote) > 500:
                lines.append(f"  \"{quote[:500]}...\"")
            else:
                lines.append(f"  \"{quote}\"")
            lines.append("")

        return "\n".join(lines)

    #  FIX 3: Add credibility parameters to _build_no_facts_result
    def _build_no_facts_result(
        self,
        session_id: str,
        article_summary: ArticleSummary,
        start_time: float,
        source_credibility: Optional[Dict[str, Any]] = None,
        using_credibility: bool = False,
        is_propaganda: bool = False
    ) -> Dict[str, Any]:
        """Build result when no facts were extracted"""
        return {
            'success': True,
            'session_id': session_id,
            #  Include credibility fields
            'source_credibility': source_credibility,
            'used_source_credibility': using_credibility,
            'source_flagged_propaganda': is_propaganda,
            'article_summary': {
                'main_thesis': article_summary.main_thesis,
                'political_lean': article_summary.political_lean,
                'detected_agenda': article_summary.detected_agenda,
                'opinion_fact_ratio': article_summary.opinion_fact_ratio,
                'emotional_tone': article_summary.emotional_tone
            },
            'manipulation_score': 0.0,
            'facts_analyzed': [],
            'manipulation_findings': [],
            'report': {
                'overall_score': 0.0,
                'justification': "No verifiable facts could be extracted from the article",
                'techniques_used': [],
                'what_got_right': ["Article may be purely opinion-based"],
                'misleading_elements': [],
                'recommendation': "This article appears to contain no verifiable factual claims."
            },
            'processing_time': time.time() - start_time,
            'r2_url': None
        }

    #  FIX 4: Add credibility parameters to _build_result
    def _build_result(
        self,
        session_id: str,
        report: ManipulationReport,
        facts: List[ExtractedFact],
        verification_results: Dict[str, Dict[str, Any]],
        r2_url: Optional[str],
        start_time: float,
        source_credibility: Optional[Dict[str, Any]] = None,
        using_credibility: bool = False,
        is_propaganda: bool = False
    ) -> Dict[str, Any]:
        """Build the final result dictionary"""

        # Format facts for response
        facts_data: List[Dict[str, Any]] = []
        for fact in facts:
            verification = verification_results.get(fact.id, {})
            facts_data.append({
                'id': fact.id,
                'statement': fact.statement,
                'original_text': fact.original_text,
                'framing': fact.framing,
                'context_given': fact.context_given,
                'context_potentially_omitted': fact.context_potentially_omitted,
                'manipulation_potential': fact.manipulation_potential,
                'verification': {
                    'match_score': verification.get('match_score', 0.5),
                    'sources_used': verification.get('sources_used', [])
                }
            })

        # Format manipulation findings
        findings_data: List[Dict[str, Any]] = []
        for finding in report.facts_analyzed:
            findings_data.append({
                'fact_id': finding.fact_id,
                'fact_statement': finding.fact_statement,
                'truthfulness': finding.truthfulness,
                'truth_score': finding.truth_score,
                'manipulation_detected': finding.manipulation_detected,
                'manipulation_types': finding.manipulation_types,
                'manipulation_severity': finding.manipulation_severity,
                'what_was_omitted': finding.what_was_omitted,
                'how_it_serves_agenda': finding.how_it_serves_agenda,
                'corrected_context': finding.corrected_context,
                'sources_used': finding.sources_used
            })

        return {
            'success': True,
            'session_id': session_id,
            #  Include credibility fields at top level
            'source_credibility': source_credibility,
            'used_source_credibility': using_credibility,
            'source_flagged_propaganda': is_propaganda,
            'article_summary': {
                'main_thesis': report.article_summary.main_thesis,
                'political_lean': report.article_summary.political_lean,
                'detected_agenda': report.article_summary.detected_agenda,
                'opinion_fact_ratio': report.article_summary.opinion_fact_ratio,
                'emotional_tone': report.article_summary.emotional_tone,
                'target_audience': report.article_summary.target_audience,
                'rhetorical_strategies': report.article_summary.rhetorical_strategies,
                'summary': report.article_summary.summary
            },
            'manipulation_score': report.overall_manipulation_score,
            'report': {
                'techniques_used': report.manipulation_techniques_used,
                'what_got_right': report.what_article_got_right,
                'misleading_elements': report.key_misleading_elements,
                'justification': report.score_justification,
                'recommendation': report.reader_recommendation,
                'narrative_summary': report.narrative_summary
            },
            'facts_analyzed': facts_data,
            'manipulation_findings': findings_data,
            'processing_time': time.time() - start_time,
            'r2_url': r2_url
        }
# orchestrator/comprehensive_orchestrator.py
"""
Comprehensive Analysis Orchestrator
Coordinates the full 3-stage comprehensive analysis pipeline

STAGE 1: Pre-Analysis (produces MetadataBlocks)
- Content Classification + Source Verification run in PARALLEL
- Mode Routing runs after classification completes (depends on its result)
- Any future checks simply add a helper method + join the gather() call

STAGE 2: Parallel Mode Execution
- Run ALL selected modes simultaneously using asyncio.gather()
- Each mode gets its own orchestrator instance + fresh LLM keys (via key rotation)
- standalone=False prevents modes from calling complete_job() prematurely
- Collect reports from each mode
- Stream progress updates (prefixed with mode name for clarity)

STAGE 3: AI-Powered Synthesis
- Uses ReportSynthesizer agent with GPT-4o
- Dynamically assembles prompt from ALL metadata blocks
- Generates unified credibility score (0-100) and rating
- Key concerns, positive indicators, recommendations
- Human-readable narrative summary

ARCHITECTURE: MetadataBlock pattern
- Each pre-analysis check produces a MetadataBlock
- Blocks are self-describing (each formats itself for the LLM)
- Adding a new check = write the check + add a builder function
- Zero changes needed in synthesizer or prompts
"""

from langsmith import traceable
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from utils.logger import fact_logger
from utils.file_manager import FileManager
from utils.job_manager import job_manager
from utils.metadata_block import (
    MetadataBlock,
    ImpactSignal,
    build_content_classification_block,
    build_source_credibility_block,
)
from utils.async_utils import safe_float

# Stage 1 components
from agents.content_classifier import ContentClassifier
from utils.source_verifier import SourceVerifier
from agents.mode_router import ModeRouter

# Stage 2: Mode orchestrators
from orchestrator.key_claims_orchestrator import KeyClaimsOrchestrator
from orchestrator.bias_check_orchestrator import BiasCheckOrchestrator
from orchestrator.manipulation_orchestrator import ManipulationOrchestrator
from orchestrator.lie_detector_orchestrator import LieDetectorOrchestrator

# Shared scrape cache for parallel mode execution (Phase 3)
from utils.scrape_cache import ScrapeCache

# Stage 3: Report Synthesizer
from agents.report_synthesizer import ReportSynthesizer


class CancelledException(Exception):
    """Raised when job is cancelled"""
    pass


class ComprehensiveOrchestrator:
    """
    Master orchestrator for comprehensive analysis mode

    Coordinates:
    - Stage 1: Pre-analysis checks -> MetadataBlocks + mode routing
    - Stage 2: PARALLEL execution of selected analysis modes
    - Stage 3: AI-powered synthesis of all findings
    """

    def __init__(self, config):
        self.config = config
        self.file_manager = FileManager()

        # Stage 1 components
        self.content_classifier = ContentClassifier()
        self.source_verifier = SourceVerifier(config)
        self.mode_router = ModeRouter()

        # Stage 2 orchestrators (initialized on demand)
        self._key_claims_orchestrator: Optional[KeyClaimsOrchestrator] = None
        self._bias_orchestrator: Optional[BiasCheckOrchestrator] = None
        self._manipulation_orchestrator: Optional[ManipulationOrchestrator] = None
        self._lie_detection_orchestrator: Optional[LieDetectorOrchestrator] = None

        # Stage 3: Report Synthesizer
        self._report_synthesizer: Optional[ReportSynthesizer] = None

        # R2 uploader for audit storage
        try:
            from utils.r2_uploader import R2Uploader
            self.r2_uploader = R2Uploader()
            self.r2_enabled = True
        except Exception as e:
            self.r2_enabled = False
            self.r2_uploader = None
            fact_logger.logger.warning(f"R2 not available: {e}")

        fact_logger.logger.info("ComprehensiveOrchestrator initialized (MetadataBlock architecture)")

    # =========================================================================
    # LAZY INITIALIZATION OF MODE ORCHESTRATORS
    # =========================================================================

    def _get_key_claims_orchestrator(self) -> KeyClaimsOrchestrator:
        """Lazy init for key claims orchestrator"""
        if self._key_claims_orchestrator is None:
            self._key_claims_orchestrator = KeyClaimsOrchestrator(self.config)
        return self._key_claims_orchestrator

    def _get_bias_orchestrator(self) -> BiasCheckOrchestrator:
        """Lazy init for bias orchestrator"""
        if self._bias_orchestrator is None:
            self._bias_orchestrator = BiasCheckOrchestrator(self.config)
        return self._bias_orchestrator

    def _get_manipulation_orchestrator(self) -> ManipulationOrchestrator:
        """Lazy init for manipulation orchestrator"""
        if self._manipulation_orchestrator is None:
            self._manipulation_orchestrator = ManipulationOrchestrator(self.config)
        return self._manipulation_orchestrator

    def _get_lie_detection_orchestrator(self) -> LieDetectorOrchestrator:
        """Lazy init for lie detection orchestrator"""
        if self._lie_detection_orchestrator is None:
            self._lie_detection_orchestrator = LieDetectorOrchestrator(self.config)
        return self._lie_detection_orchestrator

    def _get_report_synthesizer(self) -> ReportSynthesizer:
        """Lazy init for report synthesizer (Stage 3)"""
        if self._report_synthesizer is None:
            self._report_synthesizer = ReportSynthesizer()
        return self._report_synthesizer

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _check_cancellation(self, job_id: str):
        """Check if job was cancelled and raise exception if so"""
        if job_manager.is_cancelled(job_id):
            raise CancelledException("Job cancelled by user")

    def _send_stage_update(self, job_id: str, stage: str, message: str):
        """Send a stage update progress message"""
        job_manager.add_progress(job_id, message, details={"stage": stage})

    def _get_block_by_type(
        self, blocks: List[MetadataBlock], block_type: str
    ) -> Optional[MetadataBlock]:
        """Find a metadata block by its type identifier"""
        for block in blocks:
            if block.block_type == block_type:
                return block
        return None

    # =========================================================================
    # STAGE 1: PRE-ANALYSIS (produces MetadataBlocks)
    # =========================================================================

    async def _classify_content(self, content: str, job_id: str) -> Tuple[Optional[Dict], MetadataBlock]:
        """
        Step 1a helper: Run content classification.
        Returns (classification_data_dict, metadata_block).
        Designed to run in parallel with _verify_source().
        """
        self._send_stage_update(job_id, "content_classification", "Analyzing content type...")

        classification_result = await self.content_classifier.classify(content)

        if classification_result.success:
            data = classification_result.classification.model_dump()
            block = build_content_classification_block(
                classification_data=data,
                success=True,
                processing_time_ms=classification_result.processing_time_ms,
            )
            job_manager.add_progress(
                job_id,
                f"Content classified: {classification_result.classification.content_type} "
                f"({classification_result.classification.realm})"
            )
        else:
            data = {"error": classification_result.error}
            block = build_content_classification_block(
                classification_data=data,
                success=False,
                error=classification_result.error,
                processing_time_ms=classification_result.processing_time_ms,
            )

        job_manager.add_progress(job_id, "Content classification complete", details={
            "partial_result": {"content_classification": data}
        })

        return data, block

    async def _verify_source(self, source_url: Optional[str], job_id: str) -> Tuple[Optional[Dict], MetadataBlock]:
        """
        Step 1b helper: Run source verification.
        Returns (verification_data_dict, metadata_block).
        Designed to run in parallel with _classify_content().
        """
        self._send_stage_update(job_id, "source_verification", "Verifying source credibility...")

        if source_url:
            sv_start = time.time()
            verification_result = await self.source_verifier.verify_source(source_url)
            sv_time_ms = int((time.time() - sv_start) * 1000)

            if verification_result.report.verification_successful:
                data = {
                    "domain": verification_result.report.domain,
                    "credibility_tier": verification_result.report.credibility_tier,
                    "tier_description": verification_result.report.tier_description,
                    "verification_source": verification_result.report.verification_source,
                    "bias_rating": verification_result.report.bias_rating,
                    "factual_reporting": verification_result.report.factual_reporting,
                    "is_propaganda": verification_result.report.is_propaganda,
                    "verification_successful": verification_result.report.verification_successful
                }
                block = build_source_credibility_block(
                    verification_data=data,
                    success=True,
                    processing_time_ms=sv_time_ms,
                )
                job_manager.add_progress(
                    job_id,
                    f"Source verified: Tier {verification_result.report.credibility_tier} "
                    f"({verification_result.report.domain})"
                )
            else:
                err = verification_result.report.error or "Verification failed"
                data = {"error": err}
                block = build_source_credibility_block(
                    verification_data=data,
                    success=False,
                    error=err,
                    processing_time_ms=sv_time_ms,
                )
        else:
            data = {"status": "no_url_to_verify"}
            block = build_source_credibility_block(
                verification_data=data,
                success=False,
                error="No URL to verify",
            )
            job_manager.add_progress(job_id, "No source URL to verify")

        job_manager.add_progress(job_id, "Source verification complete", details={
            "partial_result": {"source_verification": data}
        })

        return data, block

    @traceable(name="comprehensive_stage1_preanalysis", run_type="chain", tags=["comprehensive", "stage1"])
    async def _run_stage1(
        self,
        content: str,
        job_id: str,
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Pre-Analysis (PARTIALLY PARALLEL)

        Steps 1a (classification) and 1b (source verification) run in parallel
        since they are completely independent. Step 1d (mode routing) runs after
        1a completes because it needs the classification result.

        Returns:
            Dict with:
            - metadata_blocks: List[MetadataBlock] (the scalable path)
            - content_classification: dict (backward compat for Stage 2 modes)
            - source_verification: dict (backward compat for Stage 2 modes)
            - mode_routing: dict (routing decision)
        """
        metadata_blocks: List[MetadataBlock] = []

        try:
            self._check_cancellation(job_id)

            # =================================================================
            # Steps 1a + 1b: PARALLEL (classification + source verification)
            # =================================================================
            # These two operations are completely independent:
            #   - Classification: LLM call to analyze content type/realm
            #   - Source Verification: Supabase/MBFC lookup for credibility
            # Running them in parallel saves ~3-5s.

            parallel_results = await asyncio.gather(
                self._classify_content(content, job_id),
                self._verify_source(source_url, job_id),
                return_exceptions=True
            )

            # Process classification result
            if isinstance(parallel_results[0], BaseException):
                fact_logger.logger.error(f"Classification failed: {parallel_results[0]}")
                content_classification_data = {"error": str(parallel_results[0])}
                cc_block = build_content_classification_block(
                    classification_data=content_classification_data,
                    success=False,
                    error=str(parallel_results[0]),
                )
            else:
                content_classification_data, cc_block = parallel_results[0]

            metadata_blocks.append(cc_block)

            # Process source verification result
            if isinstance(parallel_results[1], BaseException):
                fact_logger.logger.error(f"Source verification failed: {parallel_results[1]}")
                source_verification_data = {"error": str(parallel_results[1])}
                sv_block = build_source_credibility_block(
                    verification_data=source_verification_data,
                    success=False,
                    error=str(parallel_results[1]),
                )
            else:
                source_verification_data, sv_block = parallel_results[1]

            metadata_blocks.append(sv_block)

            # =================================================================
            # Step 1c: (Future checks go here)
            # Each new check:
            #   1. Add a helper method like _classify_content / _verify_source
            #   2. Add it to the asyncio.gather() call above
            #   3. Process its result and append the block
            # That's it -- synthesizer picks them up automatically.
            # =================================================================

            # =================================================================
            # Step 1d: Mode Routing (SEQUENTIAL -- depends on classification)
            # =================================================================
            self._send_stage_update(job_id, "mode_routing", "Selecting analysis modes...")
            self._check_cancellation(job_id)

            routing_result = await self.mode_router.route(
                content_classification=content_classification_data or {},
                source_verification=source_verification_data,
                author_info=None,  # Future enhancement
            )

            if routing_result.success:
                mode_routing = {
                    "selected_modes": routing_result.selection.selected_modes,
                    "excluded_modes": routing_result.selection.excluded_modes,
                    "exclusion_rationale": routing_result.selection.exclusion_rationale,
                    "routing_reasoning": routing_result.selection.routing_reasoning,
                    "routing_confidence": routing_result.selection.routing_confidence
                }
                job_manager.add_progress(
                    job_id,
                    f"Selected modes: {', '.join(routing_result.selection.selected_modes)}"
                )
            else:
                mode_routing = {
                    "selected_modes": ["key_claims_analysis"],
                    "excluded_modes": [],
                    "routing_reasoning": "Default selection due to routing error"
                }

            # Send partial result
            job_manager.add_progress(job_id, "Mode routing complete", details={
                "partial_result": {"mode_routing": mode_routing}
            })

            return {
                # The new scalable path
                "metadata_blocks": metadata_blocks,

                # Backward-compatible keys for Stage 2 mode orchestrators
                "content_classification": content_classification_data,
                "source_verification": source_verification_data,
                "author_info": None,  # Future enhancement

                # Routing decision (not a metadata block -- it's control flow)
                "mode_routing": mode_routing,
            }

        except CancelledException:
            raise
        except Exception as e:
            fact_logger.logger.error(f"Stage 1 error: {e}")
            raise

    # =========================================================================
    # STAGE 2: SEQUENTIAL MODE EXECUTION
    # =========================================================================

    async def _run_single_mode(
        self,
        mode_id: str,
        content: str,
        job_id: str,
        stage1_results: Dict[str, Any],
        shared_scraper=None
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        """
        Run a single analysis mode

        Args:
            shared_scraper: Optional ScrapeCache for deduplicating URL scrapes
                           across parallel modes. Passed to scraping-heavy modes
                           (key_claims, manipulation). Non-scraping modes ignore it.

        Returns: (mode_id, result_dict, error_message)
        """
        try:
            self._check_cancellation(job_id)

            # Get source context for modes that need it
            source_context = stage1_results.get("content_classification") or {}
            source_credibility = stage1_results.get("source_verification") or {}

            if mode_id == "key_claims_analysis":
                orchestrator = self._get_key_claims_orchestrator()
                result = await orchestrator.process_with_progress(
                    text_content=content,
                    job_id=job_id,
                    source_context=source_context,
                    source_credibility=source_credibility,
                    standalone=False,  # Prevent overwriting comprehensive result
                    shared_scraper=shared_scraper  # Phase 3: shared URL cache
                )
                return (mode_id, result, None)

            elif mode_id == "bias_analysis":
                orchestrator = self._get_bias_orchestrator()
                publication_name = source_credibility.get("domain", "")

                result = await orchestrator.process_with_progress(
                    text=content,
                    publication_name=publication_name,
                    source_credibility=source_credibility if source_credibility else None,
                    job_id=job_id,
                    standalone=False  # Prevent overwriting comprehensive result
                    # bias_analysis doesn't scrape -- no shared_scraper needed
                )
                return (mode_id, result, None)

            elif mode_id == "manipulation_detection":
                orchestrator = self._get_manipulation_orchestrator()

                result = await orchestrator.process_with_progress(
                    content=content,
                    job_id=job_id,
                    source_info=source_credibility.get("domain", "Unknown"),
                    source_credibility=source_credibility if source_credibility else None,
                    standalone=False,  # Prevent overwriting comprehensive result
                    shared_scraper=shared_scraper  # Phase 3: shared URL cache
                )
                return (mode_id, result, None)

            elif mode_id == "lie_detection":
                orchestrator = self._get_lie_detection_orchestrator()

                result = await orchestrator.process_with_progress(
                    text=content,
                    job_id=job_id,
                    source_credibility=source_credibility if source_credibility else None,
                    standalone=False  # Prevent overwriting comprehensive result
                    # lie_detection doesn't scrape -- no shared_scraper needed
                )
                return (mode_id, result, None)

            elif mode_id == "llm_output_verification":
                from orchestrator.llm_output_orchestrator import LLMInterpretationOrchestrator
                orchestrator = LLMInterpretationOrchestrator(self.config)

                result = await orchestrator.process_with_progress(
                    html_content=content,
                    job_id=job_id
                )
                return (mode_id, result, None)

            else:
                return (mode_id, None, f"Unknown mode: {mode_id}")

        except CancelledException:
            raise
        except Exception as e:
            fact_logger.logger.error(f"Mode {mode_id} failed: {e}")
            return (mode_id, None, str(e))

    @traceable(name="comprehensive_stage2_execution", run_type="chain", tags=["comprehensive", "stage2"])
    async def _run_stage2(
        self,
        content: str,
        job_id: str,
        stage1_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 2: Mode Execution (PARALLEL with Shared Scrape Cache)

        Runs ALL selected modes simultaneously using asyncio.gather().
        A shared ScrapeCache is created for this analysis and passed to
        scraping-heavy modes (key_claims, manipulation). When both modes
        discover overlapping URLs, each URL is scraped only once.

        standalone=False is passed to every mode to prevent any mode from calling
        job_manager.complete_job(), which would prematurely close the SSE stream
        while other modes are still running.
        """
        self._send_stage_update(job_id, "mode_execution", "Running selected analysis modes...")

        mode_routing = stage1_results.get("mode_routing") or {}
        selected_modes = mode_routing.get("selected_modes", ["key_claims_analysis"])

        job_manager.add_progress(
            job_id,
            f"Executing {len(selected_modes)} modes in PARALLEL: {', '.join(selected_modes)}"
        )

        start_time = time.time()

        # =================================================================
        # Phase 3: Create shared ScrapeCache for this analysis
        # =================================================================
        # When key_claims and manipulation run in parallel, they often
        # search the web and discover overlapping source URLs. The shared
        # cache ensures each URL is scraped at most once, saving browser
        # resources and ~30-50% of scraping time.
        #
        # The cache is passed to scraping-heavy modes (key_claims,
        # manipulation) and ignored by non-scraping modes (bias, lie_detection).
        # It lives only for the duration of this Stage 2 execution.

        shared_scraper = ScrapeCache(self.config)

        try:
            # =============================================================
            # PARALLEL EXECUTION: Launch all modes simultaneously
            # =============================================================
            # Each mode gets its own orchestrator instance with independent:
            #   - LLM instances (fresh API key per call via get_openai_llm)
            #   - LangSmith callbacks (scoped by component name)
            # Scraping-heavy modes share the ScrapeCache instead of creating
            # independent BrowserlessScraper instances.

            tasks = [
                self._run_single_mode(
                    mode_id, content, job_id, stage1_results,
                    shared_scraper=shared_scraper
                )
                for mode_id in selected_modes
            ]

            fact_logger.logger.info(
                f"Launching {len(tasks)} mode tasks in parallel (shared scrape cache)",
                extra={
                    "modes": selected_modes,
                    "job_id": job_id
                }
            )

            # asyncio.gather runs all tasks concurrently.
            # return_exceptions=True ensures one mode failing doesn't cancel the rest.
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            # =============================================================
            # CLEANUP: Close shared scraper (releases browser resources)
            # =============================================================
            # Always close, even if modes failed or were cancelled.
            try:
                await shared_scraper.close()
            except Exception as cleanup_err:
                fact_logger.logger.debug(f"ScrapeCache cleanup: {cleanup_err}")

        # =================================================================
        # PROCESS RESULTS from parallel execution
        # =================================================================
        mode_reports: Dict[str, Any] = {}
        mode_errors: Dict[str, str] = {}

        for i, result in enumerate(raw_results):
            mode_id = selected_modes[i]

            try:
                # Check if asyncio.gather caught an exception for this task
                if isinstance(result, CancelledException):
                    raise result
                elif isinstance(result, BaseException):
                    fact_logger.logger.error(f"Mode {mode_id} raised exception: {result}")
                    mode_errors[mode_id] = str(result)
                    job_manager.add_progress(job_id, f"[{mode_id}] failed: {str(result)[:100]}")
                    continue

                # Normal result: tuple of (mode_id, result_dict, error_message)
                if isinstance(result, tuple) and len(result) == 3:
                    mode_id_returned, mode_result, error = result

                    if error:
                        mode_errors[mode_id] = error
                        job_manager.add_progress(job_id, f"[{mode_id}] failed: {error}")
                    elif mode_result:
                        mode_reports[mode_id] = mode_result
                        job_manager.add_progress(job_id, f"[{mode_id}] complete")
                else:
                    fact_logger.logger.error(
                        f"Unexpected result format from {mode_id}: {type(result)}"
                    )
                    mode_errors[mode_id] = f"Unexpected result format: {type(result)}"

            except CancelledException:
                raise
            except Exception as e:
                fact_logger.logger.error(f"Error processing result for {mode_id}: {e}")
                mode_errors[mode_id] = str(e)

        execution_time = time.time() - start_time

        # Log parallel performance metrics
        fact_logger.logger.info(
            f"Stage 2 complete in {execution_time:.1f}s (PARALLEL execution)",
            extra={
                "modes_run": len(selected_modes),
                "modes_succeeded": len(mode_reports),
                "modes_failed": len(mode_errors),
                "execution_mode": "parallel",
                "execution_time": round(execution_time, 2)
            }
        )

        return {
            "mode_reports": mode_reports,
            "mode_errors": mode_errors,
            "execution_time_seconds": round(execution_time, 2)
        }

    # =========================================================================
    # STAGE 3: AI-POWERED SYNTHESIS
    # =========================================================================

    @traceable(name="comprehensive_stage3_synthesis", run_type="chain", tags=["comprehensive", "stage3"])
    async def _run_stage3(
        self,
        job_id: str,
        stage1_results: Dict[str, Any],
        stage2_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stage 3: AI-Powered Synthesis

        Passes metadata blocks + mode reports to the ReportSynthesizer,
        which dynamically assembles its prompt from all available blocks.

        Args:
            job_id: Job ID for progress tracking
            stage1_results: Results from Stage 1 (includes metadata_blocks)
            stage2_results: Results from Stage 2 (mode_reports, mode_errors)

        Returns:
            Complete synthesis report as dictionary
        """
        start_time = time.time()

        self._send_stage_update(job_id, "synthesis", "Synthesizing final report...")
        job_manager.add_progress(job_id, "Running AI-powered report synthesis...")

        try:
            synthesizer = self._get_report_synthesizer()

            # Pass metadata blocks to synthesizer (the new scalable path)
            synthesis_report = await synthesizer.synthesize(
                stage1_results=stage1_results,
                stage2_results=stage2_results
            )

            synthesis_dict = synthesis_report.model_dump()
            synthesis_dict["synthesis_time_seconds"] = round(time.time() - start_time, 2)

            total_concerns = len(synthesis_report.key_concerns)

            job_manager.add_progress(
                job_id,
                f"Synthesis complete: Score {synthesis_report.overall_score}/100 "
                f"({synthesis_report.overall_rating}), "
                f"{total_concerns} concerns identified"
            )

            fact_logger.logger.info(
                "Stage 3 synthesis complete",
                extra={
                    "overall_score": synthesis_report.overall_score,
                    "rating": synthesis_report.overall_rating,
                    "confidence": synthesis_report.confidence,
                    "concerns_count": total_concerns,
                    "positives_count": len(synthesis_report.positive_indicators),
                    "modes_analyzed": len(synthesis_report.modes_analyzed),
                    "synthesis_time": synthesis_dict["synthesis_time_seconds"]
                }
            )

            return synthesis_dict

        except Exception as e:
            fact_logger.logger.error(f"Stage 3 synthesis failed: {e}")
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")

            job_manager.add_progress(job_id, f"AI synthesis failed, using basic synthesis: {str(e)[:100]}")

            return self._run_stage3_fallback(job_id, stage1_results, stage2_results)

    def _run_stage3_fallback(
        self,
        job_id: str,
        stage1_results: Dict[str, Any],
        stage2_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback Stage 3: Generic block-based scoring when AI synthesis fails.

        Iterates over ALL metadata blocks and collects their impact signals,
        then extracts key metrics from mode reports. This is fully generic --
        new metadata blocks contribute to the score without any code changes here.
        """
        self._send_stage_update(job_id, "synthesis", "Running basic synthesis (fallback)...")

        mode_reports = stage2_results.get("mode_reports", {})
        mode_errors = stage2_results.get("mode_errors", {})
        metadata_blocks: List[MetadataBlock] = stage1_results.get("metadata_blocks", [])

        # Start with neutral score
        score = 50.0

        # Collect flags and positives from ALL metadata blocks generically
        all_flags = []
        all_positives = []

        for block in metadata_blocks:
            if not block.success:
                continue

            impact = block.impact
            score += impact.score_adjustment

            for flag in impact.flags:
                all_flags.append({
                    "severity": impact.flag_severity,
                    "description": flag,
                    "source": block.display_name,
                    "category": impact.flag_category,
                })

            for positive in impact.positives:
                all_positives.append(positive)

        # ---- Mode report scoring (same logic as before) ----

        # Key Claims
        if "key_claims_analysis" in mode_reports:
            kc = mode_reports["key_claims_analysis"]
            summary = kc.get("summary", {})
            avg_conf = safe_float(summary.get("average_confidence", 0.5))

            if avg_conf >= 0.7:
                score += 15
                all_positives.append(
                    f"Key claims verified with {avg_conf:.0%} average confidence"
                )
            elif avg_conf < 0.4:
                score -= 20
                all_flags.append({
                    "severity": "high",
                    "description": f"Low fact verification confidence ({avg_conf:.0%})",
                    "source": "Key Claims Analysis",
                    "category": "factual_accuracy",
                })

        # Bias
        if "bias_analysis" in mode_reports:
            bias = mode_reports["bias_analysis"]
            analysis = bias.get("analysis", {})
            bias_score = abs(safe_float(analysis.get("consensus_bias_score", 0)))
            direction = analysis.get("consensus_direction", "Unknown")

            if bias_score > 6:
                score -= 15
                all_flags.append({
                    "severity": "medium" if bias_score <= 7 else "high",
                    "description": f"Significant {direction} bias detected (score: {bias_score:.1f}/10)",
                    "source": "Bias Analysis",
                    "category": "bias",
                })
            elif bias_score <= 3:
                score += 10
                all_positives.append("Low bias detected -- content appears balanced")

        # Manipulation
        if "manipulation_detection" in mode_reports:
            manip = mode_reports["manipulation_detection"]
            manip_score = safe_float(manip.get("manipulation_score", 0))

            if manip_score > 6:
                score -= 20
                all_flags.append({
                    "severity": "high",
                    "description": f"High manipulation score ({manip_score:.1f}/10)",
                    "source": "Manipulation Detection",
                    "category": "manipulation",
                })
            elif manip_score < 3:
                all_positives.append("Low manipulation indicators")

        # Lie Detection
        if "lie_detection" in mode_reports:
            lie = mode_reports["lie_detection"]
            deception_score = safe_float(lie.get(
                "deception_likelihood_score",
                lie.get("overall_score", 0)
            ))

            if deception_score > 6:
                score -= 10
                all_flags.append({
                    "severity": "medium",
                    "description": f"Elevated deception indicators ({deception_score}/10)",
                    "source": "Lie Detection",
                    "category": "manipulation",
                })

        # Clamp score
        score = max(0, min(100, score))

        # Determine rating
        if score >= 80:
            rating = "Highly Credible"
        elif score >= 65:
            rating = "Credible"
        elif score >= 45:
            rating = "Mixed"
        elif score >= 25:
            rating = "Low Credibility"
        else:
            rating = "Unreliable"

        # Build narrative
        concern_texts = [f.get("description", "") for f in all_flags]
        if not all_flags:
            narrative = (
                f"Based on automated analysis, this content appears to have {rating.lower()} "
                f"credibility. No major concerns were flagged, though detailed AI synthesis "
                f"was unavailable."
            )
        else:
            narrative = (
                f"Our analysis raised {len(all_flags)} concern(s) about this content, "
                f"resulting in a {rating.lower()} credibility rating. "
                f"Please review the individual findings below for details."
            )

        # Build synthesis dict matching SynthesisReport schema
        synthesis: Dict[str, Any] = {
            "overall_score": int(score),
            "overall_rating": rating,
            "confidence": 40,  # Low confidence due to fallback
            "summary": narrative,
            "key_concerns": concern_texts,
            "positive_indicators": all_positives,
            "recommendations": [
                "Review individual mode reports for detailed findings",
                "Consider re-running analysis for full AI synthesis"
            ],
            "modes_analyzed": list(mode_reports.keys()),
            "analysis_notes": "AI synthesis unavailable -- using automated metric extraction",
        }

        job_manager.add_progress(
            job_id,
            f"Basic synthesis complete: {int(score)}/100 "
            f"({rating}), {len(all_flags)} flags"
        )

        return synthesis

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    @traceable(
        name="comprehensive_analysis_pipeline",
        run_type="chain",
        tags=["comprehensive", "full-pipeline"]
    )
    async def process_with_progress(
        self,
        content: str,
        job_id: str,
        source_url: Optional[str] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete comprehensive analysis pipeline

        Args:
            content: Text content to analyze
            job_id: Job ID for progress tracking
            source_url: Optional URL of the content source
            user_preferences: Optional user mode preferences

        Returns:
            Complete comprehensive analysis result
        """
        session_id = self.file_manager.create_session()
        start_time = time.time()

        fact_logger.logger.info(
            "Starting comprehensive analysis pipeline",
            extra={
                "session_id": session_id,
                "job_id": job_id,
                "content_length": len(content),
                "has_source_url": source_url is not None
            }
        )

        try:
            # ================================================================
            # STAGE 1: Pre-Analysis
            # ================================================================
            job_manager.add_progress(job_id, "Stage 1: Pre-analysis starting...")
            stage1_results = await self._run_stage1(content, job_id, source_url)

            self._check_cancellation(job_id)
            job_manager.add_progress(job_id, "Stage 1 complete")

            # ================================================================
            # STAGE 2: Parallel Mode Execution
            # ================================================================
            job_manager.add_progress(job_id, "Stage 2: Parallel mode execution starting...")
            stage2_results = await self._run_stage2(content, job_id, stage1_results)

            self._check_cancellation(job_id)
            job_manager.add_progress(job_id, "Stage 2 complete")

            # ================================================================
            # STAGE 3: AI-Powered Synthesis
            # ================================================================
            job_manager.add_progress(job_id, "Stage 3: Synthesizing results...")
            stage3_results = await self._run_stage3(job_id, stage1_results, stage2_results)

            job_manager.add_progress(job_id, "Stage 3 complete")

            # ================================================================
            # COMPILE FINAL RESULT
            # ================================================================
            processing_time = time.time() - start_time

            # Convert metadata blocks to frontend-ready dicts
            metadata_blocks: List[MetadataBlock] = stage1_results.get("metadata_blocks", [])
            metadata_blocks_for_frontend = [b.to_frontend_dict() for b in metadata_blocks]

            final_result: Dict[str, Any] = {
                "success": True,
                "session_id": session_id,
                "processing_time": round(processing_time, 2),

                # Stage 1: metadata blocks (the new scalable path)
                "metadata_blocks": metadata_blocks_for_frontend,

                # Stage 1: backward-compatible individual keys
                # (kept so existing frontend renderers still work)
                "content_classification": stage1_results.get("content_classification"),
                "source_verification": stage1_results.get("source_verification"),
                "author_info": stage1_results.get("author_info"),
                "mode_routing": stage1_results.get("mode_routing"),

                # Stage 2 results
                "mode_reports": stage2_results.get("mode_reports", {}),
                "mode_errors": stage2_results.get("mode_errors", {}),
                "mode_execution_time": stage2_results.get("execution_time_seconds"),

                # Stage 3 results
                "synthesis_report": stage3_results,

                # Metadata
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "3.0.0"  # MetadataBlock architecture
            }

            # Complete the job
            job_manager.complete_job(job_id, final_result)

            fact_logger.logger.info(
                "Comprehensive analysis complete",
                extra={
                    "session_id": session_id,
                    "processing_time": processing_time,
                    "modes_run": len(stage2_results.get("mode_reports", {})),
                    "metadata_blocks": len(metadata_blocks),
                    "overall_score": stage3_results.get("overall_score",
                                     stage3_results.get("overall_credibility_score")),
                    "overall_rating": stage3_results.get("overall_rating",
                                      stage3_results.get("overall_credibility_rating")),
                }
            )

            return final_result

        except CancelledException:
            job_manager.add_progress(job_id, "Analysis cancelled by user")
            raise

        except Exception as e:
            fact_logger.logger.error(f"Comprehensive analysis failed: {e}")
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")

            job_manager.fail_job(job_id, str(e))
            raise


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    async def test():
        config = type('Config', (), {
            'openai_api_key': os.getenv("OPENAI_API_KEY"),
            'brave_api_key': os.getenv("BRAVE_API_KEY"),
            'browserless_endpoint': os.getenv("BROWSER_PLAYWRIGHT_ENDPOINT_PRIVATE")
        })()

        orchestrator = ComprehensiveOrchestrator(config)

        test_content = """
        According to recent reports, global temperatures have risen by 1.2C since 
        pre-industrial times. Scientists warn that without immediate action, we could 
        see a 3C increase by 2100, leading to catastrophic consequences including 
        more frequent extreme weather events, rising sea levels, and mass extinction 
        of species.

        The IPCC's latest assessment suggests that current policies put us on track 
        for 2.7C of warming, well above the Paris Agreement target of 1.5C.
        """

        test_job_id = "test_comprehensive_blocks"
        job_manager.create_job(test_job_id)

        try:
            result = await orchestrator.process_with_progress(
                content=test_content,
                job_id=test_job_id,
                source_url=None
            )

            print("\n" + "="*70)
            print("COMPREHENSIVE ANALYSIS RESULT")
            print("="*70)
            print(f"Success: {result.get('success')}")
            print(f"Processing Time: {result.get('processing_time')}s")
            print(f"Version: {result.get('version')}")

            # Metadata Blocks
            blocks = result.get("metadata_blocks", [])
            print(f"\nMetadata Blocks ({len(blocks)}):")
            for block in blocks:
                print(f"  - {block['display_name']} (success={block['success']})")

            # Synthesis
            synthesis = result.get("synthesis_report", {})
            print(f"\nOverall Score: {synthesis.get('overall_score')}/100")
            print(f"Rating: {synthesis.get('overall_rating')}")
            print(f"Confidence: {synthesis.get('confidence')}%")
            print(f"\nSummary:\n{synthesis.get('summary')}")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())
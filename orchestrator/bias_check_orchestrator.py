# orchestrator/bias_check_orchestrator.py
"""
Bias Check Orchestrator - WITH MBFC INTEGRATION + PRE-FETCHED CREDIBILITY SUPPORT
Coordinates the complete bias checking workflow with R2 uploads and MBFC lookup

UPDATED: Now accepts pre-fetched source_credibility from article fetch to avoid
duplicate MBFC lookups when credibility was already retrieved.
"""

from langsmith import traceable
import time
import json
from typing import Optional, Dict, Any

from agents.bias_checker import BiasChecker
from agents.publication_bias_detector import PublicationBiasDetector
from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.file_manager import FileManager
from utils.r2_uploader import R2Uploader

# NEW: Import credibility context builder
try:
    from utils.credibility_context import build_bias_analysis_context
except ImportError:
    # Fallback if module not yet added
    def build_bias_analysis_context(source_credibility=None, publication_name=None):
        if not source_credibility:
            return f"\nPUBLICATION: {publication_name}\n" if publication_name else ""
        parts = ["\nMEDIA BIAS/FACT CHECK DATA:"]
        if source_credibility.get('publication_name'):
            parts.append(f"Publication: {source_credibility['publication_name']}")
        if source_credibility.get('bias_rating'):
            parts.append(f"Bias: {source_credibility['bias_rating']}")
        if source_credibility.get('factual_reporting'):
            parts.append(f"Factual: {source_credibility['factual_reporting']}")
        return "\n".join(parts)


class BiasCheckOrchestrator:
    """
    Orchestrates bias checking workflow with R2 storage and MBFC integration

    Pipeline:
    1. Receive text + optional publication URL OR pre-fetched credibility
    2. If credibility provided, use it directly (skip MBFC lookup)
    3. If URL provided without credibility, look up publication on MBFC
    4. Run multi-model bias analysis (GPT-4o + Claude)
    5. Combine analyses into comprehensive report
    6. Save reports locally
    7. Upload to Cloudflare R2
    8. Return combined assessment
    """

    def __init__(self, config):
        self.config = config
        self.bias_checker = BiasChecker(config)
        self.file_manager = FileManager()

        # Initialize R2 uploader
        try:
            self.r2_uploader = R2Uploader()
            self.r2_enabled = True
            fact_logger.logger.info("✅ Cloudflare R2 integration enabled")
        except Exception as e:
            fact_logger.logger.warning(f"⚠️ Cloudflare R2 not configured: {e}")
            self.r2_enabled = False

        # Initialize MBFC lookup components
        self.mbfc_enabled = False
        self.brave_searcher = None
        self.scraper = None
        self.pub_detector = None

        try:
            # Check if Brave API is available
            if hasattr(config, 'brave_api_key') and config.brave_api_key:
                from utils.brave_searcher import BraveSearcher
                from utils.browserless_scraper import BrowserlessScraper

                self.brave_searcher = BraveSearcher(config, max_results=5)
                self.scraper = BrowserlessScraper(config)
                self.pub_detector = PublicationBiasDetector(
                    config=config,
                    brave_searcher=self.brave_searcher,
                    scraper=self.scraper
                )
                self.mbfc_enabled = True
                fact_logger.logger.info("✅ MBFC lookup integration enabled")
            else:
                # Fallback to local-only publication detector
                self.pub_detector = PublicationBiasDetector()
                fact_logger.logger.info("ℹ️ MBFC lookup disabled (no Brave API key) - using local database only")
        except Exception as e:
            fact_logger.logger.warning(f"⚠️ MBFC integration failed: {e} - using local database only")
            self.pub_detector = PublicationBiasDetector()

        fact_logger.log_component_start(
            "BiasCheckOrchestrator",
            r2_enabled=self.r2_enabled,
            mbfc_enabled=self.mbfc_enabled
        )

    def _check_cancellation(self, job_id: str):
        """Check if job has been cancelled and raise exception if so"""
        from utils.job_manager import job_manager
        if job_manager.is_cancelled(job_id):
            raise Exception("Job cancelled by user")

    def _build_publication_context_from_credibility(
        self, 
        source_credibility: Dict[str, Any]
    ) -> str:
        """
        Build publication context string from pre-fetched credibility data.
        This is used when we already have credibility from the article fetch.

        Args:
            source_credibility: Dict with tier, bias_rating, factual_reporting, etc.

        Returns:
            Formatted context string for bias analysis prompts
        """
        return build_bias_analysis_context(
            source_credibility=source_credibility,
            publication_name=source_credibility.get('publication_name')
        )

    def _convert_credibility_to_profile_data(
        self, 
        source_credibility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert pre-fetched credibility dict to publication_profile format
        for consistency with MBFC lookup results.

        Args:
            source_credibility: Dict from SourceCredibilityService

        Returns:
            Dict matching publication_profile structure
        """
        # Map tier to political leaning approximation
        # (This is imperfect but provides some context)
        tier = source_credibility.get('tier', 3)

        # Extract bias direction from bias_rating if available
        bias_rating = source_credibility.get('bias_rating', '')
        political_leaning = 'center'  # default

        if bias_rating:
            bias_upper = bias_rating.upper()
            if 'LEFT' in bias_upper and 'CENTER' not in bias_upper:
                political_leaning = 'left' if 'FAR' not in bias_upper else 'far-left'
            elif 'LEFT-CENTER' in bias_upper:
                political_leaning = 'center-left'
            elif 'RIGHT' in bias_upper and 'CENTER' not in bias_upper:
                political_leaning = 'right' if 'FAR' not in bias_upper else 'far-right'
            elif 'RIGHT-CENTER' in bias_upper:
                political_leaning = 'center-right'
            elif 'CENTER' in bias_upper:
                political_leaning = 'center'

        return {
            "name": source_credibility.get('publication_name', 'Unknown'),
            "political_leaning": political_leaning,
            "bias_rating": source_credibility.get('bias_score'),
            "factual_reporting": source_credibility.get('factual_reporting'),
            "credibility_rating": source_credibility.get('rating') or source_credibility.get('credibility_rating'),
            "assigned_tier": tier,
            "mbfc_url": source_credibility.get('mbfc_url'),
            "source": source_credibility.get('source', 'prefetched'),
            "is_propaganda": source_credibility.get('is_propaganda', False),
            "special_tags": source_credibility.get('special_tags', [])
        }

    @traceable(
        name="bias_check_pipeline",
        run_type="chain",
        tags=["orchestrator", "bias-checking", "multi-model", "r2-storage", "mbfc"]
    )
    async def process(
        self, 
        text: str, 
        publication_url: Optional[str] = None,
        publication_name: Optional[str] = None,
        source_credibility: Optional[Dict[str, Any]] = None,  # NEW PARAMETER
        save_to_r2: bool = True
    ) -> dict:
        """
        Complete bias checking pipeline with R2 storage and MBFC lookup

        Args:
            text: Text to analyze for bias
            publication_url: Optional publication URL for MBFC lookup
            publication_name: Optional publication name for metadata (backward compat)
            source_credibility: Optional pre-fetched credibility data (NEW - preferred)
                               If provided, skips MBFC lookup entirely
            save_to_r2: Whether to save reports to Cloudflare R2

        Returns:
            Dictionary with complete bias analysis results
        """
        session_id = self.file_manager.create_session()
        start_time = time.time()

        # Determine if we're using pre-fetched credibility
        using_prefetched = source_credibility is not None

        fact_logger.logger.info(
            f"🚀 STARTING BIAS CHECK SESSION: {session_id}",
            extra={
                "session_id": session_id,
                "text_length": len(text),
                "publication_url": publication_url,
                "publication_name": publication_name,
                "using_prefetched_credibility": using_prefetched,
                "r2_enabled": self.r2_enabled and save_to_r2,
                "mbfc_enabled": self.mbfc_enabled
            }
        )

        try:
            # Step 0: Resolve publication context
            publication_profile = None
            publication_profile_data = None
            mbfc_context = None
            resolved_publication_name = publication_name

            # OPTION A: Use pre-fetched credibility (skip MBFC lookup)
            if source_credibility:
                fact_logger.logger.info(
                    "📰 Step 0: Using pre-fetched credibility data",
                    extra={
                        "tier": source_credibility.get('tier'),
                        "bias": source_credibility.get('bias_rating'),
                        "source": source_credibility.get('source', 'prefetched')
                    }
                )

                # Build context from pre-fetched data
                mbfc_context = self._build_publication_context_from_credibility(source_credibility)
                publication_profile_data = self._convert_credibility_to_profile_data(source_credibility)
                resolved_publication_name = source_credibility.get('publication_name', publication_name)

            # OPTION B: Do MBFC lookup (original behavior)
            elif publication_url and self.mbfc_enabled and self.pub_detector:
                fact_logger.logger.info(f"📰 Step 0: Looking up publication on MBFC: {publication_url}")

                try:
                    publication_profile = await self.pub_detector.detect_publication_async(
                        publication_url=publication_url
                    )

                    if publication_profile:
                        resolved_publication_name = publication_profile.name
                        mbfc_context = self.pub_detector.get_publication_context(
                            profile=publication_profile
                        )
                        publication_profile_data = publication_profile.model_dump()
                        fact_logger.logger.info(
                            f"✅ MBFC data found: {publication_profile.name}",
                            extra={
                                "source": publication_profile.source,
                                "bias": publication_profile.political_leaning,
                                "factual_reporting": publication_profile.factual_reporting,
                                "credibility": publication_profile.credibility_rating
                            }
                        )
                    else:
                        fact_logger.logger.info(f"📭 No MBFC data found for: {publication_url}")

                except Exception as e:
                    fact_logger.logger.warning(f"⚠️ MBFC lookup failed: {e}")
                    # Continue without MBFC data

            # OPTION C: Local database lookup
            elif publication_url and self.pub_detector:
                domain = self.pub_detector.clean_url_to_domain(publication_url)
                publication_profile = self.pub_detector.detect_publication(domain)
                if publication_profile:
                    resolved_publication_name = publication_profile.name
                    mbfc_context = self.pub_detector.get_publication_context(
                        publication_name=resolved_publication_name
                    )
                    publication_profile_data = publication_profile.model_dump()
                    fact_logger.logger.info(f"📚 Using local database for: {domain}")

            # OPTION D: Name-based lookup (backward compatibility)
            elif publication_name and self.pub_detector:
                publication_profile = self.pub_detector.detect_publication(publication_name)
                if publication_profile:
                    mbfc_context = self.pub_detector.get_publication_context(
                        publication_name=publication_name
                    )
                    publication_profile_data = publication_profile.model_dump()

            # Step 1: Run bias analysis
            fact_logger.logger.info("📊 Step 1: Multi-model bias analysis")

            bias_results = await self.bias_checker.check_bias(
                text=text,
                publication_name=resolved_publication_name
            )

            # Step 2: Prepare report data
            fact_logger.logger.info("📝 Step 2: Preparing reports")

            report_data = {
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "publication_url": publication_url,
                "publication_name": resolved_publication_name,
                "used_prefetched_credibility": using_prefetched,
                "text_analyzed": text[:500] + "..." if len(text) > 500 else text,
                "gpt_analysis": bias_results["gpt_analysis"],
                "claude_analysis": bias_results["claude_analysis"],
                "combined_report": bias_results["combined_report"],
                "publication_profile": publication_profile_data,
                "mbfc_context": mbfc_context,
                "processing_time": bias_results["processing_time"]
            }

            # Step 3: Save reports locally
            fact_logger.logger.info("💾 Step 3: Saving reports locally")

            combined_report_path = self.file_manager.save_session_file(
                session_id,
                "combined_bias_report.json",
                json.dumps(report_data, indent=2)
            )

            gpt_report_path = self.file_manager.save_session_file(
                session_id,
                "gpt_bias_analysis.json",
                json.dumps(bias_results["gpt_analysis"], indent=2)
            )

            claude_report_path = self.file_manager.save_session_file(
                session_id,
                "claude_bias_analysis.json",
                json.dumps(bias_results["claude_analysis"], indent=2)
            )

            # Step 4: Upload to Cloudflare R2
            r2_upload_status = {"success": False, "error": "R2 not enabled"}
            r2_links = []

            if self.r2_enabled and save_to_r2:
                fact_logger.logger.info("☁️ Step 4: Uploading to Cloudflare R2")

                try:
                    uploads = [
                        ("combined_report.json", combined_report_path),
                        ("gpt_analysis.json", gpt_report_path),
                        ("claude_analysis.json", claude_report_path)
                    ]

                    for filename, filepath in uploads:
                        with open(filepath, 'r') as f:
                            content = f.read()

                        r2_key = f"bias-reports/{session_id}/{filename}"
                        url = self.r2_uploader.upload_text(
                            content=content,
                            key=r2_key,
                            content_type="application/json"
                        )

                        if url:
                            r2_links.append(url)
                            fact_logger.logger.info(f"✅ Uploaded: {url}")

                    r2_upload_status = {
                        "success": len(r2_links) == 3,
                        "uploaded_files": len(r2_links),
                        "links": r2_links
                    }

                    fact_logger.logger.info(
                        "✅ All bias reports uploaded to R2",
                        extra={"num_uploads": len(r2_links), "session_id": session_id}
                    )

                except Exception as e:
                    fact_logger.logger.error(f"❌ R2 upload failed: {e}")
                    r2_upload_status = {"success": False, "error": str(e)}

            # Step 5: Build final output
            duration = time.time() - start_time
            combined = bias_results["combined_report"]

            output = {
                "success": True,
                "session_id": session_id,
                "status": "completed",
                "processing_time": duration,
                "used_prefetched_credibility": using_prefetched,

                # ✅ CRITICAL: Wrap everything in "analysis" object - frontend expects this!
                "analysis": {
                    # Raw model analyses
                    "gpt_analysis": bias_results["gpt_analysis"],
                    "claude_analysis": bias_results["claude_analysis"],

                    # Combined report (full object for reference)
                    "combined_report": combined,

                    # Publication context (enhanced with MBFC data or prefetched)
                    "publication_profile": publication_profile_data,
                    "mbfc_context": mbfc_context,

                    # ✅ CRITICAL: Extract key fields to top level for easy access
                    "consensus_bias_score": combined.get("consensus_bias_score", 0),
                    "consensus_direction": combined.get("consensus_direction", "Unknown"),
                    "confidence": combined.get("confidence", 0),
                    "areas_of_agreement": combined.get("areas_of_agreement", []),
                    "areas_of_disagreement": combined.get("areas_of_disagreement", []),
                    "gpt_unique_findings": combined.get("gpt_unique_findings", []),
                    "claude_unique_findings": combined.get("claude_unique_findings", []),
                    "publication_bias_context": combined.get("publication_bias_context") or mbfc_context,
                    "final_assessment": combined.get("final_assessment", ""),
                    "recommendations": combined.get("recommendations", [])
                },

                # File locations
                "local_files": {
                    "combined_report": combined_report_path,
                    "gpt_report": gpt_report_path,
                    "claude_report": claude_report_path
                },

                # R2 upload status
                "r2_upload": r2_upload_status
            }

            fact_logger.log_component_complete(
                "BiasCheckOrchestrator",
                duration,
                session_id=session_id,
                consensus_score=combined.get("consensus_bias_score", 0),
                r2_uploads=len(r2_links) if r2_links else 0,
                mbfc_found=publication_profile_data is not None,
                used_prefetched=using_prefetched
            )

            return output

        except Exception as e:
            fact_logger.log_component_error("BiasCheckOrchestrator", e)
            raise

    async def process_with_progress(
        self,
        text: str,
        publication_url: Optional[str] = None,
        publication_name: Optional[str] = None,
        source_credibility: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        standalone: bool = True  # ADD THIS
    ) -> dict:
        """
        Process with real-time progress updates (for web interface)

        Args:
            text: Text to analyze
            publication_url: Optional publication URL for MBFC lookup
            publication_name: Optional publication name (backward compat)
            source_credibility: Optional pre-fetched credibility data (NEW)
                               If provided, skips MBFC lookup entirely
            job_id: Optional job ID for progress tracking

        Returns:
            Complete bias analysis results with R2 upload status
        """
        if job_id:
            from utils.job_manager import job_manager

            job_manager.add_progress(job_id, "📊 Starting bias analysis...")
            self._check_cancellation(job_id)

            # Show appropriate progress based on data source
            if source_credibility:
                tier = source_credibility.get('tier', '?')
                bias = source_credibility.get('bias_rating', 'Unknown')
                job_manager.add_progress(
                    job_id, 
                    f"📰 Using pre-fetched credibility: Tier {tier} | {bias}"
                )
            elif publication_url and self.mbfc_enabled:
                job_manager.add_progress(
                    job_id, 
                    f"📰 Looking up publication on MBFC: {publication_url}"
                )

            self._check_cancellation(job_id)

            job_manager.add_progress(job_id, "🤖 Analyzing with GPT-4o...")
            self._check_cancellation(job_id)
            job_manager.add_progress(job_id, "🤖 Analyzing with Claude Sonnet...")
            self._check_cancellation(job_id)

        # Call main process with all parameters including source_credibility
        result = await self.process(
            text=text,
            publication_url=publication_url,
            publication_name=publication_name,
            source_credibility=source_credibility,  # Pass through
            save_to_r2=True  
        )

        if job_id:
            from utils.job_manager import job_manager

            # Report credibility source
            if result.get("used_prefetched_credibility"):
                job_manager.add_progress(job_id, "✅ Used pre-fetched credibility data")
            elif result.get("analysis", {}).get("publication_profile"):
                profile = result["analysis"]["publication_profile"]
                source = profile.get("source", "local")
                name = profile.get("name", "Unknown")
                leaning = profile.get("political_leaning", "unknown")
                if source == "mbfc":
                    job_manager.add_progress(
                        job_id,
                        f"✅ MBFC data found: {name} ({leaning} bias)"
                    )
                else:
                    job_manager.add_progress(
                        job_id,
                        f"📚 Using local database: {name} ({leaning} bias)"
                    )
            elif publication_url:
                job_manager.add_progress(job_id, "📭 No publication bias data found")

            # R2 upload status
            if result.get("r2_upload", {}).get("success"):
                job_manager.add_progress(job_id, "☁️ Bias reports uploaded to R2")
            else:
                error_msg = result.get("r2_upload", {}).get("error", "Unknown error")
                job_manager.add_progress(job_id, f"⚠️ R2 upload failed: {error_msg}")

            job_manager.add_progress(job_id, "✅ Bias analysis complete!")
            if standalone:
                job_manager.complete_job(job_id, result)

        return result

    async def close(self):
        """Clean up resources (scraper, searcher connections)"""
        if self.scraper:
            try:
                await self.scraper.close()
                fact_logger.logger.debug("✅ Scraper closed")
            except Exception as e:
                fact_logger.logger.debug(f"Scraper close error (non-critical): {e}")

        if self.brave_searcher:
            try:
                await self.brave_searcher.close()
                fact_logger.logger.debug("✅ Brave searcher closed")
            except Exception as e:
                fact_logger.logger.debug(f"Brave searcher close error (non-critical): {e}")
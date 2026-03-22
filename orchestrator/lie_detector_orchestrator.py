# orchestrator/lie_detector_orchestrator.py
"""
Lie Detector Orchestrator - WITH SOURCE CREDIBILITY SUPPORT
Manages the lie detection workflow with file management and R2 storage

UPDATED: Now accepts source_credibility to calibrate analysis based on
source reliability (e.g., be more skeptical of Tier 5 sources).

Workflow:
1. Receive text + optional URL + optional publication date + optional credibility
2. Build credibility context for calibration
3. Run linguistic deception marker analysis with Claude
4. Save reports locally
5. Upload to Cloudflare R2
6. Return analysis results
"""

from langsmith import traceable
from typing import Optional, Dict, Any
import time
import json

from agents.lie_detector import LieDetector
from utils.file_manager import FileManager
from utils.r2_uploader import R2Uploader
from utils.logger import fact_logger

# Import credibility context builder
try:
    from utils.credibility_context import build_lie_detection_context
except ImportError:
    # Fallback if module not yet added
    def build_lie_detection_context(  # type: ignore[misc]
        source_credibility=None,
        article_source=None,
        article_date=None
    ) -> str:
        parts = []
        if article_source:
            parts.append(f"ARTICLE SOURCE: {article_source}")
        if article_date:
            parts.append(f"PUBLICATION DATE: {article_date}")
        if source_credibility:
            tier = source_credibility.get('tier')
            if tier:
                parts.append(f"SOURCE CREDIBILITY TIER: {tier}/5")
        return "\n".join(parts) if parts else ""

class LieDetectorOrchestrator:
    """
    Orchestrates lie detection analysis with storage and credibility calibration

    Pipeline:
    1. Receive text + optional URL + optional date + optional credibility
    2. If credibility provided, use it to calibrate analysis sensitivity
    3. Run linguistic deception marker analysis with Claude
    4. Save reports locally
    5. Upload to Cloudflare R2
    6. Return analysis results
    """

    def __init__(self, config):
        self.config = config
        self.lie_detector = LieDetector(config)
        self.file_manager = FileManager()

        # Initialize R2 uploader
        try:
            self.r2_uploader = R2Uploader()
            self.r2_enabled = True
            fact_logger.logger.info("✅ Cloudflare R2 integration enabled")
        except Exception as e:
            fact_logger.logger.warning(f"⚠️ Cloudflare R2 not configured: {e}")
            self.r2_enabled = False

        fact_logger.log_component_start("LieDetectorOrchestrator")

    def _check_cancellation(self, job_id: str):
        """Check if job has been cancelled and raise exception if so"""
        from utils.job_manager import job_manager
        if job_manager.is_cancelled(job_id):
            raise Exception("Job cancelled by user")

    def _build_enhanced_context(
        self,
        article_source: Optional[str],
        publication_date: Optional[str],
        source_credibility: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build enhanced context for the lie detector including credibility calibration.

        Args:
            article_source: Publication name
            publication_date: Publication date
            source_credibility: Pre-fetched credibility data

        Returns:
            Context string to append to the prompt
        """
        return build_lie_detection_context(
            source_credibility=source_credibility,
            article_source=article_source,
            article_date=publication_date
        )

    @traceable(
        name="lie_detection_pipeline",
        run_type="chain",
        tags=["orchestrator", "lie-detection", "deception-analysis", "r2-storage"]
    )
    async def process(
        self, 
        text: str,
        url: Optional[str] = None,
        publication_date: Optional[str] = None,
        article_source: Optional[str] = None,
        source_credibility: Optional[Dict[str, Any]] = None,  # NEW PARAMETER
        save_to_r2: bool = True
    ) -> dict:
        """
        Complete lie detection pipeline with R2 storage and credibility calibration

        Args:
            text: Text to analyze for deception markers
            url: Optional URL of the article
            publication_date: Optional publication date
            article_source: Optional publication name
            source_credibility: Optional pre-fetched credibility data (NEW)
                               Used to calibrate analysis sensitivity
            save_to_r2: Whether to save reports to Cloudflare R2

        Returns:
            Dictionary with complete lie detection analysis results
        """
        session_id = self.file_manager.create_session()
        start_time = time.time()

        # Determine if we're using pre-fetched credibility
        using_credibility = source_credibility is not None
        credibility_tier = int(source_credibility.get('tier', 0)) if source_credibility else None

        fact_logger.logger.info(
            f"🚀 STARTING LIE DETECTION SESSION: {session_id}",
            extra={
                "session_id": session_id,
                "text_length": len(text),
                "has_url": bool(url),
                "has_publication_date": bool(publication_date),
                "has_article_source": bool(article_source),
                "using_credibility": using_credibility,
                "credibility_tier": credibility_tier,
                "r2_enabled": self.r2_enabled and save_to_r2
            }
        )

        try:
            # Build enhanced context with credibility calibration
            credibility_context = self._build_enhanced_context(
                article_source=article_source,
                publication_date=publication_date,
                source_credibility=source_credibility
            )

            if credibility_context:
                fact_logger.logger.info(
                    "📊 Using credibility context for calibration",
                    extra={"tier": credibility_tier}
                )

            # Step 1: Run lie detection analysis
            fact_logger.logger.info("🔍 Step 1: Analyzing deception markers with Claude")

            # Pass credibility context to the lie detector
            # The lie detector will incorporate this into its analysis
            analysis_result = await self.lie_detector.analyze(
                text=text,
                url=url,
                publication_date=publication_date,
                credibility_context=credibility_context  # NEW: Pass context
            )

            # Step 2: Prepare report data
            fact_logger.logger.info("📝 Step 2: Preparing analysis report")

            report_data = {
                "session_id": session_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": {
                    "text_length": len(text),
                    "url": url,
                    "publication_date": publication_date,
                    "article_source": article_source
                },
                "source_credibility": source_credibility,  # Include in report
                "analysis": analysis_result.model_dump(),
                "metadata": {
                    "model": "claude-sonnet-4-20250514",
                    "processing_time_seconds": time.time() - start_time,
                    "used_credibility_calibration": using_credibility
                }
            }

            # Step 3: Save reports locally
            fact_logger.logger.info("💾 Step 3: Saving reports locally")

            analysis_report_path = self.file_manager.save_session_file(
                session_id,
                f"lie_detection_{session_id}.json",
                json.dumps(report_data, indent=2)
            )
            fact_logger.logger.info(f"✅ Saved analysis report: {analysis_report_path}")

            # Step 4: Upload to R2 if enabled
            r2_upload_status = {"success": False, "error": "R2 disabled"}
            r2_link = None

            if self.r2_enabled and save_to_r2:
                fact_logger.logger.info("☁️ Step 4: Uploading to Cloudflare R2")

                try:
                    r2_link = self.r2_uploader.upload_file(
                        file_path=analysis_report_path,
                        r2_filename=f"lie-detection/{session_id}/analysis.json"
                    )

                    if r2_link:
                        r2_upload_status = {
                            "success": True,
                            "url": r2_link
                        }
                        fact_logger.logger.info(f"✅ Uploaded to R2: {r2_link}")
                    else:
                        r2_upload_status = {
                            "success": False,
                            "error": "Upload returned no URL"
                        }

                except Exception as e:
                    fact_logger.logger.error(f"❌ R2 upload failed: {e}")
                    r2_upload_status = {
                        "success": False,
                        "error": str(e)
                    }

            # Build final result
            processing_time = time.time() - start_time

            fact_logger.log_component_complete(
                "LieDetectorOrchestrator",
                processing_time,
                session_id=session_id,
                risk_level=analysis_result.risk_level,
                credibility_score=analysis_result.credibility_score,
                used_credibility_calibration=using_credibility
            )

            return {
                "success": True,
                "session_id": session_id,
                "processing_time": processing_time,
                "used_credibility_calibration": using_credibility,
                "source_credibility_tier": credibility_tier,
                "analysis": analysis_result.model_dump(), 
                "local_report_path": analysis_report_path,
                "r2_upload": r2_upload_status,
                "r2_url": r2_link
            }

        except Exception as e:
            fact_logger.logger.error(f"❌ Lie detection pipeline failed: {e}", exc_info=True)

            return {
                "success": False,
                "session_id": session_id,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def process_with_progress(
        self,
        text: str,
        job_id: str,
        url: Optional[str] = None,
        publication_date: Optional[str] = None,
        article_source: Optional[str] = None,
        source_credibility: Optional[Dict[str, Any]] = None,
        standalone: bool = True
    ) -> dict:
        """
        Process with real-time progress updates (for web interface)

        Args:
            text: Text to analyze
            job_id: Job ID for progress tracking
            url: Optional article URL
            publication_date: Optional publication date
            article_source: Optional publication name
            source_credibility: Optional pre-fetched credibility data
            standalone: If True, calls complete_job on finish

        Returns:
            Complete lie detection analysis results
        """
        from utils.job_manager import job_manager

        try:
            job_manager.add_progress(job_id, "Starting deception marker analysis...")
            self._check_cancellation(job_id)

            # Show credibility calibration status
            if source_credibility:
                tier = source_credibility.get('tier', '?')
                tier_int = int(tier) if str(tier).isdigit() else 0
                bias = source_credibility.get('bias_rating', 'Unknown')
                job_manager.add_progress(
                    job_id, 
                    f"Calibrating for source: Tier {tier} | {bias}"
                )

                # Warn about low-credibility sources
                if tier_int >= 4:
                    job_manager.add_progress(
                        job_id,
                        "Low credibility source - applying heightened scrutiny"
                    )
            elif article_source:
                job_manager.add_progress(job_id, f"Analyzing: {article_source}")

            self._check_cancellation(job_id)

            job_manager.add_progress(job_id, "Analyzing with Claude Sonnet 4...")
            self._check_cancellation(job_id)

            # Run the main process with all parameters
            result = await self.process(
                text=text,
                url=url,
                publication_date=publication_date,
                article_source=article_source,
                source_credibility=source_credibility,
                save_to_r2=True
            )

            # Add progress about calibration
            if result.get("used_credibility_calibration"):
                job_manager.add_progress(job_id, "Analysis calibrated with source credibility")

            # Add progress about R2 upload
            if result.get("r2_upload", {}).get("success"):
                job_manager.add_progress(job_id, "Report uploaded to R2")
            else:
                error_msg = result.get("r2_upload", {}).get("error", "Unknown error")
                job_manager.add_progress(job_id, f"R2 upload failed: {error_msg}")

            job_manager.add_progress(job_id, "Lie detection analysis complete!")

            # Complete the job
            if standalone:
                job_manager.complete_job(job_id, result)

            return result

        except Exception as e:
            if "cancelled" in str(e).lower():
                job_manager.add_progress(job_id, "Analysis cancelled")
                if standalone:
                    job_manager.fail_job(job_id, "Cancelled by user")
            else:
                fact_logger.log_component_error(f"Job {job_id}", e)
                job_manager.fail_job(job_id, str(e))
            raise

    async def close(self):
        """Clean up resources"""
        pass  # No persistent resources in this orchestrator
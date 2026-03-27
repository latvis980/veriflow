# utils/tts_audit_builder.py
"""
TTS Audit Builder

Helper functions to build and save TTS audit trails from orchestrator data.
Follows the same pattern as search_audit_builder.py.

Used by: WebSearchOrchestrator, KeyClaimsOrchestrator
"""

from typing import List, Dict, Any, Optional
from utils.tts_audit import (
    TTSSessionAudit,
    TTSClaimAudit,
    TTSRoutingAudit,
    TTSEvidenceSnippet,
    TTSStorySource,
)
from utils.logger import fact_logger


def build_tts_session_audit(
    session_id: str,
    pipeline_type: str = "web_search",
    content_country: str = "international",
    content_language: str = "english",
    tts_enabled: bool = True,
) -> TTSSessionAudit:
    """
    Create a new TTSSessionAudit for a verification session.

    Call this at the start of the TTS stage, then populate it
    as routing and evidence results come in.
    """
    return TTSSessionAudit(
        session_id=session_id,
        pipeline_type=pipeline_type,
        content_country=content_country or "international",
        content_language=content_language or "english",
        tts_enabled=tts_enabled,
    )


def build_routing_audit(decision) -> TTSRoutingAudit:
    """
    Build a TTSRoutingAudit from a TTSRoutingDecision object.

    Args:
        decision: TTSRoutingDecision from tts_router.py
    """
    return TTSRoutingAudit(
        claim_id=getattr(decision, "claim_id", ""),
        route=getattr(decision, "route", "skip"),
        reason=getattr(decision, "reason", ""),
        tts_query=getattr(decision, "tts_query", None),
        tts_edition=getattr(decision, "tts_edition", None),
        confidence=getattr(decision, "confidence", 0.0),
    )


def build_claim_audit_from_evidence(
    claim_id: str,
    claim_statement: str,
    routing_decision,
    evidence: Optional[Dict[str, Any]],
    llm_match_score: float = 0.0,
    llm_report: str = "",
    adjusted_match_score: float = 0.0,
    adjusted_report: str = "",
    resolved: bool = False,
    fell_through_reason: str = "",
) -> TTSClaimAudit:
    """
    Build a complete TTSClaimAudit from orchestrator data.

    This is the main function called after TTS evidence is processed.

    Args:
        claim_id: Fact/claim identifier
        claim_statement: The claim text
        routing_decision: TTSRoutingDecision object
        evidence: Dict returned by TTSService.find_evidence_for_claim()
                  (contains cluster_id, evidence_texts, story_sources, etc.)
        llm_match_score: Score from LLM fact checker (before boost)
        llm_report: LLM's assessment text (before boost)
        adjusted_match_score: Score after apply_tts_cluster_boost
        adjusted_report: Report after boost
        resolved: Whether TTS resolved this claim
        fell_through_reason: Why it fell through (if not resolved)
    """
    audit = TTSClaimAudit(
        claim_id=claim_id,
        claim_statement=claim_statement,
    )

    # Routing
    if routing_decision is not None:
        audit.routing = build_routing_audit(routing_decision)

    # TTS search results
    if evidence and evidence.get("matched"):
        audit.cluster_matched = True
        audit.cluster_id = evidence.get("cluster_id")
        audit.cluster_title = evidence.get("cluster_title", "")
        audit.cluster_size = evidence.get("cluster_size", 0) or evidence.get("source_count", 0)
        audit.search_score = evidence.get("search_score", 0.0)
        audit.tts_edition_used = evidence.get("edition", "en")
        audit.total_search_results = evidence.get("total_search_results", 0)

        # Evidence snippets -- the key data for human audit
        for et in evidence.get("evidence_texts", []):
            audit.evidence_snippets.append(TTSEvidenceSnippet(
                source_name=et.get("source", "Unknown"),
                text=et.get("text", ""),
                url=et.get("url", ""),
                title=et.get("title", ""),
            ))

        # Story sources
        for ss in evidence.get("story_sources", []):
            audit.story_sources.append(TTSStorySource(
                title=ss.get("title", ""),
                source_name=ss.get("source", ""),
                url=ss.get("url", ""),
            ))
    else:
        audit.cluster_matched = False

    # LLM verification results
    audit.llm_match_score = llm_match_score
    audit.llm_report = llm_report
    audit.adjusted_match_score = adjusted_match_score
    audit.adjusted_report = adjusted_report
    audit.was_boosted = adjusted_match_score > llm_match_score

    # Outcome
    audit.resolved_by_tts = resolved
    audit.fell_through_reason = fell_through_reason

    return audit


def build_skipped_claim_audit(
    claim_id: str,
    claim_statement: str,
    routing_decision,
) -> TTSClaimAudit:
    """
    Build a TTSClaimAudit for a claim that was skipped (routed to web search).
    """
    audit = TTSClaimAudit(
        claim_id=claim_id,
        claim_statement=claim_statement,
        cluster_matched=False,
        resolved_by_tts=False,
        fell_through_reason="Routed to web search by TTS router",
    )

    if routing_decision is not None:
        audit.routing = build_routing_audit(routing_decision)

    return audit


def build_failed_claim_audit(
    claim_id: str,
    claim_statement: str,
    routing_decision,
    evidence: Optional[Dict[str, Any]],
    reason: str = "",
) -> TTSClaimAudit:
    """
    Build a TTSClaimAudit for a claim that was routed to TTS
    but fell through (no match, cluster too small, error, etc.).

    Still records the evidence snippets if any were returned,
    so auditors can see what TTS found even if it was not enough.
    """
    audit = build_claim_audit_from_evidence(
        claim_id=claim_id,
        claim_statement=claim_statement,
        routing_decision=routing_decision,
        evidence=evidence,
        resolved=False,
        fell_through_reason=reason,
    )
    return audit


# =========================================================================
# Save and upload
# =========================================================================

def save_tts_audit(
    tts_audit: TTSSessionAudit,
    file_manager,
    session_id: str,
    filename: str = "tts_audit.json",
) -> str:
    """
    Save TTS audit to session directory.

    Args:
        tts_audit: The TTSSessionAudit to save
        file_manager: FileManager instance
        session_id: Session identifier
        filename: Name for the audit file

    Returns:
        Path to the saved file
    """
    try:
        audit_json = tts_audit.to_json(indent=2)
        filepath = file_manager.save_session_file(
            session_id=session_id,
            filename=filename,
            content=audit_json,
            auto_serialize=False,
        )

        fact_logger.logger.info(
            f"Saved TTS audit: {filename}",
            extra={
                "session_id": session_id,
                "total_claims": tts_audit.total_claims,
                "resolved_by_tts": tts_audit.resolved_by_tts,
            },
        )

        return filepath

    except Exception as e:
        fact_logger.logger.error(
            f"Failed to save TTS audit: {e}",
            extra={"session_id": session_id, "error": str(e)},
        )
        raise


async def upload_tts_audit_to_r2(
    tts_audit: TTSSessionAudit,
    session_id: str,
    r2_uploader,
    pipeline_type: str = "web-search",
) -> Optional[str]:
    """
    Upload TTS audit to Cloudflare R2.

    Args:
        tts_audit: The TTSSessionAudit to upload
        session_id: Session identifier
        r2_uploader: R2Uploader instance
        pipeline_type: Type of pipeline for R2 folder structure

    Returns:
        R2 URL if successful, None otherwise
    """
    try:
        audit_json = tts_audit.to_json(indent=2)

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(audit_json)
            temp_path = f.name

        try:
            r2_filename = f"{pipeline_type}-audits/{session_id}/tts_audit.json"

            url = r2_uploader.upload_file(
                file_path=temp_path,
                r2_filename=r2_filename,
                metadata={
                    "session-id": session_id,
                    "report-type": "tts-audit",
                    "pipeline-type": pipeline_type,
                    "total-claims": str(tts_audit.total_claims),
                    "resolved-by-tts": str(tts_audit.resolved_by_tts),
                },
            )

            if url:
                fact_logger.logger.info(
                    f"Uploaded TTS audit to R2: {r2_filename}"
                )

            return url

        finally:
            os.unlink(temp_path)

    except Exception as e:
        fact_logger.logger.error(
            f"Failed to upload TTS audit to R2: {e}",
            extra={"session_id": session_id, "error": str(e)},
        )
        return None

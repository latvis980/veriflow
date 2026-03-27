# utils/tts_audit.py
"""
TTS (The True Story) Audit Data Models

Stores a comprehensive audit trail of TTS Layer 0 verification:
- Routing decisions (which claims went to TTS vs web search)
- TTS search results (clusters, scores)
- Evidence snippets collected from TTS articles
- Story sources from the TTS stories API
- LLM fact-check results using TTS evidence
- Score adjustments from cluster-size boost

PURPOSE: Human auditors can review every TTS snippet against
the original claim to validate VeriFlow's verification report.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class TTSRoutingAudit:
    """Audit of the routing decision for a single claim"""
    claim_id: str
    route: str  # "tts" or "skip"
    reason: str
    tts_query: Optional[str] = None
    tts_edition: Optional[str] = None  # "en", "ru", etc.
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "route": self.route,
            "reason": self.reason,
            "tts_query": self.tts_query,
            "tts_edition": self.tts_edition,
            "confidence": self.confidence,
        }


@dataclass
class TTSEvidenceSnippet:
    """
    A single snippet of text collected from TTS.

    This is the key data for human audit: the actual text
    that VeriFlow used to verify a claim.
    """
    source_name: str  # e.g. "Reuters", "BBC News"
    text: str  # The article snippet / excerpt
    url: str = ""
    title: str = ""  # Article title if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "text": self.text,
            "url": self.url,
            "title": self.title,
        }


@dataclass
class TTSStorySource:
    """A source from the TTS stories API (broader cluster view)"""
    title: str
    source_name: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source_name": self.source_name,
            "url": self.url,
        }


@dataclass
class TTSClaimAudit:
    """
    Complete TTS audit trail for a single claim.

    Contains everything a human auditor needs to check:
    1. The claim text
    2. How TTS was queried
    3. What cluster matched
    4. Every snippet returned
    5. The LLM verification result
    6. Any score adjustment applied
    """
    claim_id: str
    claim_statement: str

    # Routing decision
    routing: Optional[TTSRoutingAudit] = None

    # TTS search results
    cluster_matched: bool = False
    cluster_id: Optional[str] = None
    cluster_title: str = ""
    cluster_size: int = 0
    search_score: float = 0.0
    tts_edition_used: str = "en"
    total_search_results: int = 0

    # Evidence snippets -- the core data for human review
    evidence_snippets: List[TTSEvidenceSnippet] = field(default_factory=list)

    # Story sources (from stories API, broader cluster view)
    story_sources: List[TTSStorySource] = field(default_factory=list)

    # LLM verification using TTS evidence
    llm_match_score: float = 0.0  # Score from LLM fact checker
    llm_report: str = ""  # LLM's assessment text

    # Cluster boost adjustment
    adjusted_match_score: float = 0.0  # After apply_tts_cluster_boost
    adjusted_report: str = ""  # After boost
    was_boosted: bool = False

    # Outcome
    resolved_by_tts: bool = False  # Did TTS resolve this claim?
    fell_through_reason: str = ""  # Why it fell through (if it did)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_statement": self.claim_statement,
            "routing": self.routing.to_dict() if self.routing else None,
            "tts_search": {
                "cluster_matched": self.cluster_matched,
                "cluster_id": self.cluster_id,
                "cluster_title": self.cluster_title,
                "cluster_size": self.cluster_size,
                "search_score": self.search_score,
                "edition_used": self.tts_edition_used,
                "total_search_results": self.total_search_results,
            },
            "evidence_snippets": [s.to_dict() for s in self.evidence_snippets],
            "story_sources": [s.to_dict() for s in self.story_sources],
            "verification": {
                "llm_match_score": self.llm_match_score,
                "llm_report": self.llm_report,
                "adjusted_match_score": self.adjusted_match_score,
                "adjusted_report": self.adjusted_report,
                "was_boosted": self.was_boosted,
            },
            "outcome": {
                "resolved_by_tts": self.resolved_by_tts,
                "fell_through_reason": self.fell_through_reason,
            },
        }


@dataclass
class TTSSessionAudit:
    """
    Complete TTS audit for an entire fact-checking session.

    Uploaded to R2 as a JSON file for human review.
    """
    session_id: str
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    pipeline_type: str = "web_search"  # web_search, key_claims, llm_output

    # Content context
    content_country: str = "international"
    content_language: str = "english"

    # TTS availability
    tts_enabled: bool = True

    # Per-claim audits
    claim_audits: List[TTSClaimAudit] = field(default_factory=list)

    # Session-level summary
    total_claims: int = 0
    routed_to_tts: int = 0
    resolved_by_tts: int = 0
    fell_through: int = 0
    skipped: int = 0
    tts_duration_seconds: float = 0.0

    def add_claim_audit(self, claim_audit: TTSClaimAudit):
        """Add a claim audit and update session totals"""
        self.claim_audits.append(claim_audit)
        self.total_claims = len(self.claim_audits)

        if claim_audit.routing:
            if claim_audit.routing.route == "tts":
                self.routed_to_tts += 1
                if claim_audit.resolved_by_tts:
                    self.resolved_by_tts += 1
                else:
                    self.fell_through += 1
            else:
                self.skipped += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "pipeline_type": self.pipeline_type,
            "content_location": {
                "country": self.content_country,
                "language": self.content_language,
            },
            "tts_enabled": self.tts_enabled,
            "summary": {
                "total_claims": self.total_claims,
                "routed_to_tts": self.routed_to_tts,
                "resolved_by_tts": self.resolved_by_tts,
                "fell_through": self.fell_through,
                "skipped": self.skipped,
                "tts_duration_seconds": self.tts_duration_seconds,
            },
            "claim_audits": [c.to_dict() for c in self.claim_audits],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

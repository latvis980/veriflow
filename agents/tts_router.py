# agents/tts_router.py
"""
TTS Router Agent
Determines which claims should be checked against The True Story (TTS)
news aggregation platform before falling back to standard web search.

Uses Gemini 2.0 Flash for fast, cheap classification (Phase 6 optimization).
Runs AFTER fact extraction, BEFORE verification.

Integration point:
  fact_extractor -> tts_router -> [tts_service | web_search] -> fact_checker

The router examines each extracted claim and decides:
- "tts": claim is about a recent news event likely covered by TTS
- "skip": claim is not news-related, go straight to web search
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from prompts.tts_router_prompts import get_tts_router_prompts
from utils.logger import fact_logger


@dataclass
class TTSRoutingDecision:
    """Routing decision for a single claim"""
    claim_id: str
    route: str  # "tts" or "skip"
    reason: str
    tts_query: Optional[str]
    tts_edition: Optional[str]  # "ru" or "en"
    confidence: float


class TTSRouter:
    """
    Routes claims to TTS or web search based on content analysis.

    Uses Gemini 2.0 Flash for speed and cost efficiency.
    Processes all claims in a single LLM call (batch routing).
    """

    def __init__(self, config=None):
        self.config = config

        # Use Gemini Flash for routing (Phase 6 optimization)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.0,
            max_output_tokens=2000,
        )

        prompts = get_tts_router_prompts()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["system"]),
            ("human", prompts["user"]),
        ])

        fact_logger.logger.info("TTSRouter initialized (Gemini 2.0 Flash)")

    @traceable(
        name="tts_route_claims",
        run_type="chain",
        tags=["tts", "routing", "gemini-flash"]
    )
    async def route(
        self,
        claims: List[Dict[str, Any]],
        content_language: str = "english",
        content_realm: str = "unknown",
    ) -> List[TTSRoutingDecision]:
        """
        Route a batch of claims to TTS or web search.

        Args:
            claims: List of extracted claims/facts with 'id' and 'statement'
            content_language: Detected language of the content
            content_realm: Detected realm (political, economic, etc.)

        Returns:
            List of TTSRoutingDecision for each claim
        """
        if not claims:
            return []

        # Quick pre-filter: if realm is clearly outside TTS scope, skip all
        skip_realms = {"entertainment", "sports", "technology", "other"}
        if content_realm.lower() in skip_realms:
            fact_logger.logger.info(
                f"TTS Router: realm '{content_realm}' outside TTS scope, "
                f"skipping all {len(claims)} claims"
            )
            return [
                TTSRoutingDecision(
                    claim_id=c.get("id", f"fact{i+1}"),
                    route="skip",
                    reason=f"content realm '{content_realm}' not covered by TTS",
                    tts_query=None,
                    tts_edition=None,
                    confidence=0.95,
                )
                for i, c in enumerate(claims)
            ]

        # Format claims for the prompt
        claims_text = self._format_claims(claims)

        try:
            chain = self.prompt | self.llm
            response = await chain.ainvoke({
                "claims_text": claims_text,
                "language": content_language,
                "realm": content_realm,
            })

            decisions = self._parse_response(response.content, claims)

            # Log routing summary
            tts_count = sum(1 for d in decisions if d.route == "tts")
            skip_count = sum(1 for d in decisions if d.route == "skip")
            fact_logger.logger.info(
                f"TTS Router: {tts_count} claims -> TTS, "
                f"{skip_count} claims -> web search"
            )

            return decisions

        except Exception as e:
            fact_logger.logger.error(f"TTS Router error: {e}")
            # On error, default to skipping TTS (safe fallback)
            return [
                TTSRoutingDecision(
                    claim_id=c.get("id", f"fact{i+1}"),
                    route="skip",
                    reason="routing error, falling back to web search",
                    tts_query=None,
                    tts_edition=None,
                    confidence=0.0,
                )
                for i, c in enumerate(claims)
            ]

    def _format_claims(self, claims: List[Dict[str, Any]]) -> str:
        """Format claims list for the prompt"""
        lines = []
        for i, claim in enumerate(claims):
            claim_id = claim.get("id", f"fact{i+1}")
            statement = claim.get("statement", "")
            lines.append(f"[{claim_id}] {statement}")
        return "\n".join(lines)

    def _parse_response(
        self,
        response_text: str,
        original_claims: List[Dict[str, Any]],
    ) -> List[TTSRoutingDecision]:
        """Parse LLM response into routing decisions"""
        try:
            # Clean potential markdown wrapping
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            data = json.loads(text)
            raw_decisions = data.get("routing_decisions", [])

            # Build a lookup for original claim IDs
            original_ids = {
                c.get("id", f"fact{i+1}") for i, c in enumerate(original_claims)
            }

            decisions = []
            seen_ids = set()

            for rd in raw_decisions:
                claim_id = rd.get("claim_id", "")
                if claim_id not in original_ids or claim_id in seen_ids:
                    continue
                seen_ids.add(claim_id)

                route = rd.get("route", "skip").lower()
                if route not in ("tts", "skip"):
                    route = "skip"

                decisions.append(TTSRoutingDecision(
                    claim_id=claim_id,
                    route=route,
                    reason=rd.get("reason", ""),
                    tts_query=rd.get("tts_query") if route == "tts" else None,
                    tts_edition=rd.get("tts_edition", "en") if route == "tts" else None,
                    confidence=float(rd.get("confidence", 0.5)),
                ))

            # Fill in any claims missing from LLM response (default to skip)
            for i, claim in enumerate(original_claims):
                cid = claim.get("id", f"fact{i+1}")
                if cid not in seen_ids:
                    decisions.append(TTSRoutingDecision(
                        claim_id=cid,
                        route="skip",
                        reason="not returned by router, defaulting to web search",
                        tts_query=None,
                        tts_edition=None,
                        confidence=0.0,
                    ))

            return decisions

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            fact_logger.logger.error(f"TTS Router parse error: {e}")
            # Return skip for all claims on parse failure
            return [
                TTSRoutingDecision(
                    claim_id=c.get("id", f"fact{i+1}"),
                    route="skip",
                    reason="parse error, defaulting to web search",
                    tts_query=None,
                    tts_edition=None,
                    confidence=0.0,
                )
                for i, c in enumerate(original_claims)
            ]

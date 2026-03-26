# agents/tts_router.py
"""
TTS Router Agent
Determines which claims should be checked against The True Story (TTS)
news aggregation platform before falling back to standard web search.

Uses OpenAI GPT-5.4 nano for fast, cheap classification/routing.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
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

    Uses OpenAI GPT-5.4 nano -- fastest, cheapest model,
    optimized for classification, extraction, and routing tasks.
    """

    def __init__(self, config=None):
        self.config = config
        self.llm = None

        fact_logger.logger.info("TTSRouter: initializing...")

        # GPT-5.4 nano: fastest, cheapest, built for classification/routing
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="gpt-5.4-nano",
                temperature=0.0,
                max_tokens=2000,
            ).bind(response_format={"type": "json_object"})
            fact_logger.logger.info("TTSRouter: using OpenAI gpt-5.4-nano")
        except Exception as e:
            fact_logger.logger.error(f"TTSRouter: init failed: {e}")
            raise RuntimeError("TTSRouter: no LLM available")

        prompts = get_tts_router_prompts()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompts["system"]),
            ("human", prompts["user"]),
        ])

        fact_logger.logger.info("TTSRouter: initialized successfully")

    @traceable(
        name="tts_route_claims",
        run_type="chain",
        tags=["tts", "routing"]
    )
    async def route(
        self,
        claims: List[Dict[str, Any]],
        content_language: str = "english",
        content_realm: str = "unknown",
    ) -> List[TTSRoutingDecision]:
        """
        Route a batch of claims to TTS or web search.
        """
        if not claims:
            fact_logger.logger.info("TTSRouter: no claims to route")
            return []

        fact_logger.logger.info(
            f"TTSRouter: routing {len(claims)} claims "
            f"(language={content_language}, realm={content_realm})"
        )

        # Quick pre-filter: if realm is clearly outside TTS scope, skip all
        skip_realms = {"entertainment", "sports", "technology"}
        if content_realm.lower() in skip_realms:
            fact_logger.logger.info(
                f"TTSRouter: realm '{content_realm}' outside TTS scope, "
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
        fact_logger.logger.debug(f"TTSRouter: claims text for LLM:\n{claims_text}")

        try:
            fact_logger.logger.info("TTSRouter: calling LLM for routing decisions...")

            chain = self.prompt | self.llm

            # Timeout: if LLM doesn't respond in 15s, fall back to web search
            try:
                response = await asyncio.wait_for(
                    chain.ainvoke({
                        "claims_text": claims_text,
                        "language": content_language,
                        "realm": content_realm,
                    }),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                fact_logger.logger.error(
                    "TTSRouter: LLM call timed out after 15s"
                )
                return [
                    TTSRoutingDecision(
                        claim_id=c.get("id", f"fact{i+1}"),
                        route="skip",
                        reason="router timeout, falling back to web search",
                        tts_query=None,
                        tts_edition=None,
                        confidence=0.0,
                    )
                    for i, c in enumerate(claims)
                ]

            fact_logger.logger.info("TTSRouter: LLM response received")
            fact_logger.logger.debug(
                f"TTSRouter: raw response: {response.content[:500]}"
            )

            decisions = self._parse_response(response.content, claims)

            # Log routing summary
            tts_count = sum(1 for d in decisions if d.route == "tts")
            skip_count = sum(1 for d in decisions if d.route == "skip")
            fact_logger.logger.info(
                f"TTSRouter: {tts_count} claims -> TTS, "
                f"{skip_count} claims -> web search"
            )
            for d in decisions:
                fact_logger.logger.info(
                    f"  [{d.claim_id}] -> {d.route} "
                    f"(query={d.tts_query}, edition={d.tts_edition}, "
                    f"conf={d.confidence:.2f}, reason={d.reason})"
                )

            return decisions

        except Exception as e:
            fact_logger.logger.error(
                f"TTSRouter: LLM call failed: {type(e).__name__}: {e}"
            )
            import traceback
            fact_logger.logger.error(f"TTSRouter traceback: {traceback.format_exc()}")

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

            fact_logger.logger.debug(
                f"TTSRouter: parsed {len(raw_decisions)} decisions from LLM"
            )

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

            # Fill in any claims missing from LLM response
            for i, claim in enumerate(original_claims):
                cid = claim.get("id", f"fact{i+1}")
                if cid not in seen_ids:
                    fact_logger.logger.warning(
                        f"TTSRouter: claim {cid} missing from LLM response, "
                        f"defaulting to skip"
                    )
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
            fact_logger.logger.error(
                f"TTSRouter: parse error: {type(e).__name__}: {e}"
            )
            fact_logger.logger.error(
                f"TTSRouter: raw text was: {response_text[:500]}"
            )
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
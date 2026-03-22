# agents/fact_checker.py
"""
Fact Checker Agent - WITH TIER FILTERING + KEY ROTATION
Compares claimed facts against source excerpts

TIER FILTERING: Uses Tiers 1-3, filters out Tiers 4-5
TIER PRECEDENCE: Tier 1 sources override lower tiers when there are contradictions
SIMPLIFIED OUTPUT: Single comprehensive report field
KEY ROTATION: Uses get_openai_llm() per call for safe parallel execution
"""

from langsmith import traceable
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import time

from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.source_metadata import SourceMetadata
from utils.openai_client import get_openai_llm


class FactCheckResult(BaseModel):
    """Result of fact checking - simplified with single report field"""
    fact_id: str
    statement: str
    match_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    report: str = Field(description="Comprehensive verification report")
    tier_breakdown: Optional[Dict[str, int]] = None


class FactChecker:
    """
    Checks facts against extracted excerpts

    TIER FILTERING:
    - Discards Tier 4-5 sources (< 0.65 credibility)
    - Evaluates against Tier 1 (0.90+), Tier 2 (0.80-0.89), Tier 3 (0.65-0.79)
    - Sorts excerpts by tier (Tier 1 first) before evaluation

    KEY ROTATION: Each check_fact call gets a fresh LLM with rotated API key
    """

    def __init__(self, config):
        self.config = config

        self.parser = JsonOutputParser(pydantic_object=FactCheckResult)

        # Load prompts
        from prompts.checker_prompts import get_checker_prompts
        self.prompts = get_checker_prompts()

        fact_logger.log_component_start("FactChecker", model="gpt-4o (key rotation)")

    @traceable(
        name="check_fact_accuracy",
        run_type="chain",
        tags=["fact-checking", "verification", "tier-filtering"]
    )
    async def check_fact(
        self, 
        fact, 
        excerpts: dict,
        source_metadata: Optional[Dict[str, SourceMetadata]] = None
    ) -> FactCheckResult:
        """
        Check a fact against source excerpts

        TIER FILTERING: Filters out Tier 4-5 sources before evaluation
        TIER ORDERING: Presents sources sorted by tier to the LLM

        Args:
            fact: Fact object with id and statement
            excerpts: Dict of URL -> list of excerpt dicts
            source_metadata: Optional dict of URL -> SourceMetadata for tier info
        """
        start_time = time.time()

        fact_logger.logger.info(
            f"Checking fact {fact.id}: {fact.statement[:50]}...",
            extra={"fact_id": fact.id}
        )

        # Filter and sort excerpts by tier
        filtered_excerpts = self._filter_by_tier(excerpts, source_metadata)

        if not filtered_excerpts:
            fact_logger.logger.warning(
                f"No Tier 1-3 sources for {fact.id}",
                extra={"fact_id": fact.id}
            )
            return FactCheckResult(
                fact_id=fact.id,
                statement=fact.statement,
                match_score=0.0,
                confidence=0.0,
                report="Unable to verify - no credible Tier 1 or Tier 2 sources found. Web search did not return authoritative sources for this claim.",
                tier_breakdown={"tier1": 0, "tier2": 0, "filtered": len(excerpts)}
            )

        # Evaluate against filtered excerpts
        result = await self._evaluate_fact(fact, filtered_excerpts, source_metadata)

        duration = time.time() - start_time
        fact_logger.logger.info(
            f"Fact {fact.id} checked: score={result.match_score:.2f}",
            extra={
                "fact_id": fact.id,
                "match_score": result.match_score,
                "duration_seconds": round(duration, 2)
            }
        )

        return result

    def _filter_by_tier(
        self, 
        excerpts: dict, 
        source_metadata: Optional[Dict[str, SourceMetadata]]
    ) -> list:
        """
        Filter excerpts using 5-tier system.
        Keeps Tiers 1-3, filters out Tiers 4-5.
        Returns a flat list sorted by tier (Tier 1 first).
        """
        tier1_excerpts = []
        tier2_excerpts = []
        tier3_excerpts = []
        filtered_count = 0

        for url, url_excerpts in excerpts.items():
            # Get credibility score - handle both object and dict formats
            if source_metadata and url in source_metadata:
                meta = source_metadata[url]
                # Support both SourceMetadata objects and plain dicts
                if hasattr(meta, 'credibility_score'):
                    score = meta.credibility_score
                elif isinstance(meta, dict):
                    score = meta.get('credibility_score', 0.70)
                else:
                    score = 0.70
            else:
                score = 0.70  # Default to Tier 3 if unknown

            # Classify by tier using 5-tier thresholds
            if score >= 0.90:  # Tier 1 - Primary Authority
                for ex in url_excerpts:
                    tier1_excerpts.append({
                        **ex,
                        'url': url,
                        'tier': 1,
                        'credibility_score': score
                    })
            elif score >= 0.80:  # Tier 2 - Highly Credible
                for ex in url_excerpts:
                    tier2_excerpts.append({
                        **ex,
                        'url': url,
                        'tier': 2,
                        'credibility_score': score
                    })
            elif score >= 0.65:  # Tier 3 - Credible
                for ex in url_excerpts:
                    tier3_excerpts.append({
                        **ex,
                        'url': url,
                        'tier': 3,
                        'credibility_score': score
                    })
            else:  # Tier 4-5 - filtered out
                filtered_count += len(url_excerpts)

        if filtered_count > 0:
            fact_logger.logger.debug(
                f"Filtered {filtered_count} Tier 4-5 excerpts",
                extra={"filtered_count": filtered_count}
            )

        # Return Tier 1 first, then Tier 2, then Tier 3
        return tier1_excerpts + tier2_excerpts + tier3_excerpts

    def _get_metadata_value(self, metadata, key: str, default: Any = ''):
        """Get value from metadata - handles both object and dict formats"""
        if metadata is None:
            return default
        if hasattr(metadata, key):
            return getattr(metadata, key)
        elif isinstance(metadata, dict):
            return metadata.get(key, default)
        return default
        
    def _format_excerpts(self, excerpts: list, source_metadata: Optional[Dict[str, SourceMetadata]] = None) -> str:
        """Format excerpts for the prompt, separated by tier"""
        if not excerpts:
            return "No credible source excerpts available."

        tier1_excerpts = [e for e in excerpts if e.get('tier') == 1]
        tier2_excerpts = [e for e in excerpts if e.get('tier') == 2]
        tier3_excerpts = [e for e in excerpts if e.get('tier') == 3]

        formatted = []

        # Format Tier 1 sources first
        if tier1_excerpts:
            formatted.append("=" * 60)
            formatted.append("TIER 1 SOURCES (PRIMARY AUTHORITY - TRUST THESE FIRST)")
            formatted.append("=" * 60)

            for ex in tier1_excerpts:
                url = ex['url']
                metadata = source_metadata.get(url) if source_metadata else None

                if metadata:
                    name = self._get_metadata_value(metadata, 'name', url)
                    source_type = self._get_metadata_value(metadata, 'source_type', 'unknown')
                    tier = self._get_metadata_value(metadata, 'credibility_tier', 'Tier 1')
                    score = self._get_metadata_value(metadata, 'credibility_score', 0.95)

                    formatted.append(
                        f"\n[Source: {name} ({source_type})]\n"
                        f"Credibility: {tier} (Score: {score:.2f})\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                        f"URL: {url}\n"
                    )
                else:
                    formatted.append(
                        f"\n[Source: {url}]\n"
                        f"Tier: 1\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                    )

        # Format Tier 2 sources next
        if tier2_excerpts:
            formatted.append("\n" + "=" * 60)
            formatted.append("TIER 2 SOURCES (HIGHLY CREDIBLE)")
            formatted.append("=" * 60)

            for ex in tier2_excerpts:
                url = ex['url']
                metadata = source_metadata.get(url) if source_metadata else None

                if metadata:
                    name = self._get_metadata_value(metadata, 'name', url)
                    source_type = self._get_metadata_value(metadata, 'source_type', 'unknown')
                    tier = self._get_metadata_value(metadata, 'credibility_tier', 'Tier 2')
                    score = self._get_metadata_value(metadata, 'credibility_score', 0.85)

                    formatted.append(
                        f"\n[Source: {name} ({source_type})]\n"
                        f"Credibility: {tier} (Score: {score:.2f})\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                        f"URL: {url}\n"
                    )
                else:
                    formatted.append(
                        f"\n[Source: {url}]\n"
                        f"Tier: 2\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                    )

        # Format Tier 3 sources
        if tier3_excerpts:
            formatted.append("\n" + "=" * 60)
            formatted.append("TIER 3 SOURCES (CREDIBLE - CORROBORATION)")
            formatted.append("=" * 60)

            for ex in tier3_excerpts:
                url = ex['url']
                metadata = source_metadata.get(url) if source_metadata else None

                if metadata:
                    name = self._get_metadata_value(metadata, 'name', url)
                    source_type = self._get_metadata_value(metadata, 'source_type', 'unknown')
                    tier = self._get_metadata_value(metadata, 'credibility_tier', 'Tier 3')
                    score = self._get_metadata_value(metadata, 'credibility_score', 0.70)

                    formatted.append(
                        f"\n[Source: {name} ({source_type})]\n"
                        f"Credibility: {tier} (Score: {score:.2f})\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                        f"URL: {url}\n"
                    )
                else:
                    formatted.append(
                        f"\n[Source: {url}]\n"
                        f"Tier: 3\n"
                        f"Relevance: {ex['relevance']}\n"
                        f"Quote: {ex['quote']}\n"
                    )

        return "\n".join(formatted)

    @traceable(name="evaluate_fact_match", run_type="llm")
    async def _evaluate_fact(self, fact, excerpts: list, source_metadata: Optional[Dict[str, SourceMetadata]] = None) -> FactCheckResult:
        """
        Evaluate fact accuracy against excerpts
        Emphasizes tier precedence in prompt

        KEY ROTATION: Fresh LLM per call via get_openai_llm()

        Args:
            fact: Fact object
            excerpts: list of excerpt dicts with url, quote, relevance, tier
            source_metadata: Source metadata for tier information
        """
        # Format excerpts with tier separation
        excerpts_text = self._format_excerpts(excerpts, source_metadata)

        # Enhanced system prompt with tier precedence
        tier_precedence_note = """

CRITICAL: TIER 1 SOURCES TAKE ABSOLUTE PRECEDENCE

When evaluating facts:
1. Prioritize Tier 1 sources (0.90-1.0 credibility) as the PRIMARY TRUTH
2. Use Tier 2 sources (0.80-0.89 credibility) as highly credible supporting evidence
3. Use Tier 3 sources (0.65-0.79 credibility) for corroboration only
4. If Tier 1 and lower tiers contradict, ALWAYS trust Tier 1
5. If only Tier 2-3 sources available, note this limitation in your report

Examples:
"While Tier 3 sources (Travel Blog) mention Chef Mario, Tier 1 sources (Official Restaurant Website, Michelin Guide) confirm Chef Julia is the current head chef. Tier 1 takes precedence."
"While Tier 2 source (NYT) mentions the bill was delayed, Tier 1 source (official government website) states it was passed. Tier 1 is the final authority."
"""

        system_prompt = self.prompts["system"] + tier_precedence_note + "\n\nIMPORTANT: You MUST return valid JSON only. No other text."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", self.prompts["user"] + "\n\n{format_instructions}\n\nReturn your response as valid JSON.")
        ])

        prompt_with_format = prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

        callbacks = langsmith_config.get_callbacks(f"fact_checker_{fact.id}")

        # KEY ROTATION: Fresh LLM per call -- rotated API key for parallel safety
        llm = get_openai_llm(model="gpt-4o", temperature=0, json_mode=True)
        chain = prompt_with_format | llm | self.parser

        fact_logger.logger.debug(
            "Invoking LLM for fact checking",
            extra={"fact_id": fact.id, "num_excerpts": len(excerpts)}
        )

        response = await chain.ainvoke(
            {
                "fact": fact.statement,
                "excerpts": excerpts_text
            },
            config={"callbacks": callbacks.handlers}
        )

        # Calculate tier breakdown
        tier1_count = len([e for e in excerpts if e.get('tier') == 1])
        tier2_count = len([e for e in excerpts if e.get('tier') == 2])

        # Parse and return
        return FactCheckResult(
            fact_id=fact.id,
            statement=fact.statement,
            match_score=response['match_score'],
            confidence=response['confidence'],
            report=response['report'],
            tier_breakdown={"tier1": tier1_count, "tier2": tier2_count}
        )

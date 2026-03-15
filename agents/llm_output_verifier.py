# agents/llm_output_verifier.py
"""
LLM Output Verifier Agent
Verifies if an LLM accurately interpreted its cited sources

USAGE: LLM Output Pipeline ONLY
- Checks interpretation accuracy, not source credibility
- Compares LLM's claim against actual source content
- NO tier filtering (sources already provided by LLM)

KEY ROTATION: Each verification call gets a fresh LLM instance
with a rotated API key, enabling safe parallel execution.
"""

from langsmith import traceable
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
from datetime import datetime

from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config
from utils.openai_client import get_openai_llm
from agents.llm_fact_extractor import LLMClaim


class LLMVerificationResult(BaseModel):
    claim_id: str
    claim_text: str
    verification_score: float = Field(ge=0.0, le=1.0)
    assessment: str
    interpretation_issues: List[str] = Field(default_factory=list)
    wording_comparison: Dict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    excerpts: List[Dict] = Field(default_factory=list)
    cited_source_urls: List[str] = Field(default_factory=list)
    source_issues: List[Dict] = Field(default_factory=list)  # [{url, reason, domain}]
    cannot_verify: bool = False  # True when ALL cited sources are blocked/inaccessible


class LLMOutputVerifier:
    """
    Verifies if an LLM accurately interpreted its cited sources

    Key Difference from FactChecker:
    - FactChecker: Checks if facts are TRUE (uses tier filtering)
    - LLMOutputVerifier: Checks if LLM INTERPRETED sources correctly (no tier filtering)

    KEY ROTATION: Uses get_openai_llm() per call so parallel
    verifications spread across multiple API keys.
    """

    def __init__(self, config):
        self.config = config

        self.parser = JsonOutputParser(pydantic_object=LLMVerificationResult)

        # Import prompts
        from prompts.llm_output_verification_prompts import get_llm_verification_prompts
        self.prompts = get_llm_verification_prompts()

        fact_logger.log_component_start("LLMOutputVerifier", model="gpt-4o (key rotation)")

    @traceable(
        name="verify_llm_interpretation",
        run_type="chain",
        tags=["llm-verification", "interpretation-check"]
    )
    async def verify_interpretation(
        self,
        claim: LLMClaim,
        excerpts_by_url: Dict[str, List[Dict]],
        scraped_content: Dict[str, str]
    ) -> LLMVerificationResult:
        """
        Verify if LLM accurately interpreted its cited sources

        Checks against ALL cited sources (handles [4][6][9] style citations)

        Args:
            claim: The LLMClaim object with the LLM's claim text and cited_sources list
            excerpts_by_url: Excerpts extracted by Highlighter {url: [excerpts]}
            scraped_content: Full source content {url: content}

        Returns:
            LLMVerificationResult with verification assessment
        """
        start_time = time.time()

        fact_logger.logger.info(
            f"Verifying LLM interpretation for {claim.id}",
            extra={
                "claim_id": claim.id, 
                "num_cited_sources": len(claim.cited_sources)
            }
        )

        # Handle multiple cited sources
        all_excerpts = []
        available_sources = []
        missing_sources = []

        # Process each cited source
        for cited_url in claim.cited_sources:
            if cited_url in scraped_content:
                available_sources.append(cited_url)
                source_excerpts = excerpts_by_url.get(cited_url, [])

                fact_logger.logger.debug(
                    f"  Source available: {cited_url} ({len(source_excerpts)} excerpts)",
                    extra={"claim_id": claim.id, "source_url": cited_url}
                )

                # Tag each excerpt with its source URL for multi-source verification
                for excerpt in source_excerpts:
                    excerpt_with_source = excerpt.copy()
                    excerpt_with_source['source_url'] = cited_url
                    all_excerpts.append(excerpt_with_source)
            else:
                missing_sources.append(cited_url)
                fact_logger.logger.warning(
                    f"  Source NOT available: {cited_url}",
                    extra={"claim_id": claim.id, "source_url": cited_url}
                )

        # Check if we have any available sources
        if not available_sources:
            fact_logger.logger.error(
                f"None of the {len(claim.cited_sources)} cited sources are available for {claim.id}",
                extra={"claim_id": claim.id, "missing_sources": missing_sources}
            )
            return self._create_error_result(
                claim, 
                f"None of the {len(claim.cited_sources)} cited sources are available"
            )

        # Log verification details
        fact_logger.logger.info(
            f"  Checking {claim.id} against {len(available_sources)}/{len(claim.cited_sources)} sources ({len(all_excerpts)} total excerpts)",
            extra={
                "claim_id": claim.id,
                "available_sources": len(available_sources),
                "total_sources": len(claim.cited_sources),
                "total_excerpts": len(all_excerpts)
            }
        )

        # Format excerpts from all sources for verification
        formatted_excerpts = self._format_multi_source_excerpts(
            all_excerpts, 
            available_sources,
            scraped_content
        )

        # Build verification prompt with multi-source context
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts["system"]),
            ("user", self.prompts["user"])
        ])

        # Prepare source metadata for the prompt
        sources_info = "\n".join([
            f"[Source {i+1}]: {url}" 
            for i, url in enumerate(available_sources)
        ])

        # Execute verification with GPT-4o
        # FRESH LLM per call -- rotated API key for parallel safety
        try:
            fact_logger.logger.debug(
                f"  Calling LLM for verification of {claim.id}",
                extra={"claim_id": claim.id, "model": "gpt-4o"}
            )

            llm = get_openai_llm(model="gpt-4o", temperature=0, json_mode=True)
            chain = prompt | llm | self.parser

            # Inject current date awareness
            now = datetime.now()

            response = await chain.ainvoke({
                "claim_text": claim.claim_text,
                "claim_context": claim.context,
                "excerpts": formatted_excerpts,
                "sources_info": sources_info,
                "num_sources": len(available_sources),
                "format_instructions": self.parser.get_format_instructions(),
                "current_date": now.strftime("%B %d, %Y")
            })

            # Build result
            result = LLMVerificationResult(
                claim_id=claim.id,
                claim_text=claim.claim_text,
                verification_score=response.get('verification_score', 0.0),
                assessment=response.get('assessment', 'No assessment provided'),
                interpretation_issues=response.get('interpretation_issues', []),
                wording_comparison=response.get('wording_comparison', {}),
                confidence=response.get('confidence', 0.5),
                reasoning=response.get('reasoning', 'No reasoning provided'),
                excerpts=all_excerpts,  # Store all excerpts with source tags
                cited_source_urls=available_sources
            )

            # Add warning about missing sources if any
            if missing_sources:
                missing_warning = f"{len(missing_sources)} cited source(s) were unavailable: {', '.join([self._shorten_url(url) for url in missing_sources])}"
                result.interpretation_issues.insert(0, missing_warning)

                fact_logger.logger.warning(
                    f"  {claim.id}: Some sources unavailable",
                    extra={
                        "claim_id": claim.id,
                        "missing_count": len(missing_sources),
                        "missing_sources": missing_sources
                    }
                )

            elapsed_time = time.time() - start_time

            fact_logger.logger.info(
                f"  Verification complete for {claim.id}: {result.verification_score:.2f} ({elapsed_time:.1f}s)",
                extra={
                    "claim_id": claim.id,
                    "score": result.verification_score,
                    "duration": elapsed_time,
                    "sources_checked": len(available_sources)
                }
            )

            return result

        except Exception as e:
            fact_logger.logger.error(
                f"Verification failed for {claim.id}: {str(e)}",
                extra={"claim_id": claim.id, "error": str(e)}
            )
            return self._create_error_result(claim, f"Verification error: {str(e)}")

    def _format_multi_source_excerpts(
        self, 
        excerpts: List[Dict], 
        source_urls: List[str],
        scraped_content: Dict[str, str]
    ) -> str:
        """
        Format excerpts from multiple sources for verification prompt

        Groups excerpts by source and labels them clearly
        """
        if not excerpts:
            return "No relevant excerpts found in any cited source."

        # Group excerpts by source URL
        excerpts_by_source = {}
        for excerpt in excerpts:
            source_url = excerpt.get('source_url', 'unknown')
            if source_url not in excerpts_by_source:
                excerpts_by_source[source_url] = []
            excerpts_by_source[source_url].append(excerpt)

        # Format output
        formatted_parts = []

        for idx, source_url in enumerate(source_urls, 1):
            source_excerpts = excerpts_by_source.get(source_url, [])

            formatted_parts.append(f"\n{'='*80}")
            formatted_parts.append(f"SOURCE [{idx}]: {source_url}")
            formatted_parts.append(f"{'='*80}")

            if source_excerpts:
                formatted_parts.append(f"\nFound {len(source_excerpts)} relevant excerpt(s):\n")

                for i, excerpt in enumerate(source_excerpts, 1):
                    relevance = excerpt.get('relevance', 0.0)
                    quote = excerpt.get('quote', '')
                    context = excerpt.get('context', '')

                    formatted_parts.append(f"\nExcerpt {i} (Relevance: {relevance:.2f}):")
                    formatted_parts.append(f'Quote: "{quote}"')
                    if context and context != quote:
                        formatted_parts.append(f'Context: "{context}"')
                    formatted_parts.append("")
            else:
                formatted_parts.append("\nNo relevant excerpts found in this source.\n")

        return "\n".join(formatted_parts)


    def _shorten_url(self, url: str, max_length: int = 50) -> str:
        """Shorten URL for display in warnings"""
        if len(url) <= max_length:
            return url
        return url[:max_length-3] + "..."


    def _create_error_result(self, claim: LLMClaim, error_message: str) -> LLMVerificationResult:
        """Create an error result when verification cannot be performed"""
        return LLMVerificationResult(
            claim_id=claim.id,
            claim_text=claim.claim_text,
            verification_score=0.0,
            assessment=f"ERROR: {error_message}",
            interpretation_issues=[error_message],
            wording_comparison={},
            confidence=0.0,
            reasoning=f"Could not verify claim: {error_message}",
            excerpts=[],
            cited_source_urls=[]
        )

    def _format_excerpts(self, excerpts: List[Dict]) -> str:
        """Format excerpts for prompt"""
        if not excerpts:
            return "No excerpts found"

        formatted = []
        for i, ex in enumerate(excerpts, 1):
            formatted.append(
                f"EXCERPT #{i} (Relevance: {ex.get('relevance', 0):.2f}):\n"
                f"{ex.get('quote', '')}\n"
            )

        return "\n".join(formatted)
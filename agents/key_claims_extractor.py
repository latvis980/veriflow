# agents/key_claims_extractor.py
"""
Key Claims Extractor Agent - ENHANCED VERSION
Extracts the 2-3 central thesis claims from text PLUS content analysis

ENHANCEMENTS:
- Uses GPT-4o for more nuanced analysis
- Extracts broad_context: content credibility assessment
- Extracts me_process_responsedia_sources: all sources mentioned in text
- Generates query_instructions: strategic guidance for query generator

These new fields help the query generator create more targeted, effective searches.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time

from prompts.key_claims_extractor_prompts import get_key_claims_prompts
from utils.logger import fact_logger
from utils.langsmith_config import langsmith_config


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class KeyClaim(BaseModel):
    """A key claim (central thesis) from the text"""
    id: str
    statement: str
    sources: List[str]  # Will be empty for plain text
    original_text: str
    confidence: float


class ContentLocation(BaseModel):
    """Geographic and language context for the content"""
    country: str = Field(default="international", description="Primary country where events take place")
    country_code: str = Field(default="", description="ISO 2-letter country code")
    language: str = Field(default="english", description="Primary language for that country")
    confidence: float = Field(default=0.5, description="Confidence in location detection")


class BroadContext(BaseModel):
    """Assessment of content's overall credibility indicators"""
    content_type: str = Field(
        default="unknown",
        description="Type: news article | blog post | social media | press release | unknown"
    )
    credibility_assessment: str = Field(
        default="unknown",
        description="Assessment: appears legitimate | some concerns | significant red flags | likely hoax/satire"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the credibility assessment"
    )
    red_flags: List[str] = Field(
        default_factory=list,
        description="List of concerning indicators found"
    )
    positive_indicators: List[str] = Field(
        default_factory=list,
        description="List of credibility-boosting factors"
    )


class QueryInstructions(BaseModel):
    """Strategic instructions for query generation"""
    primary_strategy: str = Field(
        default="standard verification",
        description="Overall approach for searching"
    )
    suggested_modifiers: List[str] = Field(
        default_factory=list,
        description="Terms to add to queries (e.g., 'hoax', 'official', 'announcement')"
    )
    temporal_guidance: str = Field(
        default="recent",
        description="Time-based search guidance"
    )
    source_priority: List[str] = Field(
        default_factory=list,
        description="Types of sources to prioritize"
    )
    special_considerations: str = Field(
        default="",
        description="Any other relevant guidance"
    )


class KeyClaimsOutput(BaseModel):
    """Complete output from key claims extraction"""
    facts: List[dict] = Field(description="List of 2-3 key claims")
    all_sources: List[str] = Field(description="All source URLs mentioned")
    content_location: Optional[dict] = Field(default=None, description="Country and language info")
    # NEW FIELDS
    broad_context: Optional[dict] = Field(default=None, description="Content credibility assessment")
    media_sources: List[str] = Field(default_factory=list, description="Media sources mentioned")
    query_instructions: Optional[dict] = Field(default=None, description="Instructions for query generator")


class KeyClaimsResult(BaseModel):
    """Complete result from key claims extraction including all analysis"""
    claims: List[KeyClaim]
    all_sources: List[str]
    content_location: ContentLocation
    # NEW FIELDS
    broad_context: BroadContext
    media_sources: List[str]
    query_instructions: QueryInstructions


# ============================================================================
# MAIN EXTRACTOR CLASS
# ============================================================================

class KeyClaimsExtractor:
    """
    Extract the 2-3 key claims (central thesis) from text
    PLUS content analysis for smarter query generation
    
    Uses GPT-4o for more nuanced analysis.
    """

    def __init__(self, config):
        self.config = config

        # UPGRADED: Using GPT-4o for better analysis
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        ).bind(response_format={"type": "json_object"})

        self.parser = JsonOutputParser(pydantic_object=KeyClaimsOutput)
        self.prompts = get_key_claims_prompts()

        # Large file support
        self.max_input_chars = 100000  # ~25k tokens

        fact_logger.log_component_start(
            "KeyClaimsExtractor",
            model="gpt-4o",  # Updated
            max_claims=3
        )

    @traceable(name="key_claims_extraction")
    async def extract(self, parsed_content: dict) -> tuple[List[KeyClaim], List[str], ContentLocation, BroadContext, List[str], QueryInstructions]:
        """
        Extract 2-3 key claims from parsed content with full analysis
        
        Args:
            parsed_content: Dict with 'text', 'links', 'format' keys
            
        Returns:
            Tuple of (key_claims, all_sources, content_location, broad_context, media_sources, query_instructions)
        """
        start_time = time.time()

        text_length = len(parsed_content.get('text', ''))
        fact_logger.logger.info(
            f"🎯 Starting key claims extraction (enhanced)",
            extra={
                "text_length": text_length,
                "num_links": len(parsed_content.get('links', []))
            }
        )

        # Check if we need chunking (for very large files)
        if text_length > self.max_input_chars:
            fact_logger.logger.info(f"📄 Large content detected ({text_length} chars), using chunked extraction")
            result = await self._extract_with_chunking(parsed_content)
        else:
            result = await self._extract_single_pass(parsed_content)

        claims, sources, location, broad_context, media_sources, query_instructions = result

        duration = time.time() - start_time
        fact_logger.logger.info(
            f"✅ Key claims extraction complete (enhanced)",
            extra={
                "num_claims": len(claims),
                "duration_seconds": round(duration, 2),
                "country": location.country,
                "language": location.language,
                "content_type": broad_context.content_type,
                "credibility": broad_context.credibility_assessment,
                "num_media_sources": len(media_sources)
            }
        )

        return claims, sources, location, broad_context, media_sources, query_instructions

    async def _extract_single_pass(self, parsed_content: dict) -> tuple:
        """Extract key claims and analysis in a single LLM call"""

        # ✅ FIX: Check for minimum content first
        text = parsed_content.get('text', '')
        if not text or len(text.strip()) < 50:
            fact_logger.logger.warning(f"⚠️ Insufficient text content ({len(text)} chars)")
            return self._get_empty_result(
                parsed_content, 
                f"Content too short for analysis ({len(text)} characters). Please provide more text."
            )

        system_prompt = self.prompts["system"]
        user_prompt = self.prompts["user"]

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No other text."),
            ("user", user_prompt + "\n\n{format_instructions}\n\nReturn your response as valid JSON.")
        ])

        prompt_with_format = prompt.partial(
            format_instructions=self.parser.get_format_instructions()
        )

        callbacks = langsmith_config.get_callbacks("key_claims_extractor")
        chain = prompt_with_format | self.llm | self.parser

        fact_logger.logger.debug("🔗 Invoking LangChain for key claims extraction (GPT-4o)")

        try:
            response = await chain.ainvoke(
                {
                    "text": parsed_content['text'],
                    "sources": self._format_sources(parsed_content['links'])
                },
                config={"callbacks": callbacks.handlers}
            )

            return self._process_response(response, parsed_content)

        except Exception as e:
            error_msg = str(e)
            fact_logger.logger.error(f"❌ LLM invocation failed: {error_msg}")

            # ✅ FIX: Return graceful result instead of raising
            if "'NoneType'" in error_msg or "has no attribute" in error_msg:
                fact_logger.logger.warning("⚠️ LLM returned malformed response, using defaults")
                return self._get_empty_result(
                    parsed_content,
                    "The content could not be analyzed. It may not contain verifiable factual claims."
                )

            # For other errors, log traceback and re-raise
            import traceback
            fact_logger.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _extract_with_chunking(self, parsed_content: dict) -> tuple:
        """Extract key claims from large content by splitting into chunks"""

        text = parsed_content['text']
        chunk_size = self.max_input_chars - 10000  # Reserve space for prompts

        chunks = self._split_into_chunks(text, chunk_size)

        fact_logger.logger.info(
            f"📄 Split content into {len(chunks)} chunks",
            extra={"num_chunks": len(chunks)}
        )

        all_claims = []
        all_location_votes = []
        all_broad_contexts = []
        all_media_sources = []
        all_query_instructions = []

        for i, chunk in enumerate(chunks, 1):
            fact_logger.logger.debug(f"🔍 Analyzing chunk {i}/{len(chunks)}")

            chunk_parsed = {
                'text': chunk,
                'links': parsed_content['links'],
                'format': parsed_content.get('format', 'unknown')
            }

            chunk_result = await self._extract_single_pass(chunk_parsed)
            chunk_claims, _, chunk_location, chunk_context, chunk_media, chunk_instructions = chunk_result
            
            all_claims.extend(chunk_claims)
            all_location_votes.append(chunk_location)
            all_broad_contexts.append(chunk_context)
            all_media_sources.extend(chunk_media)
            all_query_instructions.append(chunk_instructions)

        # Deduplicate and rank claims
        unique_claims = self._deduplicate_and_rank_claims(all_claims)

        # Get all sources from parsed content
        all_sources = [link['url'] for link in parsed_content['links']]

        # Aggregate location votes
        content_location = self._aggregate_location_votes(all_location_votes)

        # Aggregate broad context (use most concerning assessment)
        broad_context = self._aggregate_broad_context(all_broad_contexts)

        # Deduplicate media sources
        unique_media = list(set(all_media_sources))

        # Merge query instructions
        query_instructions = self._merge_query_instructions(all_query_instructions)

        return unique_claims, all_sources, content_location, broad_context, unique_media, query_instructions

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks, trying to break at paragraph boundaries"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))

            if end_pos < len(text):
                # Try to find a paragraph break
                para_break = text.rfind('\n\n', current_pos, end_pos)
                if para_break > current_pos + chunk_size // 2:
                    end_pos = para_break + 2

            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

        return chunks

    def _process_response(self, response: dict, parsed_content: dict) -> tuple:
        """Process LLM response into structured output with defensive None handling"""

        # ✅ FIX: Handle case where response is None or empty
        if not response:
            fact_logger.logger.warning("⚠️ Empty or None response from LLM, returning defaults")
            return self._get_empty_result(parsed_content, "Insufficient content for analysis")

        # ✅ FIX: Check if response indicates no verifiable claims
        facts_list = response.get('facts')
        if facts_list is None or (isinstance(facts_list, list) and len(facts_list) == 0):
            # Check if the LLM explicitly said there are no claims
            reasoning = ""
            ctx = response.get('broad_context')
            if ctx and isinstance(ctx, dict):
                reasoning = ctx.get('reasoning', '')

            if not reasoning:
                reasoning = "No verifiable factual claims found in this content."

            fact_logger.logger.info(f"ℹ️ No verifiable claims found: {reasoning}")
            # Still process what we can, but return empty claims list

        # Extract claims
        claims = []
        facts_list = response.get('facts') or []  # ✅ FIX: Use 'or []' for None
        for i, fact in enumerate(facts_list, 1):
            if fact:  # Skip None entries
                claims.append(KeyClaim(
                    id=fact.get('id', f'KC{i}'),
                    statement=fact.get('statement', ''),
                    sources=fact.get('sources') or [],  # ✅ FIX
                    original_text=fact.get('original_text', ''),
                    confidence=fact.get('confidence', 0.5)
                ))

        # Get all sources
        all_sources = response.get('all_sources') or []  # ✅ FIX
        if not all_sources:
            all_sources = [link['url'] for link in parsed_content.get('links', [])]

        # ✅ FIX: Parse content location with defensive None handling
        loc_data = response.get('content_location')
        if loc_data is None:
            loc_data = {}
        content_location = ContentLocation(
            country=loc_data.get('country', 'international'),
            country_code=loc_data.get('country_code', ''),
            language=loc_data.get('language', 'english'),
            confidence=loc_data.get('confidence', 0.5)
        )

        # ✅ FIX: Parse broad context with defensive None handling
        ctx_data = response.get('broad_context')
        if ctx_data is None:
            ctx_data = {}
        broad_context = BroadContext(
            content_type=ctx_data.get('content_type', 'unknown'),
            credibility_assessment=ctx_data.get('credibility_assessment', 'unknown'),
            reasoning=ctx_data.get('reasoning', ''),
            red_flags=ctx_data.get('red_flags') or [],  # ✅ FIX
            positive_indicators=ctx_data.get('positive_indicators') or []  # ✅ FIX
        )

        # ✅ FIX: Extract media sources with None handling
        media_sources = response.get('media_sources') or []

        # ✅ FIX: Parse query instructions with defensive None handling
        qi_data = response.get('query_instructions')
        if qi_data is None:
            qi_data = {}
        query_instructions = QueryInstructions(
            primary_strategy=qi_data.get('primary_strategy', 'standard verification'),
            suggested_modifiers=qi_data.get('suggested_modifiers') or [],  # ✅ FIX
            temporal_guidance=qi_data.get('temporal_guidance', 'recent'),
            source_priority=qi_data.get('source_priority') or [],  # ✅ FIX
            special_considerations=qi_data.get('special_considerations', '')
        )

        return claims, all_sources, content_location, broad_context, media_sources, query_instructions

    def _get_empty_result(self, parsed_content: dict, reason: str = "") -> tuple:
        """Return a valid empty result tuple with defaults"""
        all_sources = [link['url'] for link in parsed_content.get('links', [])]

        return (
            [],  # claims - empty list
            all_sources,
            ContentLocation(),  # defaults
            BroadContext(
                content_type='unknown',
                credibility_assessment='unknown',
                reasoning=reason or 'No verifiable claims could be extracted from this content.'
            ),
            [],  # media_sources
            QueryInstructions(primary_strategy='standard verification')
        )


    def _format_sources(self, links: list) -> str:
        """Format source links for the prompt"""
        if not links:
            return "No source links provided"

        formatted = []
        for link in links:
            if isinstance(link, dict):
                url = link.get('url', '')
                text = link.get('text', '')
                formatted.append(f"- {text}: {url}" if text else f"- {url}")
            else:
                formatted.append(f"- {link}")

        return "\n".join(formatted)

    def _deduplicate_and_rank_claims(self, claims: List[KeyClaim]) -> List[KeyClaim]:
        """Remove duplicate claims and keep top 2-3 by confidence"""
        # Simple deduplication by statement similarity
        seen = set()
        unique = []
        for claim in claims:
            # Normalize for comparison
            normalized = claim.statement.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(claim)

        # Sort by confidence and keep top 3
        unique.sort(key=lambda c: c.confidence, reverse=True)
        return unique[:3]

    def _aggregate_location_votes(self, votes: List[ContentLocation]) -> ContentLocation:
        """Combine location votes from multiple chunks"""
        if not votes:
            return ContentLocation()

        # Simple: return highest confidence vote
        return max(votes, key=lambda v: v.confidence)

    def _aggregate_broad_context(self, contexts: List[BroadContext]) -> BroadContext:
        """Combine broad context assessments - use most concerning"""
        if not contexts:
            return BroadContext()

        # Priority order (most concerning first)
        priority = {
            'likely hoax/satire': 0,
            'significant red flags': 1,
            'some concerns': 2,
            'appears legitimate': 3,
            'unknown': 4
        }

        # Return most concerning assessment
        return min(contexts, key=lambda c: priority.get(c.credibility_assessment, 4))

    def _merge_query_instructions(self, instructions: List[QueryInstructions]) -> QueryInstructions:
        """Merge query instructions from multiple chunks"""
        if not instructions:
            return QueryInstructions()

        if len(instructions) == 1:
            return instructions[0]

        # Merge all modifiers and source priorities
        all_modifiers = []
        all_priorities = []
        
        for inst in instructions:
            all_modifiers.extend(inst.suggested_modifiers)
            all_priorities.extend(inst.source_priority)

        # Use first strategy and temporal guidance
        return QueryInstructions(
            primary_strategy=instructions[0].primary_strategy,
            suggested_modifiers=list(set(all_modifiers)),
            temporal_guidance=instructions[0].temporal_guidance,
            source_priority=list(set(all_priorities)),
            special_considerations=instructions[0].special_considerations
        )

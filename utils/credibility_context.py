# utils/credibility_context.py
"""
Utility for building credibility context strings to inject into analysis prompts.
This allows the AI to consider source reliability when analyzing content.
"""

from typing import Optional, Dict, Any


async def build_bias_analysis_context_async(
    source_credibility: Optional[Dict[str, Any]] = None,
    publication_name: Optional[str] = None
) -> str:
    """
    Async version of build_bias_analysis_context.
    Calls Claude Haiku to turn raw MBFC fields into a clean,
    human-readable publication profile sentence.

    Falls back to the raw sync version if the LLM call fails.

    Args:
        source_credibility: Dict with MBFC data (bias_rating, factual_reporting, etc.)
        publication_name: Fallback publication name

    Returns:
        Clean formatted context string
    """
    if not source_credibility:
        if publication_name:
            return f"\nPUBLICATION: {publication_name}\n(No prior bias data available)"
        return ""

    pub_name = source_credibility.get('publication_name') or publication_name

    # Collect the fields we want to summarize
    raw_fields = {}
    bias = source_credibility.get('bias_rating')
    factual = source_credibility.get('factual_reporting')
    rating = source_credibility.get('rating') or source_credibility.get('credibility_rating')
    special_tags = source_credibility.get('special_tags', [])

    if bias:
        raw_fields['bias_rating'] = bias
    if factual:
        raw_fields['factual_reporting'] = factual
    if rating:
        raw_fields['credibility_rating'] = rating
    if special_tags:
        raw_fields['special_tags'] = special_tags

    # If we have nothing useful, skip LLM and return minimal context
    if not pub_name or not raw_fields:
        return build_bias_analysis_context(source_credibility, publication_name)

    try:
        from langchain_anthropic import ChatAnthropic
        from prompts.credibility_context_prompts import (
            PUBLICATION_SUMMARY_SYSTEM,
            PUBLICATION_SUMMARY_USER,
        )
        from langchain.prompts import ChatPromptTemplate

        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=200)

        prompt = ChatPromptTemplate.from_messages([
            ("system", PUBLICATION_SUMMARY_SYSTEM),
            ("user", PUBLICATION_SUMMARY_USER),
        ])

        chain = prompt | llm

        response = await chain.ainvoke({
            "publication_name": pub_name,
            "raw_data": str(raw_fields),
        })

        summary = response.content.strip()

        if summary:
            return f"\nPUBLICATION CONTEXT:\n{summary}"

    except Exception as e:
        # Non-fatal: log and fall back to raw context
        try:
            from utils.logger import fact_logger
            fact_logger.logger.warning(
                f"Publication context summarization failed for '{pub_name}': {e} — using raw fallback"
            )
        except Exception:
            pass

    # Fallback: raw MBFC fields
    return build_bias_analysis_context(source_credibility, publication_name)


def build_credibility_context(
    source_credibility: Optional[Dict[str, Any]] = None,
    publication_name: Optional[str] = None,
    include_guidance: bool = True
) -> str:
    """
    Build a credibility context string for injection into prompts.
    
    Args:
        source_credibility: Dict with tier, bias_rating, factual_reporting, etc.
        publication_name: Name of the publication (fallback if not in credibility)
        include_guidance: Whether to include analysis guidance based on tier
        
    Returns:
        Formatted string for prompt injection, or empty string if no data
    """
    if not source_credibility:
        if publication_name:
            return f"\n\nPUBLICATION: {publication_name}\n(No credibility data available - treat with standard scrutiny)"
        return ""
    
    parts = ["", "=" * 50, "SOURCE CREDIBILITY CONTEXT", "=" * 50]
    
    # Publication name
    pub_name = source_credibility.get('publication_name') or publication_name
    if pub_name:
        parts.append(f"Publication: {pub_name}")
    
    # Credibility tier
    tier = source_credibility.get('tier') or source_credibility.get('credibility_tier')
    if tier:
        tier_labels = {
            1: "TIER 1 - Highly Credible (Official sources, major wire services)",
            2: "TIER 2 - Credible (Reputable mainstream media)",
            3: "TIER 3 - Mixed (Requires verification, may have bias)",
            4: "TIER 4 - Low Credibility (Significant bias or poor factual reporting)",
            5: "TIER 5 - Unreliable (Propaganda, conspiracy, or disinformation)"
        }
        parts.append(f"Credibility: {tier_labels.get(tier, f'Tier {tier}')}")
    
    # Bias rating
    bias = source_credibility.get('bias_rating')
    if bias:
        parts.append(f"Political Bias: {bias}")
    
    # Factual reporting
    factual = source_credibility.get('factual_reporting')
    if factual:
        parts.append(f"Factual Reporting: {factual}")
    
    # Special tags (propaganda, conspiracy, etc.)
    special_tags = source_credibility.get('special_tags', [])
    if special_tags:
        parts.append(f"⚠️ Special Tags: {', '.join(special_tags)}")
    
    # Is propaganda flag
    if source_credibility.get('is_propaganda'):
        parts.append("⚠️ WARNING: This source is flagged as PROPAGANDA")
    
    # MBFC source link
    mbfc_url = source_credibility.get('mbfc_url')
    if mbfc_url:
        parts.append(f"MBFC Reference: {mbfc_url}")
    
    parts.append("=" * 50)
    
    # Add analysis guidance based on tier
    if include_guidance and tier:
        guidance = get_tier_guidance(tier, special_tags)
        if guidance:
            parts.append("")
            parts.append("ANALYSIS GUIDANCE:")
            parts.append(guidance)
    
    return "\n".join(parts)


def get_tier_guidance(tier: int, special_tags: list = None) -> str:
    """
    Get analysis guidance based on credibility tier.
    
    Args:
        tier: Credibility tier (1-5)
        special_tags: List of special tags like PROPAGANDA, CONSPIRACY
        
    Returns:
        Guidance string for the AI analyst
    """
    special_tags = special_tags or []
    
    # Check for critical tags first
    critical_tags = {'PROPAGANDA', 'CONSPIRACY-PSEUDOSCIENCE', 'QUESTIONABLE SOURCE'}
    has_critical = bool(set(t.upper() for t in special_tags) & critical_tags)
    
    if has_critical:
        return """⚠️ CRITICAL: This source has been flagged for serious credibility issues.
- Approach ALL claims with extreme skepticism
- Look for verifiable facts vs. opinion/speculation
- Note any inflammatory or manipulative language
- Do NOT assume any factual claims are accurate without independent verification
- Highlight potential misinformation or misleading framing"""
    
    guidance_map = {
        1: """This is a highly credible source. While still applying critical analysis:
- Claims are more likely to be factually accurate
- Focus analysis on framing, emphasis, and what may be omitted
- Look for editorial slant even in factual reporting
- Note if the source is reporting vs. editorializing""",
        
        2: """This is a credible mainstream source. Apply standard analysis:
- Claims are generally reliable but verify significant facts
- Watch for political lean in framing and word choice
- Note selective emphasis or omission of context
- Distinguish between news reporting and opinion content""",
        
        3: """This source has mixed credibility. Apply heightened scrutiny:
- Verify key factual claims independently
- Watch for bias in framing and source selection
- Note emotional or loaded language
- Be alert to potential cherry-picking of facts
- Consider what perspectives may be missing""",
        
        4: """This is a low-credibility source. Apply significant skepticism:
- Do NOT assume factual accuracy of claims
- Look for verifiable facts vs. opinion presented as fact
- Note manipulation techniques and emotional appeals
- Check if claims contradict established consensus
- Highlight potential misinformation""",
        
        5: """⚠️ This is an unreliable source. Apply maximum skepticism:
- Treat ALL claims as potentially false or misleading
- Look for propaganda techniques and manipulation
- Note conspiracy theories or pseudoscience
- Identify emotional manipulation and fear tactics
- Flag any claims that could cause harm if believed"""
    }
    
    return guidance_map.get(tier, "Apply standard critical analysis.")


def build_bias_analysis_context(
    source_credibility: Optional[Dict[str, Any]] = None,
    publication_name: Optional[str] = None
) -> str:
    """
    Build context specifically for bias analysis.
    Includes MBFC data if available to inform the analysis.
    
    Args:
        source_credibility: Dict with MBFC data
        publication_name: Fallback publication name
        
    Returns:
        Formatted context string
    """
    if not source_credibility:
        if publication_name:
            return f"\nPUBLICATION: {publication_name}\n(No prior bias data available)"
        return ""
    
    parts = ["", "MEDIA BIAS/FACT CHECK DATA (if available):"]
    
    pub_name = source_credibility.get('publication_name') or publication_name
    if pub_name:
        parts.append(f"Publication: {pub_name}")
    
    bias = source_credibility.get('bias_rating')
    if bias:
        parts.append(f"MBFC Bias Rating: {bias}")
    
    factual = source_credibility.get('factual_reporting')
    if factual:
        parts.append(f"MBFC Factual Reporting: {factual}")
    
    rating = source_credibility.get('rating') or source_credibility.get('credibility_rating')
    if rating:
        parts.append(f"MBFC Credibility: {rating}")
    
    special_tags = source_credibility.get('special_tags', [])
    if special_tags:
        parts.append(f"MBFC Tags: {', '.join(special_tags)}")
    
    parts.append("")
    parts.append("NOTE: Use this MBFC data as context, but perform your own independent analysis.")
    parts.append("Your analysis may agree or disagree with MBFC - explain your reasoning.")
    
    return "\n".join(parts)


def build_lie_detection_context(
    source_credibility: Optional[Dict[str, Any]] = None,
    article_source: Optional[str] = None,
    article_date: Optional[str] = None
) -> str:
    """
    Build context for lie detection analysis.
    Focuses on source reliability for calibrating suspicion levels.
    
    Args:
        source_credibility: Dict with credibility data
        article_source: Publication name
        article_date: Article publication date
        
    Returns:
        Formatted context string
    """
    parts = []
    
    if article_source:
        parts.append(f"ARTICLE SOURCE: {article_source}")
    
    if article_date:
        parts.append(f"PUBLICATION DATE: {article_date}")
    
    if source_credibility:
        tier = source_credibility.get('tier') or source_credibility.get('credibility_tier')
        
        if tier:
            parts.append(f"SOURCE CREDIBILITY TIER: {tier}/5")
            
            # Calibration guidance
            if tier <= 2:
                parts.append("CALIBRATION: This is a credible source. Linguistic deception markers")
                parts.append("should be weighted normally - don't over-flag professional journalism style.")
            elif tier == 3:
                parts.append("CALIBRATION: Mixed credibility source. Apply standard deception analysis.")
            elif tier >= 4:
                parts.append("CALIBRATION: Low credibility source. Be alert for deception patterns,")
                parts.append("but distinguish between poor journalism and intentional deception.")
        
        if source_credibility.get('is_propaganda'):
            parts.append("⚠️ SOURCE FLAGGED AS PROPAGANDA - expect manipulation techniques")
        
        special_tags = source_credibility.get('special_tags', [])
        if special_tags:
            parts.append(f"SOURCE FLAGS: {', '.join(special_tags)}")
    
    if not parts:
        return ""
    
    return "\n" + "\n".join(parts)


def build_manipulation_context(
    source_credibility: Optional[Dict[str, Any]] = None,
    source_info: Optional[str] = None
) -> str:
    """
    Build context for manipulation detection analysis.
    
    Args:
        source_credibility: Dict with credibility data
        source_info: Source description string
        
    Returns:
        Formatted context string
    """
    parts = ["", "SOURCE CONTEXT:"]
    
    if source_info:
        parts.append(f"Source: {source_info}")
    
    if source_credibility:
        tier = source_credibility.get('tier') or source_credibility.get('credibility_tier')
        bias = source_credibility.get('bias_rating')
        factual = source_credibility.get('factual_reporting')
        is_propaganda = source_credibility.get('is_propaganda')
        special_tags = source_credibility.get('special_tags', [])
        
        if tier:
            parts.append(f"Credibility Tier: {tier}/5")
        if bias:
            parts.append(f"Known Bias: {bias}")
        if factual:
            parts.append(f"Factual Reporting History: {factual}")
        if is_propaganda:
            parts.append("⚠️ FLAGGED AS PROPAGANDA SOURCE")
        if special_tags:
            parts.append(f"Flags: {', '.join(special_tags)}")
        
        # Guidance
        parts.append("")
        if tier and tier >= 4:
            parts.append("ANALYSIS NOTE: This is a low-credibility source. Manipulation techniques")
            parts.append("are more likely. Pay special attention to:")
            parts.append("- Cherry-picking or misrepresenting facts")
            parts.append("- Emotional manipulation and fear tactics")
            parts.append("- False equivalence and strawman arguments")
            parts.append("- Omission of contradicting evidence")
        elif is_propaganda or 'PROPAGANDA' in str(special_tags).upper():
            parts.append("ANALYSIS NOTE: This source is flagged for propaganda. Expect:")
            parts.append("- Deliberate framing to push specific narratives")
            parts.append("- Selective use of facts to support predetermined conclusions")
            parts.append("- Emotional appeals over factual arguments")
    
    if len(parts) <= 2:  # Only has header
        return ""
    
    return "\n".join(parts)


def format_credibility_for_summary(source_credibility: Optional[Dict[str, Any]]) -> str:
    """
    Format credibility data for inclusion in analysis summaries.
    
    Args:
        source_credibility: Dict with credibility data
        
    Returns:
        Short formatted string for summaries
    """
    if not source_credibility:
        return "Source credibility: Unknown"
    
    parts = []
    
    tier = source_credibility.get('tier') or source_credibility.get('credibility_tier')
    if tier:
        tier_names = {1: "Highly Credible", 2: "Credible", 3: "Mixed", 4: "Low", 5: "Unreliable"}
        parts.append(f"Tier {tier} ({tier_names.get(tier, 'Unknown')})")
    
    bias = source_credibility.get('bias_rating')
    if bias:
        parts.append(f"Bias: {bias}")
    
    if source_credibility.get('is_propaganda'):
        parts.append("⚠️ Propaganda")
    
    return " | ".join(parts) if parts else "Source credibility: Unknown"

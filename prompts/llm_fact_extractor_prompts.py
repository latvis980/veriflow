# prompts/llm_fact_extractor_prompts.py
"""
Prompts for LLM Fact Extractor
Extracts claim segments from LLM output for interpretation verification

PURPOSE: Map what the LLM said to which source it cited
NOT for atomizing claims or breaking them down
"""

SYSTEM_PROMPT = """You are an expert at analyzing LLM-generated content to identify claim segments and their cited sources.

YOUR TASK:
Extract claim segments from LLM output (ChatGPT, Perplexity, etc.) and map each to the source URL the LLM cited for it.

WHAT TO EXTRACT:
- Factual claim segments as written by the LLM
- Complete thoughts or statements, not individual facts
- Claims that have a specific source citation nearby
- The context surrounding each claim (for checking cherry-picking)

MULTIPLE CITATIONS:
When a claim has multiple source citations like [4][6][9], you MUST extract ALL of them.
- Identify all citation numbers in brackets near the claim
- Map the claim to ALL corresponding source URLs
- Return cited_sources as a LIST of URLs, not a single URL

EXAMPLE:
Input: "Takoyaki is topped with sauce, mayo, and bonito [4][6][9]"
Source Links: [1]: url1, [4]: url4, [6]: url6, [9]: url9

Output:
{{
  "claim_text": "Takoyaki is topped with sauce, mayo, and bonito",
  "cited_sources": ["url4", "url6", "url9"],
  ...
}}

IMPORTANT RULES:
1. **SELF-CONTAINED CLAIM TEXT**: Write claim_text as a fully self-contained sentence that makes sense with zero surrounding context. If the original phrasing is a fragment or uses pronouns like "it", "they", "this", "some" without a clear referent, restore the missing subject or topic using the surrounding context. The reader must understand exactly what is being claimed without reading anything else.
2. **MAP TO CITED SOURCE**: Each claim should be linked to the URL mentioned near it
3. **INCLUDE CONTEXT**: Capture surrounding text to check for selective quotation
4. **DON'T BREAK DOWN**: Keep compound statements together if they cite one source
5. **FOCUS ON SOURCE-BACKED CLAIMS**: Ignore opinions or unsupported statements

SELF-CONTAINED CLAIM EXAMPLES:
- Original fragment: "some are present, but not always in maximal form"
  Context: discussing omega-3 fatty acids in farmed salmon
  claim_text: "Omega-3 fatty acids are present in farmed salmon, but not always in maximal concentrations"

- Original fragment: "it has grown significantly since 2020"
  Context: about the global EV market
  claim_text: "The global EV market has grown significantly since 2020"

- Original: "AI models can now process images 3x faster than previous versions"
  claim_text: "AI models can now process images 3x faster than previous versions" (already self-contained, keep as-is)

WHAT YOU'RE LOOKING FOR:
- "According to [source], X happened..." -> Extract "X happened" + source URL
- "The study shows A, B, and C [1]" -> Extract "A, B, and C" + source [1]
- "As reported in [link], the company..." -> Extract claim + link

EXAMPLES:

Input LLM Output:
"According to a recent study from Stanford [https://example.com/study], AI models 
can now process images 3x faster than previous versions. The research also found 
significant improvements in accuracy. Meanwhile, other studies [https://other.com] 
suggest different approaches are needed."

Your Output:
{{
  "claims": [
    {{
      "claim_text": "AI models can now process images 3x faster than previous versions, with significant improvements in accuracy.",
      "cited_sources": ["https://example.com/study"],
      "context": "According to a recent study from Stanford [...], AI models can now process images 3x faster than previous versions. The research also found significant improvements in accuracy. Meanwhile, other studies suggest...",
      "confidence": 0.95
    }},
    {{
      "claim_text": "Other studies on AI image processing suggest different approaches are needed.",
      "cited_sources": ["https://other.com"],
      "context": "...The research also found significant improvements in accuracy. Meanwhile, other studies [https://other.com] suggest different approaches are needed.",
      "confidence": 0.85
    }}
  ],
  "all_sources": ["https://example.com/study", "https://other.com"]
}}

IMPORTANT: You MUST return valid JSON only. No other text or explanations.

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "claim_text": "A fully self-contained sentence describing the claim, enriched with context if the original was a fragment",
      "cited_sources": ["https://source-url1.com", "https://source-url2.com"],
      "context": "surrounding text for context",
      "confidence": 0.90
    }}
  ],
  "all_sources": ["https://url1.com", "https://url2.com"]
}}"""

USER_PROMPT = """Extract claim segments from the following LLM output and map each to its cited source.

LLM OUTPUT:
{llm_output}

SOURCE LINKS FOUND:
{source_links}

INSTRUCTIONS:
1. Identify factual claim segments in the LLM output
2. Write each claim_text as a fully self-contained sentence - if the original is a fragment or uses pronouns without a clear referent, use the surrounding context to restore the missing subject or topic
3. Map each claim to the source URL(s) cited nearby
4. Include surrounding context (before and after the claim)
5. Focus on source-backed claims, not opinions
6. Keep related claims together if they share one source
7. If multiple sources are cited together [1][2][3], include ALL of them in cited_sources list

IMPORTANT:
- claim_text must stand alone - a reader with no other context must understand what is being claimed
- Include enough context to check for cherry-picking
- The cited_sources should be a LIST of exact URLs from the source links
- If claim has [4][6][9], extract all three URLs into the cited_sources array

Extract all claim segments now."""


def get_llm_fact_extractor_prompts():
    """Return prompts for LLM fact extraction"""
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT
    }
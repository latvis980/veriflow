# prompts/key_claims_extractor_prompts.py
"""
Prompts for the Key Claims Extractor component - ENHANCED VERSION

Extracts:
1. Up to 5 MOST IMPORTANT verifiable claims/facts from text
2. broad_context: Quick AI assessment of the content type and credibility
3. media_sources: All media platforms mentioned or referenced
4. query_instructions: Strategic suggestions for downstream query generation
"""

SYSTEM_PROMPT = """You are an expert at identifying the most important VERIFIABLE CLAIMS and FACTS in any text, AND at analyzing content for credibility indicators.

IMPORTANT — CURRENT DATE AWARENESS:
- Today's date: {current_date}
- Current year: {current_year}
- When the text mentions "this year", "currently", "now", "today", or "recently", interpret these relative to the current date above
- Do NOT assume the current year is 2023 or 2024 — use the date provided above
- When extracting claims, include the actual year/date context so claims are temporally grounded

YOUR MISSION:
1. Read and understand the ENTIRE article as a whole — its narrative, argument, and purpose
2. Extract up to 5 KEY CLAIMS and FACTS that form the backbone of the article
3. Assess the overall content context and credibility indicators
4. Identify all media sources mentioned or referenced
5. Provide strategic instructions for search query generation

=== PART 1: KEY CLAIMS & FACTS EXTRACTION ===

STEP 1 — UNDERSTAND THE ARTICLE HOLISTICALLY:
Before extracting anything, read the full text and identify:
- What is this article ACTUALLY ABOUT? What is the core story or argument?
- What are the CENTRAL CLAIMS the author is making?
- What EVIDENCE or FACTS does the author use to support those claims?
- What is the narrative arc — how do the pieces fit together?

STEP 2 — SELECT UP TO 5 CLAIMS/FACTS:
Extract a mix of:
A) CENTRAL CLAIMS (1-2): The main assertions the article is built around — the headline-level takeaways
B) KEY SUPPORTING FACTS (2-3): The most important specific facts, data points, or evidence that substantiate the central claims

WHAT MAKES A GOOD EXTRACTION?
✅ Contains specific names (people, organizations, places)
✅ Contains dates, timeframes, or numbers
✅ Makes a concrete assertion that can be true or false
✅ Can be confirmed or denied by checking other sources
✅ Is important to the article's overall argument or story

CRITICAL — STATEMENT FORMATTING RULES:
Each "statement" MUST be a SELF-CONTAINED, COMPREHENSIVE fact statement that makes sense on its own WITHOUT reading the original article. This means:
- ALWAYS include the full names of people, organizations, and places (not just pronouns or abbreviations)
- ALWAYS include enough context so a reader who has never seen the article understands what is being claimed
- ALWAYS specify who did what, to whom, when, and where — do not assume prior knowledge
- Preserve the factual precision of the original wording (specific numbers, dates, quotes) but ADD context around it

EXAMPLES:
❌ BAD: "He was arrested on Tuesday" (Who? Where? For what?)
✅ GOOD: "John Smith, CEO of Acme Corp, was arrested in New York on Tuesday, March 10, 2026, on charges of securities fraud"

❌ BAD: "The report found a 40% increase" (What report? Increase in what?)
✅ GOOD: "The WHO's 2026 Global Health Report found a 40% increase in antibiotic-resistant infections across European hospitals compared to 2023"

❌ BAD: "Officials confirmed the deal" (Which officials? What deal?)
✅ GOOD: "U.S. Treasury officials confirmed a $2 billion trade agreement between the United States and Japan covering semiconductor exports"

WHAT TO AVOID:
❌ Thesis statements or interpretations ("This reveals courage...")
❌ Opinions or subjective judgments ("This is significant because...")
❌ Vague claims without specifics ("The investigation shows...")
❌ Vague generalizations ("Many people believe...")
❌ Author's conclusions or recommendations
❌ Redundant claims that cover the same fact from different angles

THE KEY TEST:
For each claim, ask:
1. "Can I search for this and find a source that confirms or denies it?" — If YES → good
2. "Does this statement make complete sense to someone who hasn't read the article?" — If YES → good
3. "Is this one of the most important things the article is saying?" — If YES → good

=== PART 2: BROAD CONTEXT ASSESSMENT ===

Analyze the overall content to assess:
- content_type: What kind of content is this? (news article, blog post, social media post, press release, academic paper, opinion piece, satire, unknown)
- credibility_assessment: Based on observable indicators, how credible does this content appear? (appears legitimate, some concerns, significant red flags, likely hoax/satire)
- reasoning: Brief explanation of your assessment
- red_flags: Any concerning indicators you observed (sensational language, missing sources, implausible claims, etc.)
- positive_indicators: Credibility-boosting factors (named sources, specific verifiable details, reputable publication markers, etc.)

=== PART 3: MEDIA SOURCES IDENTIFICATION ===

Identify ALL media platforms, publications, or information sources mentioned or referenced in the text:
- News outlets (newspapers, TV channels, news websites)
- Social media platforms (Twitter/X, Facebook, Instagram, TikTok, etc.)
- Wire services (Reuters, AP, AFP, etc.)
- Government or official sources
- Academic or research institutions
- Any other information sources cited or referenced

=== PART 4: QUERY INSTRUCTIONS ===

Based on your analysis, provide strategic guidance for generating effective search queries:
- primary_strategy: What overall approach should be used for searching? (standard verification, hoax checking, official source confirmation, etc.)
- suggested_modifiers: What terms might help narrow or focus searches? (e.g., "official", "announcement", "fact check", "debunked", specific date ranges, etc.)
- temporal_guidance: Is this time-sensitive? What time frame is relevant? (breaking/very recent, recent, historical, ongoing)
- source_priority: What types of sources should be prioritized for verification? (official government sites, news agencies, academic sources, etc.)
- special_considerations: Any other relevant guidance based on the content analysis

=== COUNTRY AND LANGUAGE DETECTION ===

Also detect the primary geographic focus:
- Identify the PRIMARY country where the main events/claims are situated
- Determine the main language of that country for search queries

IMPORTANT: You MUST return valid JSON only. No other text or explanations."""


USER_PROMPT = """Analyze the following text and extract:
1. Up to 5 KEY CLAIMS and FACTS (the central assertions + supporting evidence)
2. Broad context assessment (content type and credibility indicators)
3. All media sources mentioned or referenced
4. Strategic instructions for query generation

TEXT TO ANALYZE:
{text}

SOURCES MENTIONED:
{sources}

INSTRUCTIONS:
1. Read the ENTIRE text carefully — understand the full narrative and argument
2. Identify what the article is fundamentally about — its core story and thesis
3. Extract the 1-2 CENTRAL CLAIMS (the main assertions the article makes)
4. Extract 2-3 KEY SUPPORTING FACTS (specific evidence, data, events that back up the claims)
5. For each extraction, write a SELF-CONTAINED statement that includes full context (names, places, dates, organizations) so it makes sense on its own without the article
6. Preserve original precision (exact numbers, dates, quotes) but add surrounding context
7. Ensure each claim/fact is VERIFIABLE — can be checked against other sources
8. Assess the overall content type and credibility indicators
9. List all media sources/platforms mentioned
10. Provide strategic guidance for downstream query generation

QUALITY CHECKLIST for each statement:
- Does it make complete sense to someone who has NOT read the article? (Must be YES)
- Does it contain specific names, dates, places, or numbers? (Must be YES)
- Can someone search for this and verify it? (Must be YES)
- Is it a concrete assertion, not an interpretation? (Must be YES)
- Is it one of the most important things this article is claiming? (Must be YES)

Return your response as valid JSON with this structure:
{{
  "facts": [
    {{
      "id": "KC1",
      "statement": "A comprehensive, self-contained fact statement with full context — names, places, dates, and enough detail to be understood without the original article",
      "sources": [],
      "original_text": "The exact text from the article that states this fact",
      "confidence": 0.95
    }}
  ],
  "all_sources": ["list of all source URLs if any"],
  "content_location": {{
    "country": "primary country",
    "country_code": "XX",
    "language": "primary language",
    "confidence": 0.8
  }},
  "broad_context": {{
    "content_type": "type of content",
    "credibility_assessment": "your assessment",
    "reasoning": "brief explanation",
    "red_flags": ["list of concerning indicators"],
    "positive_indicators": ["list of credibility boosters"]
  }},
  "media_sources": ["list of all media platforms/publications mentioned"],
  "query_instructions": {{
    "primary_strategy": "recommended search approach",
    "suggested_modifiers": ["helpful search terms"],
    "temporal_guidance": "time-related guidance",
    "source_priority": ["types of sources to prioritize"],
    "special_considerations": "any other relevant guidance"
  }}
}}

Analyze the content and return valid JSON only."""


def get_key_claims_prompts():
    """Return prompts for key claims extraction"""
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT
    }

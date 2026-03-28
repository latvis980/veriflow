# prompts/credibility_context_prompts.py
"""
Prompt for summarizing raw MBFC credibility data into
a clean, human-readable publication context sentence.
Used by build_bias_analysis_context_async() in credibility_context.py.
"""

PUBLICATION_SUMMARY_SYSTEM = """You write brief, neutral publication profiles for use in media analysis.
Your output must be exactly 1-2 sentences. No preamble, no bullet points, just the sentence(s).
Never mention MBFC, Media Bias/Fact Check, or any rating organization by name."""

PUBLICATION_SUMMARY_USER = """Write a 1-2 sentence neutral profile of "{publication_name}" based on this data:

{raw_data}

Rules:
- Translate raw ratings into plain language:
    "LEAST BIASED" or "CENTER" -> "minimal political bias"
    "LEFT-CENTER" or "RIGHT-CENTER" -> "slight left/right-leaning political orientation"
    "LEFT" or "RIGHT" -> "left/right-leaning political orientation"
    "FAR LEFT" or "FAR RIGHT" -> "strong left/right political orientation"
    "MOSTLY FACTUAL" or "VERY HIGH" -> "strong factual reporting record"
    "HIGH" -> "reliable factual reporting"
    "MIXED" -> "mixed factual reporting record"
    "LOW" or "VERY LOW" -> "poor factual reporting record"
- Use neutral attribution like "international fact-checking organizations" or "independent media monitors"
- If special_tags include PROPAGANDA, QUESTIONABLE SOURCE, or CONSPIRACY-PSEUDOSCIENCE,
  note that the source has raised credibility concerns with media monitors
- Keep it professional and concise

Examples of good output:
"The Hill is considered a highly reputable source by international fact-checking organizations, with minimal political bias and a strong factual reporting record."
"The Guardian is recognized by independent media monitors for its slightly left-of-center editorial orientation and reliable factual reporting."
"Breitbart News is noted by media credibility organizations for its strong right-wing political orientation and mixed factual reporting history."
"RT (Russia Today) has been flagged by international media monitors for serious credibility concerns, including state-affiliated content."

Now write the profile for "{publication_name}":"""

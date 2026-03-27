"""
Prompts for AI-powered article content cleaning.

Guides the LLM to extract clean article text from noisy web page scrapes.
Output is plain text - no JSON, no structured fields.
Metadata extraction is handled by a separate agent.
"""

SYSTEM_PROMPT = """You are a precise article extractor. Your job is to strip a messy web scrape down to the actual journalism and nothing else.

Return only the article text: headline, byline if present, and body paragraphs. Preserve the original language and wording exactly - do not paraphrase or add anything.

Remove everything that is not part of the article itself:
- Subscription and paywall prompts ("Subscribe to read", "Sign in to continue", "X articles remaining")
- Device and session warnings ("Reading on another device", "Continue here")
- Navigation, menus, breadcrumbs, section labels
- Related articles, "Read also", "See also" blocks
- Newsletter signups, app download prompts, social sharing
- Cookie notices, legal boilerplate, footer content
- Comments sections, like/share counts
- Repeated content from page templates
- Standalone photographer or agency credits

If the article is behind a paywall and only a fragment is available, return just that fragment - do not fill in or invent the rest.

Return nothing but the clean article text. No commentary, no labels, no explanation."""


USER_PROMPT = """Extract the article text from this scraped content. Return only what belongs to the article itself.

URL: {url}

---
{content}
---"""


def get_content_cleaner_prompts() -> dict:
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT
    }
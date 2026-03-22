# utils/article_content_cleaner.py
"""
Article Content Cleaner
AI-powered extraction of actual article content from noisy scraped web pages.

Removes:
- Subscription/paywall prompts
- Navigation elements
- Cookie notices
- Device warnings ("read on one device")
- Related articles
- Comments sections
- Social sharing widgets
- Newsletter signups
- Advertisements
- Footer boilerplate

Preserves:
- Article headline
- Byline/author info
- Publication date
- Main article body paragraphs
- Relevant quotes
- Image captions (optionally)
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import json

import os
import re

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.logger import fact_logger


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class CleanedArticle(BaseModel):
    """Cleaned article content with metadata"""

    # Core content
    title: Optional[str] = Field(
        default=None,
        description="Article headline"
    )
    subtitle: Optional[str] = Field(
        default=None,
        description="Article subtitle or deck if present"
    )
    author: Optional[str] = Field(
        default=None,
        description="Author name(s)"
    )
    publication_date: Optional[str] = Field(
        default=None,
        description="Publication date as found in article"
    )

    # Main content
    body: str = Field(
        default="",
        description="Clean article body text"
    )

    # Optional elements
    lead_paragraph: Optional[str] = Field(
        default=None,
        description="Opening/lead paragraph if distinct from body"
    )
    image_captions: list = Field(
        default_factory=list,
        description="Image captions if relevant to content"
    )

    # Metadata
    word_count: int = Field(
        default=0,
        description="Word count of cleaned body"
    )
    cleaning_confidence: float = Field(
        default=0.0,
        description="Confidence in extraction quality (0-1)"
    )
    noise_removed: list = Field(
        default_factory=list,
        description="Types of noise that were removed"
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether article appears truncated (paywall)"
    )
    truncation_reason: Optional[str] = Field(
        default=None,
        description="Why article may be truncated"
    )


class CleaningResult(BaseModel):
    """Result of article cleaning operation"""
    success: bool
    cleaned: Optional[CleanedArticle] = None
    original_length: int = 0
    cleaned_length: int = 0
    reduction_percent: float = 0.0
    error: Optional[str] = None


# ============================================================================
# PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are an expert at extracting clean article content from noisy web page scrapes.

## YOUR TASK
Given raw scraped content from a news article, extract ONLY the actual journalism:
- The headline/title
- The byline (author name, date)
- The article body paragraphs
- Relevant quotes and attributions

## NOISE TO REMOVE (ignore completely)

### Subscription/Access Noise
- "Subscribe to read more", "Sign in to continue"
- "This article is for subscribers only"
- Paywall messages, premium content notices
- "X articles remaining this month"
- Account/login prompts

### Device/Session Noise
- "You can only read on one device at a time"
- "Reading in progress on another device"
- "Continue reading here"
- Session warnings, device limits
- "Click to continue on this device"

### Navigation Noise
- Menu items, breadcrumbs
- "Back to top", "Skip to content"
- Section headers like "Politics", "Business" (unless part of article)
- "Related articles", "More from this author"
- "Read also", "See also" sections

### Promotional Noise
- Newsletter signup prompts
- "Download our app"
- "Follow us on social media"
- Donation/support requests
- Event promotions

### Interactive Noise
- Comment sections and counts
- Share buttons text
- Like/reaction counts
- "X people are reading this"
- Poll widgets

### Legal/Technical Noise
- Cookie notices
- Privacy policy links
- Terms of service
- Copyright notices at bottom
- "Contact us", "About us"
- Advertising labels

### Repeated/Duplicated Content
- Content that appears multiple times (often from page templates)
- FAQ sections about subscriptions
- Generic "how to read" instructions

## EXTRACTION RULES

1. **Title**: Extract the main headline - usually the most prominent text at the start
2. **Subtitle**: If there's a deck/subtitle under the headline, include it
3. **Author**: Look for "By [Name]" pattern near the top
4. **Date**: Look for publication date near byline
5. **Body**: Extract all substantive paragraphs that form the article narrative

## CONTENT QUALITY CHECKS

- If the body is very short (<200 words) but there are paywall messages, mark `is_truncated: true`
- If content seems incomplete mid-sentence, mark `is_truncated: true`
- Note what types of noise you removed in `noise_removed`

## OUTPUT FORMAT

Return valid JSON with the CleanedArticle structure. The `body` field should contain 
the clean article text with paragraphs separated by double newlines.

Be aggressive about removing noise - if in doubt, leave it out. The goal is clean, 
readable journalism without any web page cruft."""


USER_PROMPT = """Clean this scraped article content. Extract only the actual journalism.

URL: {url}
Domain: {domain}

RAW SCRAPED CONTENT:
---
{content}
---

Extract the clean article and return JSON with:
- title: Main headline
- subtitle: Subtitle/deck if present (null if none)
- author: Author name(s) (null if not found)
- publication_date: Date string as found (null if not found)
- body: Clean article text (paragraphs separated by \\n\\n)
- lead_paragraph: Opening paragraph if distinct (null if not)
- image_captions: List of relevant captions (empty list if none)
- word_count: Word count of body
- cleaning_confidence: 0.0-1.0 confidence score
- noise_removed: List of noise types removed (e.g., ["subscription_prompt", "device_warning", "navigation"])
- is_truncated: true if article appears cut off by paywall
- truncation_reason: Why truncated (null if not truncated)

Return ONLY valid JSON."""


# ============================================================================
# ARTICLE CONTENT CLEANER
# ============================================================================

class ArticleContentCleaner:
    """
    AI-powered article content cleaner.

    Uses GPT-4o-mini to intelligently extract only the actual article content
    from noisy web page scrapes, removing all promotional, navigation, and
    subscription-related noise.
    """

    # Processing limits
    MAX_INPUT_LENGTH = 100000  # Max chars to send to AI (~1M token context)
    MIN_CONTENT_LENGTH = 100  # Min chars to attempt cleaning

    def __init__(self, config=None):
        """
        Initialize the cleaner.

        Args:
            config: Configuration object with API keys
        """
        self.config = config

        # Initialize LLM - Gemini 2.0 Flash: 1M context, fast, cheap
        # NOTE: Using synchronous client due to async issues in langchain-google-genai
        # See: https://github.com/langchain-ai/langchain-google/issues/357
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            max_output_tokens=4096,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            # NOTE: timeout parameter doesn't work reliably, using asyncio.wait_for wrapper instead
        )

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT)
        ])

        # In-memory cache
        self.cache: Dict[str, CleanedArticle] = {}

        fact_logger.logger.info("ArticleContentCleaner initialized (gemini-2.0-flash)")

    async def clean(
        self,
        url: str,
        content: str,
        use_cache: bool = True
    ) -> CleaningResult:
        """
        Clean scraped article content.

        Args:
            url: Article URL (for context)
            content: Raw scraped content
            use_cache: Whether to use cached results

        Returns:
            CleaningResult with cleaned article
        """
        from urllib.parse import urlparse

        # Check cache
        if use_cache and url in self.cache:
            cached = self.cache[url]
            return CleaningResult(
                success=True,
                cleaned=cached,
                original_length=len(content),
                cleaned_length=len(cached.body),
                reduction_percent=self._calc_reduction(len(content), len(cached.body))
            )

        # Validate input
        if not content or len(content) < self.MIN_CONTENT_LENGTH:
            return CleaningResult(
                success=False,
                error="Content too short to clean",
                original_length=len(content) if content else 0
            )

        # Extract domain
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
        except Exception:
            domain = "unknown"

        # Truncate if too long
        content_to_clean = content[:self.MAX_INPUT_LENGTH]

        try:
            fact_logger.logger.info(
                f"Cleaning article from {domain}",
                extra={"url": url, "input_length": len(content)}
            )

            # Run AI cleaning with timeout wrapper
            # Using synchronous invoke() due to ainvoke() issues with ChatGoogleGenerativeAI
            # Wrap in asyncio.to_thread() to make it non-blocking
            # See: https://github.com/langchain-ai/langchain-google/issues/357
            chain = self.prompt | self.llm

            import asyncio

            # Create a function that runs the sync invoke
            def run_sync_invoke():
                return chain.invoke({
                    "url": url,
                    "domain": domain,
                    "content": content_to_clean
                })

            # Run with timeout (60 seconds)
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(run_sync_invoke),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                fact_logger.logger.error(f"[LOG] Gemini cleaning timed out after 60s for {domain}")
                return CleaningResult(
                    success=False,
                    error="AI cleaning timed out",
                    original_length=len(content)
                )

            # Parse response - strip markdown fences if present
            raw_text = result.content.strip()
            raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
            raw_text = re.sub(r'\s*```$', '', raw_text)
            extracted = json.loads(raw_text)

            # Build cleaned article
            cleaned = CleanedArticle(
                title=extracted.get('title'),
                subtitle=extracted.get('subtitle'),
                author=extracted.get('author'),
                publication_date=extracted.get('publication_date'),
                body=extracted.get('body', ''),
                lead_paragraph=extracted.get('lead_paragraph'),
                image_captions=extracted.get('image_captions', []),
                word_count=extracted.get('word_count', 0),
                cleaning_confidence=extracted.get('cleaning_confidence', 0.5),
                noise_removed=extracted.get('noise_removed', []),
                is_truncated=extracted.get('is_truncated', False),
                truncation_reason=extracted.get('truncation_reason')
            )

            # Recalculate word count if not provided
            if cleaned.word_count == 0 and cleaned.body:
                cleaned.word_count = len(cleaned.body.split())

            # Cache result
            self.cache[url] = cleaned

            # Calculate reduction
            cleaned_length = len(cleaned.body)
            reduction = self._calc_reduction(len(content), cleaned_length)

            fact_logger.logger.info(
                f"Cleaned article: {len(content)} -> {cleaned_length} chars ({reduction:.0f}% reduction)",
                extra={
                    "url": url,
                    "original_length": len(content),
                    "cleaned_length": cleaned_length,
                    "word_count": cleaned.word_count,
                    "is_truncated": cleaned.is_truncated,
                    "confidence": cleaned.cleaning_confidence
                }
            )

            return CleaningResult(
                success=True,
                cleaned=cleaned,
                original_length=len(content),
                cleaned_length=cleaned_length,
                reduction_percent=reduction
            )

        except json.JSONDecodeError as e:
            fact_logger.logger.error(f"[LOG] JSON parse error in cleaning: {e}")
            return CleaningResult(
                success=False,
                error=f"Failed to parse AI response: {e}",
                original_length=len(content)
            )
        except Exception as e:
            fact_logger.logger.error(f"[LOG] Article cleaning failed: {e}")
            return CleaningResult(
                success=False,
                error=str(e),
                original_length=len(content)
            )

    def _calc_reduction(self, original: int, cleaned: int) -> float:
        """Calculate percentage reduction"""
        if original == 0:
            return 0.0
        return ((original - cleaned) / original) * 100

    async def clean_batch(
        self,
        articles: Dict[str, str],
        use_cache: bool = True
    ) -> Dict[str, CleaningResult]:
        """
        Clean multiple articles.

        Args:
            articles: Dict mapping URL to raw content
            use_cache: Whether to use cached results

        Returns:
            Dict mapping URL to CleaningResult
        """
        import asyncio

        results = {}

        # Process in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent AI calls

        async def clean_with_semaphore(url: str, content: str):
            async with semaphore:
                return url, await self.clean(url, content, use_cache)

        tasks = [
            clean_with_semaphore(url, content)
            for url, content in articles.items()
        ]

        for coro in asyncio.as_completed(tasks):
            try:
                url, result = await coro
                results[url] = result
            except Exception as e:
                fact_logger.logger.error(f"[LOG] Batch cleaning error: {e}")

        return results


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_article_cleaner(config=None) -> ArticleContentCleaner:
    """Get an article content cleaner instance"""
    return ArticleContentCleaner(config)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    # Sample noisy content (like the Le Monde example)
    test_content = """
    Cet article vous est offert Pour lire gratuitement cet article reserve aux abonnes, connectez-vous Se connecter Vous n'etes pas inscrit sur Le Monde ? Inscrivez-vous gratuitement

    # Plainte contre Thierry Mariani pour -> provocation -> la discrimination au logement ->

    Thierry Mariani, depute europeen du Rassemblement national (RN) et candidat a la Mairie de Paris, devant le marche couvert des Batignolles, dans le 17 arrondissement de Paris, le 11 janvier 2026. KIRAN RIDLEY/AFP

    Thierry Mariani, candidat Rassemblement national (RN) a la Mairie de Paris et depute europeen, est vise par une plainte de l'association La Maison des potes pour " provocation a la discrimination au logement ", en raison de sa promesse de campagne d'instaurer la priorite nationale, a declare vendredi la plaignante a l'Agence France-Presse (AFP).

    L'association estime qu'avec cet argument Thierry Mariani " appelle explicitement tous ceux qui seront candidats sur sa liste " a " l'instauration d'une politique municipale fondee sur un critere de nationalite, lequel est prohibe par la loi ".

    Cette plainte, transmise au parquet de Paris recemment, mentionne le site Internet de la candidature de M. Mariani.

    Lecture du Monde en cours sur un autre appareil.
    Vous pouvez lire Le Monde sur un seul appareil a la fois
    Continuer a lire ici
    Ce message s'affichera sur l'autre appareil.

    Votre abonnement n'autorise pas la lecture de cet article
    Pour plus d'informations, merci de contacter notre service commercial.
    """

    async def test():
        cleaner = ArticleContentCleaner()

        result = await cleaner.clean(
            url="https://www.lemonde.fr/politique/article/2026/01/30/plainte-contre-thierry-mariani",
            content=test_content
        )

        if result.success and result.cleaned:
            print("[LOG] Cleaning successful!")
            print(f"\nTitle: {result.cleaned.title}")
            print(f"Author: {result.cleaned.author}")
            print(f"Date: {result.cleaned.publication_date}")
            print(f"\nBody ({result.cleaned.word_count} words):")
            print(result.cleaned.body[:500] + "..." if len(result.cleaned.body) > 500 else result.cleaned.body)
            print(f"\nNoise removed: {result.cleaned.noise_removed}")
            print(f"Truncated: {result.cleaned.is_truncated}")
            print(f"Confidence: {result.cleaned.cleaning_confidence}")
            print(f"\nReduction: {result.original_length} -> {result.cleaned_length} ({result.reduction_percent:.0f}%)")
        else:
            print(f"[LOG] Failed: {result.error}")

    asyncio.run(test())
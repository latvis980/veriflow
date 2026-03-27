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

STRICT: Never invents, paraphrases, or adds content not present in raw input.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field
import json
import os
import re

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.logger import fact_logger
from prompts.article_content_cleaner_prompts import SYSTEM_PROMPT, USER_PROMPT


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
# ARTICLE CONTENT CLEANER
# ============================================================================

class ArticleContentCleaner:
    """
    AI-powered article content cleaner.

    Uses gpt-4.1-nano to extract only the actual article content from noisy
    web page scrapes, removing all promotional, navigation, and
    subscription-related noise. Never invents or paraphrases content.
    """

    # Processing limits
    MAX_INPUT_LENGTH = 100000  # Max chars to send to AI
    MIN_CONTENT_LENGTH = 100   # Min chars to attempt cleaning

    def __init__(self, config=None):
        """
        Initialize the cleaner.

        Args:
            config: Configuration object with API keys
        """
        self.config = config

        # gpt-4.1-nano: fast, cheap, reliable async support
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0,
            max_tokens=4096,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60,
        )

        # Build prompt template from imported prompts
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT)
        ])

        # In-memory cache
        self.cache: Dict[str, CleanedArticle] = {}

        fact_logger.logger.info("ArticleContentCleaner initialized (gpt-4.1-nano)")

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

            chain = self.prompt | self.llm

            # OpenAI supports ainvoke natively - no thread workaround needed
            import asyncio
            try:
                result = await asyncio.wait_for(
                    chain.ainvoke({
                        "url": url,
                        "domain": domain,
                        "content": content_to_clean
                    }),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                fact_logger.logger.error(f"[LOG] Cleaning timed out after 60s for {domain}")
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
        Clean multiple articles in parallel.

        Args:
            articles: Dict mapping URL to raw content
            use_cache: Whether to use cached results

        Returns:
            Dict mapping URL to CleaningResult
        """
        import asyncio

        results = {}

        # Limit concurrency to avoid rate limits
        semaphore = asyncio.Semaphore(5)

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

    test_content = """
    Cet article vous est offert Pour lire gratuitement cet article reserve aux abonnes, connectez-vous Se connecter Vous n'etes pas inscrit sur Le Monde ? Inscrivez-vous gratuitement

    # Plainte contre Thierry Mariani pour provocation a la discrimination au logement

    Thierry Mariani, depute europeen du Rassemblement national (RN) et candidat a la Mairie de Paris, devant le marche couvert des Batignolles, dans le 17 arrondissement de Paris, le 11 janvier 2026. KIRAN RIDLEY/AFP

    Thierry Mariani, candidat Rassemblement national (RN) a la Mairie de Paris et depute europeen, est vise par une plainte de l'association La Maison des potes pour "provocation a la discrimination au logement", en raison de sa promesse de campagne d'instaurer la priorite nationale, a declare vendredi la plaignante a l'Agence France-Presse (AFP).

    L'association estime qu'avec cet argument Thierry Mariani "appelle explicitement tous ceux qui seront candidats sur sa liste" a "l'instauration d'une politique municipale fondee sur un critere de nationalite, lequel est prohibe par la loi".

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
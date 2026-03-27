# utils/article_content_cleaner.py
"""
Article Content Cleaner
AI-powered extraction of actual article content from noisy scraped web pages.
Returns plain text - metadata extraction is handled by a separate agent.
"""

from typing import Optional, Dict
import asyncio
import os

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from utils.logger import fact_logger
from prompts.article_content_cleaner_prompts import SYSTEM_PROMPT, USER_PROMPT


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class CleanedArticle:
    """Cleaned article - plain text body only"""
    def __init__(self, body: str):
        self.body = body
        self.word_count = len(body.split()) if body else 0


class CleaningResult:
    """Result of article cleaning operation"""
    def __init__(
        self,
        success: bool,
        cleaned: Optional[CleanedArticle] = None,
        original_length: int = 0,
        cleaned_length: int = 0,
        reduction_percent: float = 0.0,
        error: Optional[str] = None
    ):
        self.success = success
        self.cleaned = cleaned
        self.original_length = original_length
        self.cleaned_length = cleaned_length
        self.reduction_percent = reduction_percent
        self.error = error


# ============================================================================
# ARTICLE CONTENT CLEANER
# ============================================================================

class ArticleContentCleaner:
    """
    AI-powered article content cleaner.

    Uses gpt-4.1-nano to extract only the actual article content from noisy
    web page scrapes, removing all promotional, navigation, and
    subscription-related noise. Returns plain text.
    """

    MAX_INPUT_LENGTH = 100000
    MIN_CONTENT_LENGTH = 100

    def __init__(self, config=None):
        self.config = config

        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0,
            max_tokens=4096,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=60,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT)
        ])

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
            CleaningResult with cleaned article body as plain text
        """
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

        content_to_clean = content[:self.MAX_INPUT_LENGTH]

        try:
            fact_logger.logger.info(
                "Cleaning article content",
                extra={"url": url, "input_length": len(content)}
            )

            chain = self.prompt | self.llm

            try:
                result = await asyncio.wait_for(
                    chain.ainvoke({
                        "url": url,
                        "content": content_to_clean
                    }),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                fact_logger.logger.error(f"[LOG] Cleaning timed out after 60s for {url}")
                return CleaningResult(
                    success=False,
                    error="AI cleaning timed out",
                    original_length=len(content)
                )

            body = result.content.strip()
            cleaned = CleanedArticle(body=body)
            self.cache[url] = cleaned

            cleaned_length = len(cleaned.body)
            reduction = self._calc_reduction(len(content), cleaned_length)

            fact_logger.logger.info(
                f"Cleaned article: {len(content)} -> {cleaned_length} chars ({reduction:.0f}% reduction)",
                extra={
                    "url": url,
                    "original_length": len(content),
                    "cleaned_length": cleaned_length,
                    "word_count": cleaned.word_count,
                }
            )

            return CleaningResult(
                success=True,
                cleaned=cleaned,
                original_length=len(content),
                cleaned_length=cleaned_length,
                reduction_percent=reduction
            )

        except Exception as e:
            fact_logger.logger.error(f"[LOG] Article cleaning failed: {e}")
            return CleaningResult(
                success=False,
                error=str(e),
                original_length=len(content)
            )

    def _calc_reduction(self, original: int, cleaned: int) -> float:
        if original == 0:
            return 0.0
        return ((original - cleaned) / original) * 100

    async def clean_batch(
        self,
        articles: Dict[str, str],
        use_cache: bool = True
    ) -> Dict[str, CleaningResult]:
        """Clean multiple articles in parallel."""
        results = {}
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
    return ArticleContentCleaner(config)
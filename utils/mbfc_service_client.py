# utils/mbfc_service_client.py
"""
Client for the MBFC scraper service's HTTP API.
Used by VeriFlow to offload MBFC lookups to the dedicated service
instead of doing them inline (which blocks the main pipeline).

Usage:
    from utils.mbfc_service_client import request_mbfc_lookup

    # Fire-and-forget: returns immediately, MBFC service scrapes in background
    await request_mbfc_lookup(domain="lemonde.fr")

    # With known MBFC URL (skips Brave search on the MBFC service side):
    await request_mbfc_lookup(
        domain="lemonde.fr",
        mbfc_url="https://mediabiasfactcheck.com/le-monde-bias/"
    )

Environment variables:
    MBFC_SERVICE_URL    - Base URL of the MBFC service
                          e.g. http://mbfc-scraper.railway.internal:8080
                          or   https://your-mbfc-service.up.railway.app
    MBFC_API_SECRET     - Shared secret for authentication (must match MBFC service)
"""

import os
import asyncio
from typing import Optional

import httpx

from utils.logger import fact_logger


MBFC_SERVICE_URL = os.getenv("MBFC_SERVICE_URL", "").rstrip("/")
MBFC_API_SECRET = os.getenv("MBFC_API_SECRET", "")

# Track domains we've already requested in this process lifetime
# to avoid spamming the MBFC service with duplicate requests
_requested_domains: set = set()


def is_mbfc_service_configured() -> bool:
    """Check if the MBFC service URL is configured."""
    return bool(MBFC_SERVICE_URL)


async def request_mbfc_lookup(
    domain: str,
    mbfc_url: Optional[str] = None,
    fire_and_forget: bool = True,
) -> Optional[dict]:
    """
    Request the MBFC service to look up a domain.

    Args:
        domain: Publication domain (e.g. "lemonde.fr")
        mbfc_url: Optional direct MBFC URL to skip Brave search
        fire_and_forget: If True (default), don't wait for the result.
                         If False, wait and return the response.

    Returns:
        None if fire_and_forget=True
        Response dict if fire_and_forget=False
    """
    if not MBFC_SERVICE_URL:
        fact_logger.logger.debug(
            "MBFC service not configured (MBFC_SERVICE_URL not set)"
        )
        return None

    # Deduplicate: don't request the same domain twice in one process run
    domain_lower = domain.lower()
    if domain_lower in _requested_domains:
        fact_logger.logger.debug(
            f"MBFC lookup already requested for {domain_lower}, skipping"
        )
        return None
    _requested_domains.add(domain_lower)

    url = f"{MBFC_SERVICE_URL}/api/lookup"
    payload = {"domain": domain_lower}
    if mbfc_url:
        payload["mbfc_url"] = mbfc_url

    headers = {"Content-Type": "application/json"}
    if MBFC_API_SECRET:
        headers["Authorization"] = f"Bearer {MBFC_API_SECRET}"

    if fire_and_forget:
        # Launch in background, don't block the caller
        asyncio.create_task(
            _send_lookup_request(url, payload, headers, domain_lower)
        )
        fact_logger.logger.info(
            f"Fired background MBFC lookup for {domain_lower}"
        )
        return None
    else:
        return await _send_lookup_request(url, payload, headers, domain_lower)


async def _send_lookup_request(
    url: str,
    payload: dict,
    headers: dict,
    domain: str,
) -> Optional[dict]:
    """Send the HTTP request to the MBFC service."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code == 409:
                # Lookup already in progress on the MBFC service side
                fact_logger.logger.debug(
                    f"MBFC service: lookup already in progress for {domain}"
                )
                return None

            if resp.status_code == 200:
                data = resp.json()
                if data.get("success"):
                    fact_logger.logger.info(
                        f"MBFC service: saved {domain} "
                        f"(tier {data.get('tier', '?')})"
                    )
                else:
                    fact_logger.logger.info(
                        f"MBFC service: no data for {domain} - "
                        f"{data.get('message', 'unknown reason')}"
                    )
                return data
            else:
                fact_logger.logger.warning(
                    f"MBFC service returned {resp.status_code} for {domain}"
                )
                return None

    except httpx.ConnectError:
        fact_logger.logger.warning(
            f"Cannot reach MBFC service at {url} - is it running?"
        )
        return None
    except Exception as e:
        fact_logger.logger.warning(
            f"MBFC service request failed for {domain}: {e}"
        )
        return None


def clear_requested_cache():
    """Clear the dedup cache. Useful for testing or long-running processes."""
    _requested_domains.clear()

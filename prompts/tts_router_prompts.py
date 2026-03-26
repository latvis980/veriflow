# prompts/tts_router_prompts.py
"""
Prompts for the TTS (The True Story) Router Agent

Determines whether extracted claims/facts should be checked against
The True Story news aggregation platform before falling back to web search.

TTS covers: current news events (last ~6 months), with strong coverage of:
- International politics and diplomacy
- Russia/Ukraine conflict
- Israel/Iran/Middle East conflicts
- US politics and foreign policy
- European politics (EU, NATO, individual countries)
- Major global economic events
- War and military operations
- Major social/protest movements

TTS does NOT cover well:
- Local/municipal news (unless internationally significant)
- Entertainment, celebrity gossip
- Sports (limited coverage, mostly major events)
- Science/technology (unless politically significant)
- Historical facts (before ~6 months ago)
- Personal advice, recipes, tutorials
- Product reviews, business-specific claims
- Medical/health claims (unless pandemic/policy-level)
"""

SYSTEM_PROMPT = """You are a routing classifier for a fact-checking system.
Your job is to decide which claims from the input should be checked against
a news aggregation database called The True Story (TTS).

TTS is a news aggregation platform that clusters articles from hundreds of
media outlets about the same news events. It covers the last 6 months of
world news with editions in Russian and English.

STRONG COVERAGE (high chance of finding relevant clusters):
- International politics, diplomacy, treaties, sanctions
- Russia-Ukraine conflict (all aspects: military, diplomatic, humanitarian)
- Israel-Iran-Middle East conflicts and politics
- US presidential actions, executive orders, foreign policy
- European politics: EU decisions, NATO, elections in major countries
- Major global economic events: trade wars, oil prices, BRICS, G7/G20
- War, military operations, ceasefire negotiations
- Large-scale protests, regime changes, coups
- UN, ICC, international law events

MODERATE COVERAGE:
- Major terrorist attacks or mass casualty events
- Global pandemic or health emergency news
- Climate/environmental policy at international level
- Major sanctions and their effects
- Elections in medium-sized countries

WEAK OR NO COVERAGE (skip TTS, go straight to web search):
- Local or municipal news
- Entertainment, celebrities, movies, music
- Sports results and transfers
- Science and technology discoveries
- Historical facts (events before September 2025)
- Personal finance, investment advice
- Product reviews, company-specific business news
- Medical claims, drug information
- Recipes, tutorials, how-to content
- Academic or theoretical claims
- Statements about the future (predictions without news backing)

For each claim, decide:
- "tts" = check against TTS first (likely to find matching news clusters)
- "skip" = skip TTS, use web search directly (TTS unlikely to have this)

Also generate a compact search query for claims routed to TTS.
The query should be 2-5 keywords that would match news article titles.
Use the original language of the claim for the query (Russian claims get
Russian keywords, English claims get English keywords).

Return ONLY valid JSON. No other text."""


USER_PROMPT = """Analyze these claims and decide which should be checked against
The True Story news database.

CLAIMS TO ROUTE:
{claims_text}

CONTENT LANGUAGE: {language}
CONTENT REALM: {realm}

For each claim, return:
- claim_id: the claim identifier
- route: "tts" or "skip"
- reason: brief explanation (5-10 words)
- tts_query: search keywords if route is "tts", null if "skip"
- tts_edition: "ru" or "en" based on claim language/region
- confidence: 0.0-1.0 how confident you are in the routing decision

Return valid JSON:
{{
  "routing_decisions": [
    {{
      "claim_id": "fact1",
      "route": "tts",
      "reason": "recent political event with known actors",
      "tts_query": "Macron Ukraine negotiations Kremlin",
      "tts_edition": "en",
      "confidence": 0.9
    }},
    {{
      "claim_id": "fact2",
      "route": "skip",
      "reason": "historical fact, not recent news",
      "tts_query": null,
      "tts_edition": null,
      "confidence": 0.95
    }}
  ]
}}"""


def get_tts_router_prompts() -> dict:
    """Return system and user prompts for the TTS router"""
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT,
    }

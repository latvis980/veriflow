# test_tts.py
"""
Standalone test for TTS integration.
Run this directly to check each component:
  python test_tts.py

Tests:
  1. TTSRouter initialization (LLM availability)
  2. TTSService.search (esearch API connectivity)
  3. TTSRouter.route (full routing pipeline)
  4. TTSService.find_evidence_for_claim (end-to-end)
"""

import asyncio
import os
import sys
import time


async def test_tts_service():
    """Test 1: Can we reach the TTS search API?"""
    print("\n" + "=" * 60)
    print("TEST 1: TTSService.search")
    print("=" * 60)

    try:
        from utils.tts_service import TTSService
        service = TTSService()
        print("[OK] TTSService created")

        # Search for something we know TTS has
        print("[..] Searching TTS for 'Trump Iran' (en edition)...")
        start = time.time()
        results = await service.search("Trump Iran", edition="en", grouping=True)
        elapsed = time.time() - start

        print(f"[OK] Search completed in {elapsed:.2f}s")
        print(f"     Total results: {results.total}")
        print(f"     Hits returned: {len(results.hits)}")
        print(f"     Took (elastic): {results.took_ms}ms")

        if results.hits:
            top = results.hits[0]
            print(f"     Top hit: score={top.score:.2f}")
            print(f"       cluster_id: {top.cluster_id}")
            print(f"       cluster_size: {top.cluster_size}")
            print(f"       source: {top.source_title}")
            print(f"       title: {top.clean_title[:80]}")
        else:
            print("[WARN] No hits returned - TTS may be unreachable")

        await service.close()
        return True

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_tts_service_ru():
    """Test 1b: Russian edition"""
    print("\n" + "=" * 60)
    print("TEST 1b: TTSService.search (ru edition)")
    print("=" * 60)

    try:
        from utils.tts_service import TTSService
        service = TTSService()

        print("[..] Searching TTS for 'Украина переговоры' (ru edition)...")
        start = time.time()
        results = await service.search("Украина переговоры", edition="ru", grouping=True)
        elapsed = time.time() - start

        print(f"[OK] Search completed in {elapsed:.2f}s")
        print(f"     Total results: {results.total}")
        print(f"     Hits returned: {len(results.hits)}")

        if results.hits:
            top = results.hits[0]
            print(f"     Top hit: cluster_size={top.cluster_size}, score={top.score:.2f}")
            print(f"       title: {top.clean_title[:80]}")

        await service.close()
        return True

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        return False


async def test_tts_router_init():
    """Test 2: Can the TTSRouter initialize an LLM?"""
    print("\n" + "=" * 60)
    print("TEST 2: TTSRouter initialization")
    print("=" * 60)

    # Check env vars
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"  GOOGLE_API_KEY: {'set (' + gemini_key[:10] + '...)' if gemini_key else 'NOT SET'}")
    print(f"  OPENAI_API_KEY: {'set (' + openai_key[:10] + '...)' if openai_key else 'NOT SET'}")

    # Check langchain-google-genai
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("  langchain-google-genai: installed")
    except ImportError:
        print("  langchain-google-genai: NOT INSTALLED")

    try:
        from agents.tts_router import TTSRouter
        print("[..] Creating TTSRouter...")
        router = TTSRouter()
        print(f"[OK] TTSRouter created, LLM type: {type(router.llm).__name__}")
        return True, router

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_tts_router_route(router):
    """Test 3: Can the router route claims?"""
    print("\n" + "=" * 60)
    print("TEST 3: TTSRouter.route")
    print("=" * 60)

    test_claims = [
        {
            "id": "claim1",
            "statement": "Trump ordered a five-day pause on strikes against Iranian energy infrastructure"
        },
        {
            "id": "claim2",
            "statement": "The Eiffel Tower is 330 meters tall"
        },
        {
            "id": "claim3",
            "statement": "Russia rejected European participation in Ukraine peace negotiations"
        },
    ]

    try:
        print(f"[..] Routing {len(test_claims)} claims...")
        start = time.time()

        decisions = await router.route(
            claims=test_claims,
            content_language="english",
            content_realm="political",
        )

        elapsed = time.time() - start
        print(f"[OK] Routing completed in {elapsed:.2f}s")
        print(f"     Decisions: {len(decisions)}")

        for d in decisions:
            print(f"     [{d.claim_id}] -> {d.route} (conf={d.confidence:.2f})")
            if d.tts_query:
                print(f"       query: '{d.tts_query}' ({d.tts_edition})")
            print(f"       reason: {d.reason}")

        tts_count = sum(1 for d in decisions if d.route == "tts")
        skip_count = sum(1 for d in decisions if d.route == "skip")
        print(f"\n     Summary: {tts_count} -> TTS, {skip_count} -> skip")

        # Sanity check: claim2 (Eiffel Tower) should be skipped
        claim2_decision = next((d for d in decisions if d.claim_id == "claim2"), None)
        if claim2_decision and claim2_decision.route == "skip":
            print("[OK] Eiffel Tower correctly skipped (not news)")
        elif claim2_decision:
            print(f"[WARN] Eiffel Tower routed to {claim2_decision.route} (expected skip)")

        return True

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_end_to_end():
    """Test 4: Full find_evidence_for_claim flow"""
    print("\n" + "=" * 60)
    print("TEST 4: End-to-end find_evidence_for_claim")
    print("=" * 60)

    try:
        from utils.tts_service import TTSService
        service = TTSService()

        print("[..] Searching for evidence: 'Trump Iran negotiations pause strikes'...")
        start = time.time()

        evidence = await service.find_evidence_for_claim(
            query="Trump Iran negotiations pause strikes",
            edition="en",
            min_cluster_size=3,
        )

        elapsed = time.time() - start

        if evidence:
            print(f"[OK] Evidence found in {elapsed:.2f}s")
            print(f"     Cluster: {evidence.get('cluster_title', '')[:80]}")
            print(f"     Sources: {evidence.get('source_count', 0)}")
            print(f"     Score: {evidence.get('search_score', 0):.2f}")
            print(f"     Evidence texts: {len(evidence.get('evidence_texts', []))}")
            print(f"     Story sources: {len(evidence.get('story_sources', []))}")

            # Show first evidence text
            texts = evidence.get("evidence_texts", [])
            if texts:
                first = texts[0]
                print(f"\n     First evidence ({first.get('source', 'unknown')}):")
                text = first.get("text", "")[:200]
                print(f"       {text}...")
        else:
            print(f"[WARN] No evidence found in {elapsed:.2f}s")
            print("       This could mean TTS search returned no results")

        await service.close()
        return True

    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("  TTS Integration Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Service search
    results["service_en"] = await test_tts_service()
    results["service_ru"] = await test_tts_service_ru()

    # Test 2: Router init
    router_ok, router = await test_tts_router_init()
    results["router_init"] = router_ok

    # Test 3: Router route
    if router:
        results["router_route"] = await test_tts_router_route(router)
    else:
        results["router_route"] = False
        print("\n[SKIP] TEST 3: Router not initialized")

    # Test 4: End-to-end
    results["end_to_end"] = await test_end_to_end()

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")

    all_pass = all(results.values())
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

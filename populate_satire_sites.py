# populate_satire_mbfc.py
"""
Populate satire_parody_sites table with sites from MBFC's Satire category.

These sites were listed on mediabiasfactcheck.com/satire/ but could not be
bulk-scraped because the satire category page has a different structure.
Instead, the list was exported to Excel and parsed here.

Uses upsert on 'domain' so it's safe to re-run -- existing records from
the earlier populate_satire_sites.py script won't be duplicated.

Prerequisites:
  - SUPABASE_URL and SUPABASE_KEY env vars set
  - satire_parody_sites table already exists (created by original script)

Usage:
  python populate_satire_mbfc.py              # Insert all
  python populate_satire_mbfc.py --dry-run    # Preview without writing
"""

import os
import sys
import argparse
from typing import List, Dict

from supabase import create_client, Client


# ===========================================
# DATA: MBFC Satire category sites
# Parsed from MBFC satire listing page export
# ===========================================

MBFC_SATIRE_SITES = [
    {"domain": "adobochronicles.com", "name": "Adobo Chronicles", "category": "Satire"},
    {"domain": "alternativelyfacts.com", "name": "Alternative Facts", "category": "Satire"},
    {"domain": "alternative-science.com", "name": "Alternative Science", "category": "Satire"},
    {"domain": "babylonbee.com", "name": "Babylon Bee", "category": "Satire"},
    {"domain": "burrardstreetjournal.com", "name": "Burrard Street Journal", "category": "Satire"},
    {"domain": "clickhole.com", "name": "ClickHole", "category": "Satire"},
    {"domain": "confederacyofdrones.com", "name": "Confederacy of Drones", "category": "Satire"},
    {"domain": "cracked.com", "name": "Cracked", "category": "Satire"},
    {"domain": "dailysquib.co.uk", "name": "Daily Squib", "category": "Satire", "country": "UK"},
    {"domain": "dailyworldupdate.us", "name": "Daily World Update", "category": "Satire"},
    {"domain": "dailysnark.com", "name": "DailySnark", "category": "Satire"},
    {"domain": "delawareohionews.com", "name": "Delaware Ohio News", "category": "Satire"},
    {"domain": "speld.nl", "name": "De Speld", "category": "Satire", "country": "Netherlands", "language": "Dutch"},
    {"domain": "dietagespresse.com", "name": "Die Tagespresse", "category": "Satire", "country": "Austria", "language": "German"},
    {"domain": "dnatured.com", "name": "DNAtured", "category": "Satire"},
    {"domain": "duffelblog.com", "name": "Duffel Blog", "category": "Satire"},
    {"domain": "dunning-kruger-times.com", "name": "Dunning-Kruger Times", "category": "Satire"},
    {"domain": "empirenews.net", "name": "Empire News", "category": "Satire"},
    {"domain": "empiresports.co", "name": "Empire Sports News", "category": "Satire"},
    {"domain": "fark.com", "name": "FARK", "category": "Satire"},
    {"domain": "flakenews.com", "name": "Flake News", "category": "Satire"},
    {"domain": "fmobserver.com", "name": "FM Observer", "category": "Satire"},
    {"domain": "frankmag.ca", "name": "Frank Magazine", "category": "Satire", "country": "Canada"},
    {"domain": "freedomcrossroads.com", "name": "Freedom Crossroads", "category": "Satire"},
    {"domain": "genesiustimes.com", "name": "Genesius Times", "category": "Satire"},
    {"domain": "gomerblog.com", "name": "Gomer Blog", "category": "Satire"},
    {"domain": "humortimes.com", "name": "Humor Times", "category": "Satire"},
    {"domain": "imao.us", "name": "IMAO (In My Arrogant Opinion)", "category": "Satire"},
    {"domain": "infobattle.org", "name": "InfoBattle", "category": "Satire"},
    {"domain": "chronicle.su", "name": "Internet Chronicle", "category": "Satire"},
    {"domain": "legorafi.fr", "name": "Le Gorafi", "category": "Satire", "country": "France", "language": "French"},
    {"domain": "madhousemagazine.com", "name": "Madhouse Magazine", "category": "Satire"},
    {"domain": "mcsweeneys.net", "name": "McSweeney's Internet Tendency", "category": "Satire"},
    {"domain": "moronmajority.com", "name": "MoronMajority", "category": "Satire"},
    {"domain": "newsbiscuit.com", "name": "NewsBiscuit", "category": "Satire", "country": "UK"},
    {"domain": "newsthump.com", "name": "NewsThump", "category": "Satire", "country": "UK"},
    {"domain": "neutralgroundnews.com", "name": "Neutral Ground News", "category": "Satire"},
    {"domain": "prettycoolsite.com", "name": "Pretty Cool Site", "category": "Satire"},
    {"domain": "realnewsrightnow.com", "name": "Real News Right Now", "category": "Satire"},
    {"domain": "realrawnews.com", "name": "Real Raw News", "category": "Satire / Fake News"},
    {"domain": "reductress.com", "name": "Reductress", "category": "Satire"},
    {"domain": "robotbutt.com", "name": "Robot Butt", "category": "Satire"},
    {"domain": "satirev.org", "name": "Satire V", "category": "Satire"},
    {"domain": "southdakotatruth.com", "name": "South Dakota Truth", "category": "Satire"},
    {"domain": "southendnewsnetwork.net", "name": "SouthEnd News Network", "category": "Satire"},
    {"domain": "spacexmania.com", "name": "SpaceXMania", "category": "Satire"},
    {"domain": "sportspickle.com", "name": "Sports Pickle", "category": "Satire"},
    {"domain": "stiltonsplace.blogspot.com", "name": "Stilton's Place", "category": "Satire"},
    {"domain": "stubhillnews.com", "name": "Stubhill News", "category": "Satire"},
    {"domain": "suffolkgazette.com", "name": "Suffolk Gazette", "category": "Satire", "country": "UK"},
    {"domain": "sundaysportonline.co.uk", "name": "Sunday Sport", "category": "Satire", "country": "UK"},
    {"domain": "takomatorch.com", "name": "Takoma Torch", "category": "Satire"},
    {"domain": "thebeaverton.com", "name": "The Beaverton", "category": "Satire", "country": "Canada"},
    {"domain": "betootaadvocate.com", "name": "The Betoota Advocate", "category": "Satire", "country": "Australia"},
    {"domain": "chaser.com.au", "name": "The Chaser", "category": "Satire", "country": "Australia"},
    {"domain": "dailydiscord.com", "name": "The Daily Discord", "category": "Satire"},
    {"domain": "thedailymash.co.uk", "name": "The Daily Mash", "category": "Satire", "country": "UK"},
    {"domain": "dailyskrape.com", "name": "The Daily Skrape", "category": "Satire"},
    {"domain": "thedailywasp.com", "name": "The Daily Wasp", "category": "Satire"},
    {"domain": "halfwaypost.com", "name": "The Halfway Post", "category": "Satire"},
    {"domain": "thehardtimes.net", "name": "The Hard Times", "category": "Satire"},
    {"domain": "thejuicemedia.com", "name": "The Juice Media", "category": "Satire", "country": "Australia"},
    {"domain": "themideastbeast.com", "name": "The Mideast Beast", "category": "Satire"},
    {"domain": "theneedling.com", "name": "The Needling", "category": "Satire"},
    {"domain": "thenib.com", "name": "The Nib", "category": "Satire"},
    {"domain": "theonion.com", "name": "The Onion", "category": "Satire"},
    {"domain": "thepeoplescube.com", "name": "The People's Cube", "category": "Satire"},
    {"domain": "thepoke.com", "name": "The Poke", "category": "Satire", "country": "UK"},
    {"domain": "politicalgarbagechute.com", "name": "The Political Garbage Chute", "category": "Satire"},
    {"domain": "the-postillon.com", "name": "The Postillon", "category": "Satire", "country": "Germany", "language": "English/German"},
    {"domain": "theredshtick.com", "name": "The Red Shtick", "category": "Satire"},
    {"domain": "thesciencepost.com", "name": "The Science Post", "category": "Satire"},
    {"domain": "theshovel.com.au", "name": "The Shovel", "category": "Satire", "country": "Australia"},
    {"domain": "thespoof.com", "name": "The Spoof", "category": "Satire"},
    {"domain": "waterfordwhispersnews.com", "name": "Waterford Whispers News", "category": "Satire", "country": "Ireland"},
    {"domain": "weeklyworldnews.com", "name": "Weekly World News", "category": "Satire"},
    {"domain": "zaytung.com", "name": "Zaytung", "category": "Satire", "country": "Turkey", "language": "Turkish"},
]


def get_supabase_client() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("[ERROR] SUPABASE_URL and SUPABASE_KEY must be set.")
        sys.exit(1)
    return create_client(supabase_url, supabase_key)


def populate(client: Client, data: List[Dict], dry_run: bool = False) -> Dict:
    results = {"success": 0, "failed": 0, "errors": []}

    print(f"\nInserting {len(data)} MBFC satire sites into satire_parody_sites...\n")

    # Batch upsert in chunks of 50 for efficiency
    batch_size = 50
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]

        if dry_run:
            for site in chunk:
                print(f"  [DRY RUN] {site['name']:45s} ({site['domain']})")
                results["success"] += 1
            continue

        try:
            response = client.table("satire_parody_sites").upsert(
                chunk, on_conflict="domain"
            ).execute()

            if response.data:
                for site in chunk:
                    print(f"  OK  {site['name']:45s} ({site['domain']})")
                    results["success"] += 1
            else:
                for site in chunk:
                    results["failed"] += 1
                    results["errors"].append(f"{site['domain']}: no data returned")
                    print(f"  WARN {site['name']} - no data returned")

        except Exception as e:
            # Fall back to one-by-one on batch failure
            print(f"  Batch insert failed ({e}), trying one by one...")
            for site in chunk:
                try:
                    resp = client.table("satire_parody_sites").upsert(
                        site, on_conflict="domain"
                    ).execute()
                    if resp.data:
                        print(f"  OK  {site['name']:45s} ({site['domain']})")
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                        print(f"  WARN {site['name']} - no data returned")
                except Exception as e2:
                    results["failed"] += 1
                    results["errors"].append(f"{site['domain']}: {e2}")
                    print(f"  FAIL {site['name']} - {e2}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Populate satire_parody_sites with MBFC satire list"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    print("=" * 60)
    print("  MBFC Satire Sites -> satire_parody_sites")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] No changes will be written.\n")
        # Still need client to verify table exists
        try:
            client = get_supabase_client()
        except SystemExit:
            # In dry run, allow running without Supabase
            client = None
    else:
        client = get_supabase_client()
        print("[INFO] Supabase connected.\n")

    if client is None and args.dry_run:
        # Pure preview mode
        for site in MBFC_SATIRE_SITES:
            print(f"  [DRY RUN] {site['name']:45s} ({site['domain']})")
        print(f"\nTotal: {len(MBFC_SATIRE_SITES)} sites")
        return

    results = populate(client, MBFC_SATIRE_SITES, dry_run=args.dry_run)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")

    if results["errors"]:
        print("\n  Errors:")
        for err in results["errors"]:
            print(f"    - {err}")

    print("\nDone.")


if __name__ == "__main__":
    main()

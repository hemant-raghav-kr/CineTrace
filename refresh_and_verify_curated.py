import time
from discovery_service import DiscoveryService, CURATED_POPULAR_PICKS

print(f"Total curated picks configured: {len(CURATED_POPULAR_PICKS)}", flush=True)

service = DiscoveryService()
t0 = time.time()
print("Fetching and verifying all curated popular picks with force_refresh=True...", flush=True)
data = service.fetch_home_data(force_refresh=True)
t1 = time.time()

featured = data.get("featured", {})
picks = data.get("picks", [])

print(f"\nCompleted in {t1 - t0:.2f}s", flush=True)
print(f"Featured: '{featured.get('title')}' ({featured.get('year')}) | Poster: {bool(featured.get('poster_url'))}", flush=True)
print(f"Total picks count: {len(picks)}", flush=True)

indian_count = 0
for idx, p in enumerate(picks, 1):
    lang = p.get('language') or 'Unknown'
    title = p.get('title')
    year = p.get('year')
    rating = p.get('imdb_rating')
    genre = p.get('genre')
    is_indian = any(l in lang for l in ('Hindi', 'Tamil', 'Malayalam', 'Telugu'))
    if is_indian:
        indian_count += 1
    flag = "[INDIAN]" if is_indian else "[INTL]  "
    print(f"  {idx:02d}. {flag} {title[:28]:28} ({year}) | Rating: {rating} | Genre: {genre[:25]}", flush=True)

print(f"\nVerified Summary: {len(picks)} total picks ({len(picks) - indian_count} International + {indian_count} Indian Cinema)", flush=True)
assert len(picks) in (30, 35), f"Expected 30 or 35 picks, got {len(picks)}"
assert indian_count in (15, 20), f"Expected 15 or 20 Indian cinema titles, got {indian_count}"
print(">>> CURATED COLLECTION VERIFICATION PASSED <<<", flush=True)

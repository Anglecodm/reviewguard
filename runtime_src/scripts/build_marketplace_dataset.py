from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from reviewguard.keywords import load_keywords
from reviewguard.scrape import ScrapedReview, scrape_reviews


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

JUMIA_PRODUCT_RE = re.compile(r"^https://www\.jumia\.co\.ke/.+\.html$", re.IGNORECASE)
KILIMALL_PRODUCT_RE = re.compile(r"^https://www\.kilimall\.co\.ke/listing/[^?#]+$", re.IGNORECASE)


def fetch_html(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def extract_links(html: str, base_url: str) -> list[str]:
    raw = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    out: list[str] = []
    for link in raw:
        if link.startswith("#") or link.startswith("javascript:"):
            continue
        abs_link = urljoin(base_url, link)
        if not abs_link.startswith("http"):
            continue
        out.append(abs_link.split("#", 1)[0])
    return list(dict.fromkeys(out))


def discover_jumia_products(max_urls: int = 120, max_pages_per_category: int = 3) -> list[str]:
    categories = [
        "https://www.jumia.co.ke/phones-tablets/",
        "https://www.jumia.co.ke/electronics/",
        "https://www.jumia.co.ke/home-office-appliances/",
        "https://www.jumia.co.ke/health-beauty/",
        "https://www.jumia.co.ke/home-office/",
        "https://www.jumia.co.ke/category-fashion-by-jumia/",
        "https://www.jumia.co.ke/computing/",
        "https://www.jumia.co.ke/video-games/",
        "https://www.jumia.co.ke/groceries/",
        "https://www.jumia.co.ke/baby-products/",
    ]

    found: set[str] = set()
    for category in categories:
        for page in range(1, max_pages_per_category + 1):
            page_url = category if page == 1 else f"{category}?page={page}"
            try:
                html = fetch_html(page_url, timeout=35)
                links = extract_links(html, page_url)
            except Exception:
                continue
            for link in links:
                if not JUMIA_PRODUCT_RE.match(link):
                    continue
                if "/assets_he/" in link or "/sp-" in link:
                    continue
                found.add(link)
                if len(found) >= max_urls:
                    return sorted(found)
            time.sleep(0.2)
    return sorted(found)


def discover_kilimall_products(max_urls: int = 120) -> list[str]:
    search_queries = [
        "smart phone",
        "laptops",
        "home appliances",
        "television",
        "earbuds",
        "watch",
        "makeup",
        "baby",
        "fashion",
        "kitchen",
        "fridge",
        "headphones",
        "power bank",
        "speaker",
        "shoes",
    ]
    seed_urls = ["https://www.kilimall.co.ke/"]
    seed_urls.extend([f"https://www.kilimall.co.ke/search?q={q.replace(' ', '+')}" for q in search_queries])

    found: set[str] = set()
    for url in seed_urls:
        try:
            html = fetch_html(url, timeout=35)
            links = extract_links(html, url)
        except Exception:
            continue
        for link in links:
            if KILIMALL_PRODUCT_RE.match(link):
                found.add(link)
                if len(found) >= max_urls:
                    return sorted(found)
        time.sleep(0.2)
    return sorted(found)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text


def collect_reviews(
    product_urls: list[str],
    platform: str,
    per_product_limit: int,
    max_reviews: int,
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    seen_text: set[str] = set()

    for idx, url in enumerate(product_urls, 1):
        try:
            scraped = scrape_reviews(url, limit=per_product_limit)
        except Exception:
            continue

        for item in scraped:
            text = normalize_text(item.text)
            if len(text) < 24:
                continue
            key = text.lower()
            if key in seen_text:
                continue
            seen_text.add(key)
            rows.append(
                {
                    "text": text,
                    "label": 0,
                    "platform": platform,
                    "source_url": url,
                    "user": item.user or "Anonymous",
                    "date": item.date or "",
                    "rating": item.rating or "",
                    "label_source": "scraped_review",
                }
            )
            if len(rows) >= max_reviews:
                return rows

        if idx % 10 == 0:
            print(f"[{platform}] scraped {idx}/{len(product_urls)} products, reviews={len(rows)}")
    return rows


def generate_synthetic_fraud(target_count: int, keywords: list[str], seed: int = 42) -> list[dict[str, str | int]]:
    rng = random.Random(seed)
    items = [
        "phone",
        "watch",
        "earbuds",
        "blender",
        "laptop",
        "speaker",
        "kettle",
        "fridge",
        "shoe",
        "bag",
    ]
    templates = [
        "AMAZING {item}!!! {kw1} and {kw2}! Buy now and thank me later!!!",
        "Best {item} ever, {kw1}. {kw2}. Don't miss this deal!!!",
        "Perfect {item} 10/10!!! {kw1}, {kw2}, and super fast delivery!!!",
        "This {item} changed my life!!! {kw1}. {kw2}. Order today!!!",
        "Unbeatable {item}! {kw1}! {kw2}! Five stars!!!",
    ]

    out: list[dict[str, str | int]] = []
    seen: set[str] = set()
    safe_keywords = [k for k in keywords if " " in k or len(k) > 5]
    if len(safe_keywords) < 10:
        safe_keywords = keywords

    while len(out) < target_count:
        kw1, kw2 = rng.sample(safe_keywords, 2)
        item = rng.choice(items)
        text = rng.choice(templates).format(item=item, kw1=kw1, kw2=kw2)
        text = normalize_text(text)
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "text": text,
                "label": 1,
                "platform": "synthetic",
                "source_url": "",
                "user": "synthetic_generator",
                "date": "",
                "rating": "",
                "label_source": "template_fraud",
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["text", "label", "platform", "source_url", "user", "date", "rating", "label_source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_urls(path: Path, urls: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["platform", "product_url"])
        writer.writerows(urls)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 300+ review CSV from Jumia and Kilimall.")
    parser.add_argument("--out-csv", default="data/marketplace_reviews_weak_labeled.csv")
    parser.add_argument("--out-urls", default="data/discovered_product_urls.csv")
    parser.add_argument("--jumia-products", type=int, default=100)
    parser.add_argument("--kilimall-products", type=int, default=100)
    parser.add_argument("--per-product-reviews", type=int, default=8)
    parser.add_argument("--max-authentic", type=int, default=260)
    parser.add_argument("--min-total", type=int, default=360)
    parser.add_argument(
        "--target-fraud",
        type=int,
        default=120,
        help="Base number of synthetic fraud rows. Increased only if needed to satisfy --min-total.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Discovering product URLs...")
    jumia_urls = discover_jumia_products(max_urls=args.jumia_products)
    kilimall_urls = discover_kilimall_products(max_urls=args.kilimall_products)
    all_urls: list[tuple[str, str]] = [("jumia", u) for u in jumia_urls] + [("kilimall", u) for u in kilimall_urls]
    print(f"Discovered URLs -> jumia={len(jumia_urls)} kilimall={len(kilimall_urls)}")
    write_urls(Path(args.out_urls), all_urls)

    print("Scraping Jumia reviews...")
    jumia_rows = collect_reviews(
        jumia_urls,
        platform="jumia",
        per_product_limit=args.per_product_reviews,
        max_reviews=args.max_authentic,
    )
    print("Scraping Kilimall reviews...")
    kilimall_rows = collect_reviews(
        kilimall_urls,
        platform="kilimall",
        per_product_limit=args.per_product_reviews,
        max_reviews=args.max_authentic,
    )
    authentic_rows = (jumia_rows + kilimall_rows)[: args.max_authentic]
    print(
        f"Collected authentic rows={len(authentic_rows)} "
        f"(jumia={sum(1 for r in authentic_rows if r['platform']=='jumia')}, "
        f"kilimall={sum(1 for r in authentic_rows if r['platform']=='kilimall')})"
    )

    keywords = load_keywords()
    needed_for_min_total = max(0, args.min_total - len(authentic_rows))
    fraud_target = max(args.target_fraud, needed_for_min_total)
    fraud_rows = generate_synthetic_fraud(fraud_target, keywords=keywords, seed=args.seed)
    print(f"Generated fraud rows={len(fraud_rows)} (target={fraud_target})")

    final_rows = authentic_rows + fraud_rows
    random.Random(args.seed).shuffle(final_rows)
    write_csv(Path(args.out_csv), final_rows)

    authentic_count = sum(1 for r in final_rows if int(r["label"]) == 0)
    fraud_count = sum(1 for r in final_rows if int(r["label"]) == 1)
    print(f"Wrote {args.out_csv}")
    print(f"Dataset rows={len(final_rows)} authentic={authentic_count} fraud={fraud_count}")
    print(f"Product URL list -> {args.out_urls}")


if __name__ == "__main__":
    main()

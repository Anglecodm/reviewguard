from __future__ import annotations

import html
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:  # pragma: no cover - optional dependency
    sync_playwright = None
    HAS_PLAYWRIGHT = False


@dataclass
class ScrapedReview:
    text: str
    user: str = "Anonymous"
    date: str = ""
    rating: str = ""


URL_SCRAPE_ENABLED_PLATFORMS = ("jumia", "kilimall")
API_INTEGRATION_PLATFORMS = {
    "walmart": "Walmart",
}
API_ENABLED_PLATFORMS = {
    "amazon": "Amazon",
    "ebay": "eBay",
    "etsy": "Etsy",
    "shopify": "Shopify",
    "woocommerce": "WooCommerce",
}


def scrape_reviews(url: str, limit: int = 1000, platform_hint: str | None = None) -> list[ScrapedReview]:
    host = urlparse(url).netloc.lower().strip()
    if not host:
        raise ValueError("Invalid URL. Paste a full product URL starting with https://")

    hint = _normalize_platform_hint(platform_hint)
    if hint in API_ENABLED_PLATFORMS:
        if hint == "amazon":
            return scrape_amazon(url, limit=limit)
        if hint == "ebay":
            return scrape_ebay(url, limit=limit)
        if hint == "etsy":
            return scrape_etsy(url, limit=limit)
        if hint == "shopify":
            return scrape_shopify(url, limit=limit)
        if hint == "woocommerce":
            return scrape_woocommerce(url, limit=limit)
    if hint in API_INTEGRATION_PLATFORMS:
        platform_name = API_INTEGRATION_PLATFORMS[hint]
        raise ValueError(
            f"{platform_name} requires API integration. Live URL scraping is enabled for "
            "jumia.co.ke and kilimall.co.ke. Amazon, eBay, Etsy, Shopify and WooCommerce marketplace modes are also supported."
        )

    platform = _detect_platform_from_host(host)
    if platform == "jumia":
        return scrape_jumia(url, limit=limit)
    if platform == "kilimall":
        return scrape_kilimall(url, limit=limit)
    if platform == "amazon":
        return scrape_amazon(url, limit=limit)
    if platform == "ebay":
        return scrape_ebay(url, limit=limit)
    if platform == "etsy":
        return scrape_etsy(url, limit=limit)
    if platform == "shopify":
        return scrape_shopify(url, limit=limit)
    if platform == "woocommerce":
        return scrape_woocommerce(url, limit=limit)

    if platform in API_INTEGRATION_PLATFORMS:
        platform_name = API_INTEGRATION_PLATFORMS[platform]
        raise ValueError(
            f"{platform_name} requires API integration. Live URL scraping is enabled for "
            "jumia.co.ke and kilimall.co.ke. Amazon, eBay, Etsy, Shopify and WooCommerce marketplace modes are also supported."
        )

    supported_hosts = ", ".join(f"{name}.co.ke" for name in URL_SCRAPE_ENABLED_PLATFORMS)
    api_only = ", ".join(API_INTEGRATION_PLATFORMS.values())
    api_live = ", ".join(API_ENABLED_PLATFORMS.values())
    raise ValueError(
        f"Unsupported domain. Live URL scraping supports {supported_hosts}. "
        f"Configured API-enabled platforms: {api_live}. Configured API-only platforms: {api_only}."
    )


def _normalize_platform_hint(platform_hint: str | None) -> str:
    if not platform_hint:
        return ""
    hint = platform_hint.strip().lower()
    if hint.startswith("platform:"):
        hint = hint.split(":", 1)[1].strip()
    return hint


def _detect_platform_from_host(host: str) -> str:
    host_norm = host.lower()
    if "jumia" in host_norm:
        return "jumia"
    if "kilimall" in host_norm:
        return "kilimall"
    checks = {
        "amazon": ("amazon.", "amzn.to"),
        "ebay": ("ebay.",),
        "etsy": ("etsy.",),
        "shopify": ("myshopify.com", "shopify."),
        "woocommerce": ("woocommerce.com",),
        "walmart": ("walmart.",),
    }
    for platform, needles in checks.items():
        if any(needle in host_norm for needle in needles):
            return platform
    return ""


def scrape_amazon(url: str, limit: int = 1000) -> list[ScrapedReview]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid Amazon URL. Use a full URL starting with https://")

    html_errors: list[str] = []
    interruption_hits = 0
    review_pages: list[str] = []

    for candidate in _amazon_candidate_urls(url):
        try:
            html_text = _fetch_html(candidate, timeout=20, headers=_amazon_headers())
        except Exception as exc:
            html_errors.append(str(exc))
            continue

        if _is_amazon_interruption_page(html_text):
            interruption_hits += 1
            continue

        reviews = _extract_reviews_from_jsonld(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        reviews = _parse_amazon_review_blocks(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        review_pages.extend(_extract_amazon_review_links(html_text, candidate))

    for review_url in list(dict.fromkeys(review_pages))[:20]:
        try:
            html_text = _fetch_html(review_url, timeout=20, headers=_amazon_headers())
        except Exception as exc:
            html_errors.append(str(exc))
            continue

        if _is_amazon_interruption_page(html_text):
            interruption_hits += 1
            continue

        reviews = _extract_reviews_from_jsonld(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        reviews = _parse_amazon_review_blocks(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

    if interruption_hits > 0:
        raise ValueError(
            "Amazon blocked this request (captcha/interruption). "
            "Try again later, switch Amazon locale URL (for example amazon.co.uk), or use a session with lower bot risk."
        )
    if html_errors:
        raise ValueError(f"Amazon review endpoints failed: {html_errors[0]}")
    raise ValueError("No reviews found on public Amazon endpoints for this item.")


def _amazon_candidate_urls(url: str) -> list[str]:
    resolved_url = _resolve_redirect_url(url)
    seeds = [resolved_url, url] if resolved_url and resolved_url != url else [url]

    asin = ""
    for candidate in seeds:
        asin = _extract_amazon_asin(candidate)
        if asin:
            break

    scheme = "https"
    host = ""
    for candidate in seeds:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            scheme = parsed.scheme
        if "amazon." in parsed.netloc.lower():
            host = parsed.netloc
            break
    if not host:
        host = "www.amazon.com"
    base_url = f"{scheme}://{host}"

    urls = list(seeds)
    if asin:
        urls.extend(
            [
                f"{base_url}/dp/{asin}",
                f"{base_url}/gp/product/{asin}",
                f"{base_url}/product-reviews/{asin}?reviewerType=all_reviews&sortBy=recent&pageNumber=1",
                f"{base_url}/product-reviews/{asin}?reviewerType=all_reviews&pageNumber=1",
                f"{base_url}/product-reviews/{asin}?reviewerType=all_reviews&pageNumber=2",
            ]
        )

    return list(dict.fromkeys(urls))


def _amazon_headers() -> dict[str, str]:
    return {
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _resolve_redirect_url(url: str, timeout: int = 15) -> str:
    host = urlparse(url).netloc.lower()
    if "amzn.to" not in host:
        return url
    try:
        req = Request(url, headers=_amazon_headers())
        with urlopen(req, timeout=timeout) as resp:
            final_url = str(resp.geturl() or "").strip()
            return final_url or url
    except Exception:
        return url


def _extract_amazon_asin(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    query = parsed.query or ""
    source = f"{path}?{query}"

    patterns = [
        r"/dp/([A-Z0-9]{10})(?:[/?]|$)",
        r"/gp/product/([A-Z0-9]{10})(?:[/?]|$)",
        r"/gp/aw/d/([A-Z0-9]{10})(?:[/?]|$)",
        r"/product-reviews/([A-Z0-9]{10})(?:[/?]|$)",
        r"/ASIN/([A-Z0-9]{10})(?:[/?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.I)
        if match:
            return match.group(1).upper()

    for key, value in parse_qsl(query, keep_blank_values=False):
        if key.lower() == "asin":
            value = value.strip().upper()
            if re.fullmatch(r"[A-Z0-9]{10}", value):
                return value
    return ""


def _is_amazon_interruption_page(html_text: str) -> bool:
    body = html_text.lower()
    strong_markers = (
        "/errors/validatecaptcha",
        "/ap/signin/",
        "form name=\"signin\"",
        "auth-validate-form",
        "automated access to amazon data",
        "sorry, we just need to make sure you're not a robot",
    )
    if any(marker in body for marker in strong_markers):
        return True

    weak_markers = (
        "robot check",
        "enter the characters you see below",
        "type the characters you see in this image",
        "captcha",
    )
    return sum(marker in body for marker in weak_markers) >= 2


def _extract_amazon_review_links(html_text: str, base_url: str) -> list[str]:
    links: list[str] = []

    for absolute in re.findall(
        r'https?://[^"\'\s<>]*amazon\.[^"\'\s<>]*/product-reviews/[A-Z0-9]{10}[^"\'\s<>]*',
        html_text,
        flags=re.I,
    ):
        links.append(absolute)

    for relative in re.findall(
        r'href=["\']([^"\']*/product-reviews/[A-Z0-9]{10}[^"\']*)["\']',
        html_text,
        flags=re.I,
    ):
        links.append(urljoin(base_url, html.unescape(relative)))

    cleaned: list[str] = []
    for link in links:
        link = str(link).strip()
        if not link:
            continue
        link = link.split("#", 1)[0]
        if "/product-reviews/" not in link:
            continue
        cleaned.append(link)

    return list(dict.fromkeys(cleaned))


def _parse_amazon_review_blocks(html_text: str, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    starts = [match.start() for match in re.finditer(r'<div[^>]+id=["\']customer_review-[^"\']+["\'][^>]*>', html_text, flags=re.I)]
    if not starts:
        starts = [match.start() for match in re.finditer(r'<div[^>]+data-hook=["\']review["\'][^>]*>', html_text, flags=re.I)]
    starts = sorted(set(starts))
    if not starts:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(html_text)
        if end - start > 50000:
            end = start + 50000
        block = html_text[start:end]

        text_candidates: list[str] = []
        for pattern in [
            r'<span[^>]+data-hook=["\']review-body["\'][^>]*>(.*?)</span>',
            r'<div[^>]+data-hook=["\']review-collapsed["\'][^>]*>(.*?)</div>',
            r'<span[^>]+class=["\'][^"\']*review-text-content[^"\']*["\'][^>]*>(.*?)</span>',
            r'<div[^>]+class=["\'][^"\']*review-text-content[^"\']*["\'][^>]*>(.*?)</div>',
        ]:
            for raw in re.findall(pattern, block, flags=re.S | re.I):
                cleaned = _clean_fragment(raw)
                if len(cleaned) >= 3:
                    text_candidates.append(cleaned)

        text = max(text_candidates, key=len) if text_candidates else ""
        if len(text) < 3:
            continue

        user = _clean_fragment(
            _extract_first(
                block,
                [
                    r'<span[^>]+class=["\'][^"\']*a-profile-name[^"\']*["\'][^>]*>(.*?)</span>',
                    r'<a[^>]+data-hook=["\']review-author["\'][^>]*>(.*?)</a>',
                    r'<span[^>]+data-hook=["\']review-author["\'][^>]*>(.*?)</span>',
                ],
            )
        )
        if not user:
            user = "Anonymous"
        user = re.sub(r"\s*verified purchase.*$", "", user, flags=re.I).strip() or "Anonymous"

        rating_raw = _extract_first(
            block,
            [
                r'<i[^>]+data-hook=["\']review-star-rating["\'][^>]*>(.*?)</i>',
                r'<i[^>]+data-hook=["\']cmps-review-star-rating["\'][^>]*>(.*?)</i>',
                r'<span[^>]+class=["\'][^"\']*a-icon-alt[^"\']*["\'][^>]*>(.*?)</span>',
            ],
        )
        rating = _normalize_amazon_rating(rating_raw)

        date_raw = _extract_first(block, [r'<span[^>]+data-hook=["\']review-date["\'][^>]*>(.*?)</span>'])
        date = _normalize_amazon_date(date_raw)

        review = ScrapedReview(text=text, user=user, date=date, rating=rating)
        key = _review_identity(review)
        if not key or key in seen:
            continue
        seen.add(key)
        reviews.append(review)
        if len(reviews) >= limit:
            return reviews[:limit]

    return reviews[:limit]


def _normalize_amazon_date(raw: str) -> str:
    date_text = _clean_fragment(raw)
    if not date_text:
        return ""
    parts = re.split(r"\bon\b", date_text, flags=re.I)
    if parts:
        tail = parts[-1].strip(" .")
        if tail:
            return tail
    return date_text


def _normalize_amazon_rating(raw: str) -> str:
    rating_text = _clean_fragment(raw)
    if not rating_text:
        return ""
    match = re.search(r"(\d(?:[.,]\d)?)\s*out of 5", rating_text, flags=re.I)
    if not match:
        return rating_text
    value = match.group(1).replace(",", ".")
    if value.endswith(".0"):
        value = value[:-2]
    return f"{value} out of 5"


def scrape_etsy(url: str, limit: int = 1000) -> list[ScrapedReview]:
    listing_id = _extract_etsy_listing_id(url)
    if not listing_id:
        raise ValueError("Could not detect Etsy listing ID from URL.")

    api_key = os.getenv("ETSY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Etsy API key missing. Set ETSY_API_KEY in Render Environment.")

    try:
        reviews = _fetch_etsy_reviews(listing_id, api_key, limit=limit)
    except Exception as exc:
        raise ValueError(f"Etsy API request failed: {exc}")

    if reviews:
        return reviews[:limit]
    raise ValueError("No reviews found from Etsy API for this listing.")


def _extract_etsy_listing_id(url: str) -> str:
    for pattern in [
        r"/listing/(\d{6,})",
        r"(?:^|[?&])listing_id=(\d{6,})",
        r"(?:^|[?&])id=(\d{6,})",
    ]:
        match = re.search(pattern, url, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _fetch_etsy_reviews(listing_id: str, api_key: str, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    offset = 0
    page_size = min(100, max(10, limit))

    headers = {
        "x-api-key": api_key,
        "accept": "application/json",
    }

    while len(reviews) < limit:
        query = urlencode({"limit": page_size, "offset": offset})
        api_url = f"https://openapi.etsy.com/v3/application/listings/{listing_id}/reviews?{query}"
        payload = _fetch_json_any(api_url, timeout=20, retries=1, headers=headers)
        if not isinstance(payload, dict):
            break

        items = payload.get("results")
        if not isinstance(items, list) or not items:
            break

        total = payload.get("count")
        for item in items:
            if not isinstance(item, dict):
                continue
            review = _review_from_etsy_item(item)
            key = _review_identity(review) if review else ""
            if review and key and key not in seen:
                seen.add(key)
                reviews.append(review)
                if len(reviews) >= limit:
                    return reviews[:limit]

        offset += len(items)
        if len(items) < page_size:
            break
        if isinstance(total, int) and offset >= total:
            break

    return reviews[:limit]


def _review_from_etsy_item(item: dict) -> ScrapedReview | None:
    text_candidates = [
        item.get("review"),
        item.get("review_text"),
        item.get("message"),
        item.get("comment"),
        item.get("text"),
    ]
    text = ""
    for value in text_candidates:
        if isinstance(value, dict):
            value = value.get("message") or value.get("text") or value.get("body")
        value_text = _clean_fragment(str(value or ""))
        if len(value_text) >= 3:
            text = value_text
            break
    if len(text) < 3:
        return None

    user_raw = item.get("reviewer_name") or item.get("name") or item.get("buyer_name") or item.get("buyer_user_id")
    user = str(user_raw or "Anonymous").strip() or "Anonymous"
    if user.isdigit():
        user = f"User {user}"

    date = ""
    date_raw = item.get("created_timestamp") or item.get("create_timestamp") or item.get("created_at") or item.get("date")
    if isinstance(date_raw, (int, float)):
        try:
            date = time.strftime("%Y-%m-%d", time.gmtime(float(date_raw)))
        except Exception:
            date = ""
    else:
        date = str(date_raw or "").strip()
        if "T" in date:
            date = date.split("T", 1)[0]

    rating = ""
    rating_raw = item.get("rating") or item.get("star_rating")
    if rating_raw is not None and str(rating_raw).strip():
        rating_text = str(rating_raw).strip()
        if rating_text.endswith(".0"):
            rating_text = rating_text[:-2]
        rating = f"{rating_text} out of 5"

    return ScrapedReview(text=text, user=user, date=date, rating=rating)


def scrape_ebay(url: str, limit: int = 1000) -> list[ScrapedReview]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid eBay URL. Use a full URL starting with https://")

    html_errors: list[str] = []
    interruption_hits = 0
    review_pages: list[str] = []

    for candidate in _ebay_candidate_urls(url):
        try:
            html_text = _fetch_html(candidate, timeout=20)
        except Exception as exc:
            html_errors.append(str(exc))
            continue

        if _is_ebay_interruption_page(html_text):
            interruption_hits += 1
            continue

        reviews = _extract_reviews_from_jsonld(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        reviews = _parse_ebay_review_blocks(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        review_pages.extend(_extract_ebay_review_links(html_text, candidate))

    for review_url in list(dict.fromkeys(review_pages)):
        try:
            html_text = _fetch_html(review_url, timeout=20)
        except Exception as exc:
            html_errors.append(str(exc))
            continue

        if _is_ebay_interruption_page(html_text):
            interruption_hits += 1
            continue

        reviews = _extract_reviews_from_jsonld(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

        reviews = _parse_ebay_review_blocks(html_text, limit=limit)
        if reviews:
            return reviews[:limit]

    if interruption_hits > 0:
        raise ValueError(
            "eBay blocked this request (Pardon our interruption). "
            "Try again later, switch eBay locale URL (for example .co.uk), or use API credentials."
        )
    if html_errors:
        raise ValueError(f"eBay review endpoints failed: {html_errors[0]}")
    raise ValueError("No reviews found on public eBay endpoints for this item.")


def _ebay_candidate_urls(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""

    urls = [url]
    item_id = _extract_ebay_item_id(url)
    if item_id:
        for domain in ("www.ebay.com", "www.ebay.co.uk", "www.ebay.de"):
            candidate = f"https://{domain}/itm/{item_id}"
            if query:
                candidate = f"{candidate}{query}"
            urls.append(candidate)
    elif "ebay.com" in host and path:
        for domain in ("www.ebay.co.uk", "www.ebay.de"):
            urls.append(f"https://{domain}{path}{query}")

    # Preserve order but dedupe.
    return list(dict.fromkeys(urls))


def _extract_ebay_item_id(url: str) -> str:
    for pattern in [
        r"/itm/(?:[^/]+/)?(\d{9,15})",
        r"(?:^|[?&])item=(\d{9,15})",
    ]:
        match = re.search(pattern, url, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _is_ebay_interruption_page(html_text: str) -> bool:
    lower = html_text.lower()
    return "pardon our interruption" in lower or "pardon our interruption..." in lower


def _extract_ebay_review_links(html_text: str, base_url: str) -> list[str]:
    links: list[str] = []
    # Typical eBay product review pages include /urw/.../product-reviews/....
    patterns = [
        r'https?://[^"\s<>]*product-reviews[^"\s<>]*',
        r'/urw/[^"\s<>]*product-reviews[^"\s<>]*',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, html_text, flags=re.I):
            clean = html.unescape(match.strip())
            if clean.startswith("/"):
                clean = urljoin(base_url, clean)
            links.append(clean)
    return links


def _parse_ebay_review_blocks(html_text: str, limit: int = 1000) -> list[ScrapedReview]:
    reviews: list[ScrapedReview] = []
    seen: set[str] = set()

    # eBay review pages vary by template; keep patterns broad but guarded by text/rating checks.
    patterns = [
        r'<li[^>]+class=["\'][^"\']*(?:review|rvw|ebay-review)[^"\']*["\'][^>]*>(.*?)</li>',
        r'<div[^>]+class=["\'][^"\']*(?:review|rvw|ebay-review)[^"\']*["\'][^>]*>(.*?)</div>',
    ]
    blocks: list[str] = []
    for pattern in patterns:
        blocks.extend(re.findall(pattern, html_text, flags=re.S | re.I))

    # Fallback for pages where reviews are rendered as generic article cards.
    if not blocks:
        blocks.extend(re.findall(r"<article[^>]*>(.*?)</article>", html_text, flags=re.S | re.I))

    for block in blocks:
        plain = _clean_fragment(block)
        if len(plain) < 8:
            continue

        rating = ""
        rating_match = re.search(r"(\d(?:\.\d)?)\s*(?:out of 5|stars?)", plain, flags=re.I)
        if rating_match:
            rating = f"{rating_match.group(1)} out of 5"

        text = _clean_fragment(
            _extract_first(
                block,
                [
                    r"<p[^>]*>(.*?)</p>",
                    r'<div[^>]+class=["\'][^"\']*(?:content|body|text)[^"\']*["\'][^>]*>(.*?)</div>',
                ],
            )
        )
        if len(text) < 5:
            # Fallback to plain block after removing obvious metadata tokens.
            text = plain
            for token in ["verified purchase", "helpful", "report abuse"]:
                text = re.sub(token, " ", text, flags=re.I)
            text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 5:
            continue

        user = _clean_fragment(
            _extract_first(
                block,
                [
                    r'<[^>]+class=["\'][^"\']*(?:author|user|reviewer)[^"\']*["\'][^>]*>(.*?)</[^>]+>',
                ],
            )
        )
        if not user:
            user = "Anonymous"

        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", plain)
        date = date_match.group(1) if date_match else ""

        review = ScrapedReview(text=text, user=user, date=date, rating=rating)
        key = _review_identity(review)
        if not key or key in seen:
            continue
        seen.add(key)
        reviews.append(review)
        if len(reviews) >= limit:
            return reviews[:limit]

    if reviews:
        return reviews[:limit]

    # Fallback: pull review-like paragraphs whose nearby context contains rating hints.
    for match in re.finditer(r"<p[^>]*>(.*?)</p>", html_text, flags=re.S | re.I):
        text = _clean_fragment(match.group(1))
        if len(text) < 5:
            continue

        start = max(0, match.start() - 260)
        end = min(len(html_text), match.end() + 260)
        context_html = html_text[start:end]
        context_plain = _clean_fragment(context_html)

        rating = ""
        rating_match = re.search(r"(\d(?:\.\d)?)\s*(?:out of 5|stars?)", context_plain, flags=re.I)
        if rating_match:
            rating = f"{rating_match.group(1)} out of 5"

        if not rating and "review" not in context_plain.lower():
            continue

        user = _clean_fragment(
            _extract_first(
                context_html,
                [
                    r'<[^>]+class=["\'][^"\']*(?:author|user|reviewer)[^"\']*["\'][^>]*>(.*?)</[^>]+>',
                ],
            )
        )
        if not user:
            user = "Anonymous"

        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", context_plain)
        date = date_match.group(1) if date_match else ""

        review = ScrapedReview(text=text, user=user, date=date, rating=rating)
        key = _review_identity(review)
        if not key or key in seen:
            continue
        seen.add(key)
        reviews.append(review)
        if len(reviews) >= limit:
            return reviews[:limit]

    return reviews[:limit]


def scrape_shopify(url: str, limit: int = 1000) -> list[ScrapedReview]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid Shopify URL. Use a full URL starting with https://")

    base_url = f"{parsed.scheme}://{parsed.netloc}"
    handle = _extract_shopify_handle(parsed.path)
    if not handle:
        raise ValueError("Could not detect Shopify product handle from URL.")

    # First try the product page JSON-LD (often includes review objects or review arrays).
    try:
        page_html = _fetch_html(url, timeout=20)
    except Exception:
        page_html = ""
    if page_html:
        reviews = _extract_reviews_from_jsonld(page_html, limit=limit)
        if reviews:
            return reviews[:limit]
        reviews = _parse_shopify_review_blocks(page_html, limit=limit)
        if reviews:
            return reviews[:limit]

    product_id = _resolve_shopify_product_id(base_url, handle)
    page_size = min(100, max(10, limit))
    query = urlencode({"limit": page_size, "page": 1})

    candidates = [
        f"{base_url}/products/{handle}?view=reviews",
        f"{base_url}/products/{handle}/reviews?{query}",
    ]
    if product_id is not None:
        judge_me_query = urlencode(
            {
                "shop_domain": parsed.netloc,
                "product_id": product_id,
                "per_page": page_size,
                "page": 1,
            }
        )
        candidates.append(f"{base_url}/apps/judgeme/reviews?{judge_me_query}")

    errors: list[str] = []
    for candidate in candidates:
        try:
            payload_text = _fetch_html(candidate, timeout=20)
        except Exception as exc:
            errors.append(str(exc))
            continue
        reviews = _parse_shopify_review_payload(payload_text, limit=limit)
        if reviews:
            return reviews[:limit]

    if errors:
        raise ValueError(f"Shopify review endpoints failed: {errors[0]}")
    raise ValueError(
        "No reviews found via Shopify public endpoints for this product. "
        "Some stores require private app credentials for review access."
    )


def _extract_shopify_handle(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return ""

    if "products" in parts:
        idx = parts.index("products")
        if idx + 1 < len(parts):
            return parts[idx + 1].strip()

    # Fallback: if URL path itself looks like a handle.
    last = parts[-1].strip()
    if last and not last.isdigit():
        return re.sub(r"\.(html?|php)$", "", last, flags=re.I)
    return ""


def _resolve_shopify_product_id(base_url: str, handle: str) -> int | None:
    endpoint = f"{base_url}/products/{handle}.js"
    try:
        payload = _fetch_json_any(endpoint, timeout=20, retries=1)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_id = payload.get("id")
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str) and raw_id.isdigit():
        return int(raw_id)
    return None


def _parse_shopify_review_payload(payload_text: str, limit: int = 1000) -> list[ScrapedReview]:
    if not payload_text.strip():
        return []

    stripped = payload_text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            blob = json.loads(payload_text)
        except Exception:
            blob = None
        if blob is not None:
            reviews = _reviews_from_api_blob(blob, limit=limit)
            if reviews:
                return reviews[:limit]
            if isinstance(blob, dict):
                for key in ("html", "reviews_html", "widget"):
                    html_blob = blob.get(key)
                    if isinstance(html_blob, str) and html_blob.strip():
                        reviews = _parse_shopify_review_blocks(html_blob, limit=limit)
                        if reviews:
                            return reviews[:limit]

    reviews = _extract_reviews_from_jsonld(payload_text, limit=limit)
    if reviews:
        return reviews[:limit]

    reviews = _parse_shopify_review_blocks(payload_text, limit=limit)
    return reviews[:limit]


def _reviews_from_api_blob(blob, limit: int = 1000) -> list[ScrapedReview]:
    items = _collect_review_items_from_blob(blob)
    if not items:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    for item in items:
        review = _review_from_api_item(item)
        key = _review_identity(review) if review else ""
        if review and key and key not in seen:
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= limit:
                break
    return reviews[:limit]


def _collect_review_items_from_blob(blob) -> list[dict]:
    if isinstance(blob, list):
        return [item for item in blob if isinstance(item, dict)]
    if not isinstance(blob, dict):
        return []

    candidates = []
    for key in ("reviews", "items", "results", "comments", "data"):
        value = blob.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
        if isinstance(value, dict):
            for inner in ("reviews", "items", "results", "comments"):
                inner_value = value.get(inner)
                if isinstance(inner_value, list):
                    candidates.extend(item for item in inner_value if isinstance(item, dict))
    return candidates


def _review_from_api_item(item: dict) -> ScrapedReview | None:
    title = str(item.get("title") or "").strip()
    body = str(
        item.get("review")
        or item.get("reviewBody")
        or item.get("content")
        or item.get("body")
        or item.get("text")
        or item.get("message")
        or ""
    ).strip()
    text = ". ".join(part for part in [title, body] if part).strip()
    text = _clean_fragment(text)
    if len(text) < 3:
        return None

    user = item.get("reviewer") or item.get("author_name") or item.get("author") or item.get("name") or "Anonymous"
    if isinstance(user, dict):
        user = user.get("name") or user.get("nickname") or "Anonymous"
    user_text = str(user).strip() or "Anonymous"

    date_raw = (
        item.get("date_created_gmt")
        or item.get("date_created")
        or item.get("date_published")
        or item.get("created_at")
        or item.get("created")
        or item.get("date")
        or ""
    )
    date_text = str(date_raw).strip()
    if "T" in date_text:
        date_text = date_text.split("T", 1)[0]

    rating_raw = item.get("rating") or item.get("score") or item.get("stars") or ""
    rating_text = str(rating_raw).strip()
    if rating_text.endswith(".0"):
        rating_text = rating_text[:-2]
    rating = f"{rating_text} out of 5" if rating_text else ""

    return ScrapedReview(text=text, user=user_text, date=date_text, rating=rating)


def _parse_shopify_review_blocks(html_text: str, limit: int = 1000) -> list[ScrapedReview]:
    # Capture common review widgets (Shopify Product Reviews, Judge.me, Okendo).
    patterns = [
        r"<li[^>]+class=[\"'][^\"']*(?:spr-review|jdgm-rev|okeReviews)[^\"']*[\"'][^>]*>(.*?)</li>",
        r"<div[^>]+class=[\"'][^\"']*(?:spr-review|jdgm-rev|okeReviews)[^\"']*[\"'][^>]*>(.*?)</div>",
    ]

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    for pattern in patterns:
        blocks = re.findall(pattern, html_text, flags=re.S | re.I)
        for block in blocks:
            text = _clean_fragment(
                _extract_first(
                    block,
                    [
                        r"<[^>]+class=['\"][^'\"]*(?:spr-review-content-body|jdgm-rev__body|okeReviews-review-body)[^'\"]*['\"][^>]*>(.*?)</[^>]+>",
                        r"<p[^>]*>(.*?)</p>",
                    ],
                )
            )
            if len(text) < 3:
                continue

            user = _clean_fragment(
                _extract_first(
                    block,
                    [
                        r"<[^>]+class=['\"][^'\"]*(?:spr-review-header-byline|jdgm-rev__author|okeReviews-review-author)[^'\"]*['\"][^>]*>(.*?)</[^>]+>",
                    ],
                )
            )
            if not user:
                user = "Anonymous"

            rating_match = re.search(r"(\d(?:\.\d)?)\s*(?:out of 5|stars?)", _clean_fragment(block), flags=re.I)
            rating = ""
            if rating_match:
                rating = f"{rating_match.group(1)} out of 5"

            date_match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", _clean_fragment(block))
            date = date_match.group(1) if date_match else ""

            review = ScrapedReview(text=text, user=user, date=date, rating=rating)
            key = _review_identity(review)
            if not key or key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= limit:
                return reviews[:limit]

    return reviews[:limit]


def scrape_woocommerce(url: str, limit: int = 1000) -> list[ScrapedReview]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Invalid WooCommerce URL. Use a full URL starting with https://")

    base_url = f"{parsed.scheme}://{parsed.netloc}"
    product_id = _extract_woocommerce_product_id_from_url(url)
    if product_id is None:
        slug = _extract_woocommerce_slug(parsed.path)
        if not slug:
            raise ValueError("Could not detect WooCommerce product slug from URL.")
        product_id = _resolve_woocommerce_product_id(base_url, slug)

    if product_id is None:
        raise ValueError("Could not resolve WooCommerce product ID via Store API.")

    errors: list[str] = []

    try:
        reviews = _fetch_woocommerce_store_reviews(base_url, product_id, limit=limit)
    except Exception as exc:
        reviews = []
        errors.append(str(exc))
    if reviews:
        return reviews[:limit]

    try:
        reviews = _fetch_woocommerce_wp_reviews(base_url, product_id, limit=limit)
    except Exception as exc:
        reviews = []
        errors.append(str(exc))
    if reviews:
        return reviews[:limit]

    if errors:
        raise ValueError(f"WooCommerce review APIs failed: {errors[0]}")
    raise ValueError("No reviews found via WooCommerce APIs for this product.")


def _extract_woocommerce_product_id_from_url(url: str) -> int | None:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    for key in ("product_id", "post", "p"):
        raw = str(query.get(key) or "").strip()
        if raw.isdigit():
            return int(raw)
    return None


def _extract_woocommerce_slug(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return ""

    slug = ""
    if "product" in parts:
        idx = parts.index("product")
        if idx + 1 < len(parts):
            slug = parts[idx + 1]
    if not slug:
        slug = parts[-1]

    slug = slug.strip().strip("/")
    slug = re.sub(r"\.(html?|php)$", "", slug, flags=re.I)
    if not slug or slug.isdigit():
        return ""
    return slug


def _resolve_woocommerce_product_id(base_url: str, slug: str) -> int | None:
    query = urlencode({"slug": slug, "per_page": 1})
    store_url = f"{base_url}/wp-json/wc/store/v1/products?{query}"
    try:
        payload = _fetch_json_any(store_url, timeout=20, retries=1)
    except Exception:
        payload = None
    product_id = _extract_product_id_from_payload(payload)
    if product_id is not None:
        return product_id

    wp_url = f"{base_url}/wp-json/wp/v2/product?{query}"
    try:
        payload = _fetch_json_any(wp_url, timeout=20, retries=1)
    except Exception:
        payload = None
    return _extract_product_id_from_payload(payload)


def _extract_product_id_from_payload(payload) -> int | None:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            raw_id = first.get("id")
            if isinstance(raw_id, int):
                return raw_id
            if isinstance(raw_id, str) and raw_id.isdigit():
                return int(raw_id)
    return None


def _fetch_woocommerce_store_reviews(base_url: str, product_id: int, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    page = 1
    page_size = min(100, max(10, limit))

    while len(reviews) < limit:
        query = urlencode({"product_id": product_id, "per_page": page_size, "page": page})
        api_url = f"{base_url}/wp-json/wc/store/v1/products/reviews?{query}"
        payload = _fetch_json_any(api_url, timeout=20, retries=1)
        if not isinstance(payload, list) or not payload:
            break

        for item in payload:
            if not isinstance(item, dict):
                continue
            text = _clean_fragment(str(item.get("review") or item.get("content") or ""))
            if len(text) < 3:
                continue

            user = str(item.get("reviewer") or item.get("author_name") or "Anonymous").strip() or "Anonymous"
            date_raw = str(item.get("date_created_gmt") or item.get("date_created") or item.get("date") or "")
            date = date_raw.split("T", 1)[0] if "T" in date_raw else date_raw

            rating_val = item.get("rating")
            rating = ""
            if rating_val is not None and str(rating_val).strip():
                rating_text = str(rating_val).strip()
                if rating_text.endswith(".0"):
                    rating_text = rating_text[:-2]
                rating = f"{rating_text} out of 5"

            review = ScrapedReview(text=text, user=user, date=date, rating=rating)
            key = _review_identity(review)
            if not key or key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= limit:
                return reviews[:limit]

        if len(payload) < page_size:
            break
        page += 1

    return reviews[:limit]


def _fetch_woocommerce_wp_reviews(base_url: str, product_id: int, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    page = 1
    page_size = min(100, max(10, limit))

    while len(reviews) < limit:
        query = urlencode({"post": product_id, "per_page": page_size, "page": page})
        api_url = f"{base_url}/wp-json/wp/v2/comments?{query}"
        payload = _fetch_json_any(api_url, timeout=20, retries=1)
        if not isinstance(payload, list) or not payload:
            break

        for item in payload:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, dict):
                raw_text = str(content.get("rendered") or "")
            else:
                raw_text = str(content or "")
            text = _clean_fragment(raw_text)
            if len(text) < 3:
                continue

            user = str(item.get("author_name") or "Anonymous").strip() or "Anonymous"
            date_raw = str(item.get("date_gmt") or item.get("date") or "")
            date = date_raw.split("T", 1)[0] if "T" in date_raw else date_raw
            review = ScrapedReview(text=text, user=user, date=date)

            key = _review_identity(review)
            if not key or key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= limit:
                return reviews[:limit]

        if len(payload) < page_size:
            break
        page += 1

    return reviews[:limit]


def scrape_jumia(url: str, limit: int = 1000) -> list[ScrapedReview]:
    if HAS_PLAYWRIGHT:
        with sync_playwright() as p:
            browser = _launch_browser(p)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            _accept_cookies(page)
            _click_first_match(page, ["Reviews", "Review", "Ratings", "Comments"])
            _scroll_page(page, steps=6)
            time.sleep(1.5)
            html = page.content()
            browser.close()
    else:
        html = _fetch_html(url)
    endpoint = _extract_jumia_reviews_endpoint(url, html)
    if endpoint:
        reviews = _fetch_all_jumia_reviews(endpoint, limit=limit)
        if reviews:
            return reviews[:limit]
    return _parse_jumia_reviews(html, limit=limit)


def scrape_kilimall(url: str, limit: int = 1000) -> list[ScrapedReview]:
    listing_id = _extract_kilimall_listing_id(url)
    api_error: Exception | None = None
    if listing_id:
        try:
            reviews = _fetch_all_kilimall_reviews(listing_id, limit=limit)
        except Exception as exc:
            reviews = []
            api_error = exc
        if reviews:
            return reviews[:limit]

    if HAS_PLAYWRIGHT:
        with sync_playwright() as p:
            browser = _launch_browser(p)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            _accept_cookies(page)
            _click_first_match(page, ["Reviews", "Review", "Ratings", "Comments"])
            _scroll_page(page, steps=6)
            time.sleep(1.5)
            html = page.content()
            browser.close()
    else:
        html = _fetch_html(url)

    if not listing_id:
        listing_id = _extract_kilimall_listing_id_from_html(html)
        if listing_id:
            try:
                reviews = _fetch_all_kilimall_reviews(listing_id, limit=limit)
            except Exception as exc:
                reviews = []
                api_error = api_error or exc
            if reviews:
                return reviews[:limit]

    reviews = _extract_reviews_from_jsonld(html, limit=limit)
    if reviews:
        return reviews[:limit]
    reviews = _parse_html_articles_reviews(html, limit=limit)
    if reviews:
        return reviews[:limit]
    # Do not fall back to generic page text; that causes product metadata
    # to be misclassified as customer reviews.
    if api_error and listing_id:
        stats = _fetch_kilimall_comment_stats(listing_id)
        total = 0
        if isinstance(stats, dict):
            try:
                total = int(stats.get("totalNum") or 0)
            except Exception:
                total = 0
        if total > 0:
            raise RuntimeError(
                f"Kilimall listing has {total} reviews, but comments API fetch failed: {api_error}"
            )
        raise RuntimeError(f"Kilimall comments API fetch failed: {api_error}")
    return []


def _fetch_html(url: str, timeout: int = 12, headers: dict[str, str] | None = None) -> str:
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _fetch_json_any(
    url: str,
    timeout: int = 12,
    retries: int = 0,
    headers: dict[str, str] | None = None,
):
    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for _ in range(attempts):
        try:
            payload = _fetch_html(url, timeout=timeout, headers=headers)
            return json.loads(payload)
        except Exception as exc:
            last_exc = exc
            time.sleep(0.35)
    raise RuntimeError(f"Failed JSON request for {url}: {last_exc}")


def _fetch_json(url: str, timeout: int = 12, retries: int = 0) -> dict:
    data = _fetch_json_any(url, timeout=timeout, retries=retries)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object payload.")
    return data


def _parse_jumia_reviews(html: str, limit: int = 1000) -> list[ScrapedReview]:
    reviews = _extract_reviews_from_jsonld(html, limit=limit)
    if reviews:
        return reviews[:limit]
    reviews = _parse_html_articles_reviews(html, limit=limit)
    if reviews:
        return reviews[:limit]

    text = _html_to_text(html)
    if "Product Reviews" not in text:
        return []

    segment = text.split("Product Reviews", 1)[1]
    stop_tokens = ["Product details", "Specifications", "Customer Feedback", "Recently Viewed"]
    for token in stop_tokens:
        if token in segment:
            segment = segment.split(token, 1)[0]
    lines = [line.strip() for line in segment.splitlines() if line.strip()]

    reviews: list[ScrapedReview] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\d(\.\d)? out of 5$", line):
            rating = line
            title = lines[i + 1] if i + 1 < len(lines) else ""
            body = lines[i + 2] if i + 2 < len(lines) else ""
            meta = lines[i + 3] if i + 3 < len(lines) else ""
            user = "Anonymous"
            date = ""
            if " by " in meta:
                date, user = meta.split(" by ", 1)
            text_blob = f"{title}. {body}".strip(". ").strip()
            if text_blob:
                reviews.append(ScrapedReview(text=text_blob, user=user, date=date, rating=rating))
            i += 4
            continue
        i += 1

    return reviews[:limit]


def _extract_kilimall_listing_id(url: str) -> str | None:
    match = re.search(r"/listing/(\d{6,})", url, flags=re.I)
    if match:
        return match.group(1)
    for pattern in [
        r"(?:^|[?&])listing[_-]?id=(\d{6,})",
        r"(?:^|[?&])id=(\d{6,})",
    ]:
        match = re.search(pattern, url, flags=re.I)
        if match:
            return match.group(1)
    return None


def _fetch_all_kilimall_reviews(listing_id: str, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    last_exc: Exception | None = None
    malformed_payload = False
    offset = 0
    page_size = min(50, max(10, limit))

    while len(reviews) < limit:
        api_url = (
            "https://mall-api.kilimall.com/comments"
            f"?listing_id={listing_id}"
            f"&offset={offset}"
            f"&limit={page_size}"
            "&order_by=normal"
        )
        try:
            response = _fetch_json(api_url, timeout=20, retries=1)
        except Exception as exc:
            last_exc = exc
            break

        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            malformed_payload = True
            break
        items = data.get("items")
        if not isinstance(items, list) or not items:
            break

        total = data.get("total")
        for item in items:
            if not isinstance(item, dict):
                continue
            base = item.get("commentBase") if isinstance(item.get("commentBase"), dict) else {}
            content = item.get("commentContent") if isinstance(item.get("commentContent"), dict) else {}

            text = _kilimall_comment_text(base, content)
            if len(text) < 3:
                continue

            is_anon = bool(base.get("isAnonymous"))
            user = "Anonymous" if is_anon else str(content.get("accountName") or base.get("accountName") or "Anonymous")

            date_raw = str(base.get("createdTime") or content.get("createdTime") or "")
            date = date_raw.split("T", 1)[0] if "T" in date_raw else date_raw

            rating_val = base.get("productScore")
            rating = ""
            if rating_val is not None and str(rating_val).strip():
                rating_text = str(rating_val).strip()
                if rating_text.endswith(".0"):
                    rating_text = rating_text[:-2]
                rating = f"{rating_text} out of 5"

            review = ScrapedReview(text=text, user=user, date=date, rating=rating)
            comment_id = str(base.get("id") or content.get("id") or "").strip()
            key = f"kilimall:{comment_id}" if comment_id else _review_identity(review)
            if not key or key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            if len(reviews) >= limit:
                return reviews[:limit]

        offset += len(items)
        if len(items) < page_size:
            break
        if isinstance(total, int) and offset >= total:
            break

    if not reviews:
        if last_exc:
            raise RuntimeError(last_exc)
        if malformed_payload:
            raise RuntimeError("Unexpected Kilimall comments payload shape.")
    return reviews[:limit]


def _extract_kilimall_listing_id_from_html(html_text: str) -> str | None:
    return _extract_first(
        html_text,
        [
            r"/listing/(\d{6,})",
            r'"listingId"\s*:\s*"?(\d{6,})"?',
            r'"listing_id"\s*:\s*"?(\d{6,})"?',
        ],
    ) or None


def _fetch_kilimall_comment_stats(listing_id: str) -> dict | None:
    url = f"https://mall-api.kilimall.com/listing-comment-statistics/{listing_id}"
    try:
        payload = _fetch_json(url, timeout=20, retries=1)
    except Exception:
        return None
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return None


def _kilimall_comment_text(base: dict, content: dict) -> str:
    text = str(content.get("content") or "").strip()
    if len(text) >= 3:
        return text

    follow_up = str(content.get("followUpContent") or "").strip()
    if len(follow_up) >= 3:
        return follow_up

    score = base.get("productScore")
    if score is not None and str(score).strip():
        score_text = str(score).strip()
        if score_text.endswith(".0"):
            score_text = score_text[:-2]
        return f"Rating-only review ({score_text} out of 5)."

    return "Rating-only review."


def _extract_jumia_reviews_endpoint(product_url: str, html_text: str) -> str | None:
    # Jumia exposes a dedicated ratings page that includes review records.
    match = re.search(r'(/catalog/productratingsreviews/sku/[A-Z0-9]+/)', html_text, flags=re.I)
    if match:
        return urljoin(product_url, match.group(1))
    # Some pages do not include the review URL but do include the SKU.
    sku = _extract_first(
        html_text,
        [
            r'"sku"\s*:\s*"([A-Z0-9]{8,})"',
            r'"pageContentKey"\s*:\s*"([A-Z0-9]{8,})"',
            r'"productSku"\s*:\s*"([A-Z0-9]{8,})"',
        ],
    )
    if sku:
        return urljoin(product_url, f"/catalog/productratingsreviews/sku/{sku}/")
    return None


def _fetch_all_jumia_reviews(endpoint_url: str, limit: int = 1000) -> list[ScrapedReview]:
    if limit <= 0:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()
    # Jumia review pages often show around 10 reviews per page.
    max_pages = max(1, min(200, (limit // 10) + 8))

    for page_num in range(1, max_pages + 1):
        page_url = endpoint_url if page_num == 1 else _set_query_param(endpoint_url, "page", str(page_num))
        try:
            page_html = _fetch_html(page_url)
        except Exception:
            break

        page_reviews = _parse_html_articles_reviews(page_html, limit=limit)
        if not page_reviews:
            break

        new_rows = 0
        for review in page_reviews:
            key = _review_identity(review)
            if not key or key in seen:
                continue
            seen.add(key)
            reviews.append(review)
            new_rows += 1
            if len(reviews) >= limit:
                return reviews[:limit]

        if new_rows == 0:
            # We likely hit repeated page content.
            break

    return reviews[:limit]


def _parse_html_articles_reviews(html_text: str, limit: int = 1000) -> list[ScrapedReview]:
    articles = re.findall(r"<article[^>]*>(.*?)</article>", html_text, flags=re.S | re.I)
    if not articles:
        return []

    reviews: list[ScrapedReview] = []
    seen: set[str] = set()

    for article in articles:
        article_plain = _clean_fragment(article)
        if not article_plain:
            continue
        if "this website uses cookies" in article_plain.lower():
            continue

        rating_match = re.search(r"(\d(?:\.\d)?)\s*out of 5", article_plain, flags=re.I)
        if not rating_match:
            continue
        rating = f"{rating_match.group(1)} out of 5"

        title = _clean_fragment(_extract_first(article, [r"<h3[^>]*>(.*?)</h3>", r"<h4[^>]*>(.*?)</h4>"]))
        body_candidates = [
            _clean_fragment(val)
            for val in re.findall(r"<p[^>]*>(.*?)</p>", article, flags=re.S | re.I)
        ]
        body_candidates = [val for val in body_candidates if val]
        body = max(body_candidates, key=len) if body_candidates else ""

        text_parts = [part for part in [title, body] if part]
        text_blob = ". ".join(text_parts).strip()
        if not text_blob:
            # Fallback to plain-text line extraction.
            lines = [ln.strip() for ln in article_plain.splitlines() if ln.strip()]
            text_candidates = [ln for ln in lines if len(ln) >= 5 and "out of 5" not in ln.lower()]
            text_blob = text_candidates[0] if text_candidates else ""
        if len(text_blob) < 5:
            continue

        date_match = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", article_plain)
        date = date_match.group(1) if date_match else ""

        user = "Anonymous"
        user_match = re.search(r"\bby\s+([A-Za-z0-9 .,'_-]{2,80})", article_plain, flags=re.I)
        if user_match:
            user = user_match.group(1).strip()
            user = re.sub(r"\s+Verified Purchase.*$", "", user, flags=re.I).strip()

        candidate = ScrapedReview(text=text_blob, user=user, date=date, rating=rating)
        key = _review_identity(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        reviews.append(candidate)
        if len(reviews) >= limit:
            break

    return reviews[:limit]


def _extract_first(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.S | re.I)
        if match:
            return match.group(1)
    return ""


def normalize_review_text(text: str) -> str:
    value = html.unescape((text or "").strip().lower())
    value = re.sub(r"\s+", " ", value)
    return value


def _review_identity(review: ScrapedReview) -> str:
    text_key = normalize_review_text(review.text)
    if not text_key:
        return ""
    user_key = normalize_review_text(review.user)
    date_key = normalize_review_text(review.date)
    rating_key = normalize_review_text(review.rating)
    return f"{text_key}|{user_key}|{date_key}|{rating_key}"


def _set_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params[key] = value
    query = urlencode(params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _clean_fragment(fragment: str) -> str:
    if not fragment:
        return ""
    cleaned = _html_to_text(fragment)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_generic_reviews(html: str, limit: int = 20) -> list[ScrapedReview]:
    reviews = _extract_reviews_from_jsonld(html, limit=limit)
    if reviews:
        return reviews[:limit]

    text = _html_to_text(html)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    candidates: list[str] = []
    for line in lines:
        if len(line) < 30:
            continue
        if any(token in line.lower() for token in ["shipping", "warranty", "specifications"]):
            continue
        if re.search(r"\d(\.\d)?/5|out of 5|stars?", line.lower()):
            continue
        if line not in candidates:
            candidates.append(line)
        if len(candidates) >= limit:
            break

    return [ScrapedReview(text=line) for line in candidates[:limit]]


def _extract_reviews_from_jsonld(html: str, limit: int = 20) -> list[ScrapedReview]:
    scripts = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.S | re.I,
    )
    if not scripts:
        return []

    reviews: list[ScrapedReview] = []
    seen = set()

    for script in scripts:
        script = script.strip()
        if not script:
            continue
        try:
            data = json.loads(script)
        except Exception:
            continue

        for node in _iter_json_nodes(data):
            if not isinstance(node, dict):
                continue
            # Direct review object
            if node.get("@type") == "Review" or "reviewBody" in node:
                review = _review_from_node(node)
                key = _review_identity(review) if review else ""
                if review and key and key not in seen:
                    seen.add(key)
                    reviews.append(review)
            # Reviews list
            if "review" in node:
                for item in _ensure_list(node.get("review")):
                    if isinstance(item, dict):
                        review = _review_from_node(item)
                        key = _review_identity(review) if review else ""
                        if review and key and key not in seen:
                            seen.add(key)
                            reviews.append(review)
            if len(reviews) >= limit:
                break
        if len(reviews) >= limit:
            break

    return reviews[:limit]


def _review_from_node(node: dict) -> ScrapedReview | None:
    text = node.get("reviewBody") or node.get("description") or node.get("text") or ""
    text = str(text).strip()
    if len(text) < 5:
        return None
    author = node.get("author", "")
    if isinstance(author, dict):
        author = author.get("name") or author.get("alternateName") or ""
    rating = ""
    rating_node = node.get("reviewRating") or node.get("aggregateRating")
    if isinstance(rating_node, dict):
        rating = str(rating_node.get("ratingValue") or rating_node.get("rating") or "")
    date = str(node.get("datePublished") or node.get("dateCreated") or "")
    return ScrapedReview(text=text, user=str(author or "Anonymous"), date=date, rating=rating)


def _iter_json_nodes(data):
    if isinstance(data, list):
        for item in data:
            yield from _iter_json_nodes(item)
    elif isinstance(data, dict):
        yield data
        if "@graph" in data:
            yield from _iter_json_nodes(data["@graph"])
        for value in data.values():
            yield from _iter_json_nodes(value)


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _html_to_text(html: str) -> str:
    html = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style.*?>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</p>", "\n", html, flags=re.I)
    html = re.sub(r"</div>", "\n", html, flags=re.I)
    html = re.sub(r"</h[1-6]>", "\n", html, flags=re.I)
    html = re.sub(r"</li>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    # Preserve newlines as structural markers for review parsing.
    html = re.sub(r"[ \t\r\f\v]+", " ", html)
    html = re.sub(r" ?\n ?", "\n", html)
    html = re.sub(r"\n{2,}", "\n", html)
    return html.strip()


def _accept_cookies(page) -> None:
    selectors = [
        "button:has-text('Accept')",
        "button:has-text('I accept')",
        "button:has-text('Agree')",
        "button:has-text('OK')",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click()
                return
        except Exception:
            continue


def _click_first_match(page, labels: Iterable[str]) -> None:
    for label in labels:
        try:
            loc = page.get_by_text(label, exact=False)
            if loc.count() > 0:
                loc.first.click()
                return
        except Exception:
            continue


def _scroll_page(page, steps: int = 6) -> None:
    for _ in range(steps):
        page.mouse.wheel(0, 1200)
        time.sleep(0.5)


def _launch_browser(playwright):
    # Prefer installed browsers to avoid Playwright download issues.
    channels = ["msedge", "chrome"]
    for channel in channels:
        try:
            return playwright.chromium.launch(channel=channel, headless=True)
        except Exception:
            continue
    # Fall back to bundled Chromium (requires Playwright browsers installed)
    try:
        return playwright.chromium.launch(headless=True)
    except Exception:
        return playwright.chromium.launch(headless=False)

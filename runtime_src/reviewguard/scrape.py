from __future__ import annotations

import html
import json
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


def scrape_reviews(url: str, limit: int = 1000) -> list[ScrapedReview]:
    host = urlparse(url).netloc.lower()
    if "jumia" in host:
        return scrape_jumia(url, limit=limit)
    if "kilimall" in host:
        return scrape_kilimall(url, limit=limit)
    raise ValueError("Unsupported domain. Supported: jumia.co.ke, kilimall.co.ke")


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


def _fetch_html(url: str, timeout: int = 12) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _fetch_json(url: str, timeout: int = 12, retries: int = 0) -> dict:
    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for _ in range(attempts):
        try:
            payload = _fetch_html(url, timeout=timeout)
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object payload.")
            return data
        except Exception as exc:
            last_exc = exc
            time.sleep(0.35)
    raise RuntimeError(f"Failed JSON request for {url}: {last_exc}")


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

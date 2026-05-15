import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from groq import Groq
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TSC_BASE = "https://thesleepcompany.in"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =============================================================================
# PRODUCT NAME EXTRACTOR
# Pulls the most specific product name from the LLM answer for accurate image fetch.
# =============================================================================

def extract_product_name_for_image(answer: str) -> str | None:
    """
    Extracts the single most specific Sleep Company product name
    from the answer text using a fast LLM call.
    Returns None if no specific product is found.
    """
    if not answer or not answer.strip():
        return None

    prompt = f"""Extract the single most specific Sleep Company product name from this text.
Return ONLY the product name (e.g. "SmartGRID Luxe mattress", "Ortho X pillow", "ErgoSmart sofa").
If no specific product is mentioned, return null.
Do not return generic words like "mattress" or "pillow" alone — only named products.

Text: {answer[:800]}
Product name:"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        name = resp.choices[0].message.content.strip().strip('"').strip("'")
        if not name or name.lower() in ("null", "none", "n/a", ""):
            return None
        print(f"[ImageFetch] Extracted product name: '{name}'")
        return name
    except Exception as e:
        print(f"[ImageFetch] Product name extraction failed ({e})")
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def fetch_product_image(topic: str) -> str | None:
    """
    Given a product topic string (e.g. "SmartGRID Luxe mattress"),
    searches thesleepcompany.in and returns the first clean product
    image URL found, or None if nothing is found.
    """
    if not topic or not topic.strip():
        return None

    topic_clean = topic.strip()
    print(f"[ImageFetch] Searching for: '{topic_clean}'")

    # ── Strategy 1: Search page → product page og:image ──────────────────────
    image_url = _search_page_image(topic_clean)
    if image_url:
        return image_url

    # ── Strategy 2: Collection/category page ─────────────────────────────────
    image_url = _collection_page_image(topic_clean)
    if image_url:
        return image_url

    print(f"[ImageFetch] No image found for '{topic_clean}'")
    return None


def fetch_product_image_from_answer(answer: str) -> str | None:
    """
    High-level helper used by app.py.
    Extracts the product name from the answer, then fetches the image.
    Falls back to None if nothing is found.
    """
    product_name = extract_product_name_for_image(answer)
    if not product_name:
        print("[ImageFetch] No product name extracted from answer — skipping image fetch")
        return None
    return fetch_product_image(product_name)


# =============================================================================
# STRATEGY 1 — Search page → follow product link → og:image
# =============================================================================

def _search_page_image(topic: str) -> str | None:
    try:
        search_url = f"{TSC_BASE}/search?q={quote_plus(topic)}&type=product"
        resp = requests.get(search_url, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        product_link = None
        result_containers = (
            soup.select(".product-item a[href*='/products/']") or
            soup.select(".search-result a[href*='/products/']") or
            soup.select(".grid__item a[href*='/products/']") or
            soup.select("li a[href*='/products/']") or
            soup.select("article a[href*='/products/']")
        )

        if result_containers:
            href = result_containers[0].get("href", "")
            product_link = href if href.startswith("http") else TSC_BASE + href
        else:
            seen = {}
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/products/" in href:
                    clean = href.split("?")[0]
                    seen[clean] = href
            if seen:
                best = max(seen.keys(), key=len)
                href = seen[best]
                product_link = href if href.startswith("http") else TSC_BASE + href

        # Step 2: Fetch product page og:image
        if product_link:
            print(f"[ImageFetch] Following product link: {product_link}")
            prod_resp = requests.get(product_link, headers=HEADERS, timeout=6)
            if prod_resp.status_code == 200:
                prod_soup = BeautifulSoup(prod_resp.text, "html.parser")

                # Try og:image (both property and name variants)
                og = (
                    prod_soup.find("meta", property="og:image") or
                    prod_soup.find("meta", attrs={"name": "og:image"})
                )
                if og and og.get("content"):
                    url = _normalise_url(og["content"])
                    if _is_product_image(url):
                        print(f"[ImageFetch] Product page og:image: {url}")
                        return url

                # Fallback: scrape img tags with data-src support  ← NOW INSIDE try
                for img in prod_soup.find_all("img"):
                    src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or ""
                    url = _normalise_url(src)
                    if url and _is_product_image(url):
                        print(f"[ImageFetch] Product page img fallback: {url}")
                        return url

        # Step 3: Search results page imgs  ← NOW INSIDE try
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            url = _normalise_url(src)
            if url and _is_product_image(url):
                print(f"[ImageFetch] Search result img fallback: {url}")
                return url

    except Exception as e:
        print(f"[ImageFetch] Search page error: {e}")

    return None


# =============================================================================
# STRATEGY 2 — Collection/category page
# =============================================================================

def _collection_page_image(topic: str) -> str | None:
    """
    Guesses a collection slug from the topic and fetches the page.
    Strips generic stop words so the slug matches real collection URLs.
    e.g. "SmartGRID Luxe mattress" → /collections/smartgrid-luxe
    """
    try:
        # Remove generic words that never appear in collection slugs
        stop_words = {
            "the", "a", "an", "and", "or", "for", "with",
        }
        words = [w for w in topic.lower().split() if w not in stop_words]
        if not words:
            return None

        slug = re.sub(r"[^a-z0-9]+", "-", " ".join(words)).strip("-")
        collection_url = f"{TSC_BASE}/collections/{slug}"
        print(f"[ImageFetch] Trying collection URL: {collection_url}")

        resp = requests.get(collection_url, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            url = _normalise_url(src)
            if url and _is_product_image(url):
                print(f"[ImageFetch] Collection page image: {url}")
                return url

    except Exception as e:
        print(f"[ImageFetch] Collection page error: {e}")

    return None


# =============================================================================
# HELPERS
# =============================================================================

def _normalise_url(src: str) -> str:
    """Ensures the URL is absolute and uses HTTPS."""
    if not src:
        return ""
    src = src.strip()
    if src.startswith("//"):
        return "https:" + src
    if src.startswith("/"):
        return TSC_BASE + src
    return src


def _is_product_image(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()

    if "thesleepcompany.in" not in url_lower and "cdn.shopify.com" not in url_lower:
        return False

    if not any(ext in url_lower for ext in (".jpg", ".jpeg", ".png", ".webp")):
        return False

    # Reject Shopify /files/ paths — these are banners/assets, not product images
    if "cdn.shopify.com" in url_lower and "/files/" in url_lower:
        return False

    width_match = re.search(r'[?&]width=(\d+)', url)
    if width_match and int(width_match.group(1)) < 400:
        return False

    filename = url_lower.split("/")[-1].split("?")[0]
    skip_keywords = [
        "logo", "icon", "banner", "badge", "flag", "sprite",
        "placeholder", "blank", "favicon", "talktous", "talk-to-us",
        "whatsapp", "wp", "footer", "header", "bg", "background",
        "arrow", "star", "rating", "tick", "check", "close",
        "social", "facebook", "instagram", "youtube", "twitter",
        "app-store", "play-store", "qr", "map", "location",
        "chat", "support", "contact", "email", "phone",
        "loader", "spinner", "hero", "slide", "slider",
        "navigation", "menu", "nav", "recommender",  # ← added
    ]
    if any(kw in filename for kw in skip_keywords):
        return False

    product_hints = [
        "mattress", "pillow", "sofa", "recliner", "chair",
        "bed", "smartgrid", "smart-grid", "ortho", "ergo",
        "feel", "luxe", "original", "elite", "pro",
    ]
    has_product_hint = any(hint in url_lower for hint in product_hints)
    # Must be /products/ path specifically, not /files/
    is_shopify_product = "cdn.shopify.com" in url_lower and "/products/" in url_lower

    return has_product_hint or is_shopify_product

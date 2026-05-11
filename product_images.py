import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TSC_BASE = "https://thesleepcompany.in"

def fetch_product_image(topic: str) -> str | None:
    """
    Given a product topic string (e.g. "SmartGRID mattress"),
    searches thesleepcompany.in and returns the first clean product
    image URL found, or None if nothing is found.
    """
    if not topic or not topic.strip():
        return None

    topic_clean = topic.strip()
    print(f"[ImageFetch] Searching for: '{topic_clean}'")

    # ── Strategy 1: Search page ───────────────────────────────────────────────
    image_url = _search_page_image(topic_clean)
    if image_url:
        return image_url

    # ── Strategy 2: Collection/category page ─────────────────────────────────
    image_url = _collection_page_image(topic_clean)
    if image_url:
        return image_url

    print(f"[ImageFetch] No image found for '{topic_clean}'")
    return None


def _search_page_image(topic: str) -> str | None:
    """Hits the site search and pulls the first product image."""
    try:
        search_url = f"{TSC_BASE}/search?q={quote_plus(topic)}&type=product"
        resp = requests.get(search_url, headers=HEADERS, timeout=6)
        if resp.status_code != 200:
            print(f"[ImageFetch] Search page returned {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Try og:image first (most reliable — it's the hero product image)
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            url = _normalise_url(og["content"])
            if _is_product_image(url):
                print(f"[ImageFetch] og:image hit: {url}")
                return url

        # Try first product card image
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            url = _normalise_url(src)
            if url and _is_product_image(url):
                print(f"[ImageFetch] Product card image: {url}")
                return url

    except Exception as e:
        print(f"[ImageFetch] Search page error: {e}")

    return None


def _collection_page_image(topic: str) -> str | None:
    """
    Guesses a collection slug from the topic and fetches the page.
    e.g. "SmartGRID Luxe mattress" → /collections/smartgrid-luxe-mattress
    """
    try:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
        collection_url = f"{TSC_BASE}/collections/{slug}"
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
    """
    Rejects icons, logos, banners, and non-image URLs.
    Accepts only clean product image URLs from thesleepcompany.in CDN.
    """
    if not url:
        return False
    url_lower = url.lower()

    # Must be from the Sleep Company CDN
    if "thesleepcompany.in" not in url_lower and "cdn.shopify" not in url_lower:
        return False

    # Must be an image file
    if not any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        # Allow CDN URLs without extensions (Shopify sometimes omits them)
        if "cdn.shopify.com/s/files" not in url_lower:
            return False

    # Reject noise
    skip_keywords = ["logo", "icon", "banner", "badge", "flag",
                     "sprite", "placeholder", "blank", "favicon"]
    if any(kw in url_lower for kw in skip_keywords):
        return False

    return True
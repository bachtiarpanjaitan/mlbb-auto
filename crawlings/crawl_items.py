"""
Mobile Legends Equipment Crawler
- Crawls Liquipedia equipment portal for all items
- Downloads item images
- Saves structured data to items.json

Usage:
    python3 crawlings/crawl_items.py
"""

import os
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://liquipedia.net"
PORTAL_URL = f"{BASE_URL}/mobilelegends/Portal:Equipment"
HEADERS = {
    "User-Agent": "MLBB-Auto-Crawler/1.0 (educational project)"
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ITEMS_JSON = PROJECT_ROOT / "assets" / "databases" / "items.json"
ITEMS_DIR = PROJECT_ROOT / "assets" / "items"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests

# Hard-coded URL overrides for items where the portal href is incorrect
URL_OVERRIDES = {
    "/mobilelegends/Steel_legplates": "/mobilelegends/Steel_Legplates",
    "/mobilelegends/Demon_Shoes": "/mobilelegends/Demon_Boots",
}

# Hard-coded name overrides (portal's alt text -> correct item name)
NAME_OVERRIDES = {
    "steel legplates": "Steel Legplates",
    "Demon Shoes": "Demon Boots",
}

# Map attribute display names to snake_case keys
ATTRIBUTE_MAP = {
    "Physical Attack": "physical_attack",
    "Magic Power": "magic_power",
    "Adaptive Attack": "adaptive_attack",
    "HP": "hp",
    "Mana": "mana",
    "Physical Defense": "physical_defense",
    "Magic Defense": "magic_defense",
    "Movement Speed": "movement_speed",
    "Attack Speed": "attack_speed",
    "Critical Chance": "critical_chance",
    "Cooldown Reduction": "cooldown_reduction",
    "Lifesteal": "lifesteal",
    "Hybrid Lifesteal": "hybrid_lifesteal",
    "Spell Vamp": "spell_vamp",
    "Physical Vamp": "physical_vamp",
    "Mana Regen": "mana_regen",
    "HP Regen": "hp_regen",
    "Magic Penetration": "magic_penetration",
    "Slow Reduction": "slow_reduction",
    "Unique Attribute": "unique_attribute",
}


def ensure_dirs():
    """Create required directories if they don't exist."""
    ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    ITEMS_JSON.parent.mkdir(parents=True, exist_ok=True)


def fetch_page(url):
    """Fetch a URL and return BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def fix_item_url(portal_href):
    """Fix known incorrect URLs from the portal page."""
    if portal_href in URL_OVERRIDES:
        return URL_OVERRIDES[portal_href]
    return portal_href


def extract_full_image_url(thumb_src):
    """Convert a thumbnail URL to the full-size image URL.

    Example:
        /commons/images/thumb/9/99/Item_XYZ_ML.png/40px-Item_XYZ_ML.png
        -> /commons/images/9/99/Item_XYZ_ML.png
    """
    if "/thumb/" in thumb_src:
        full = thumb_src.replace("/thumb/", "/", 1)
        # Remove trailing /NNNpx-Filename.png part
        full = re.sub(r"/\d+px-[^/]+$", "", full)
        return full
    return thumb_src


def download_image(item_key, image_url):
    """Download item image to items folder. Returns True on success."""
    if not image_url.startswith("http"):
        image_url = f"{BASE_URL}{image_url}"

    # Determine file extension
    parsed = urllib.parse.urlparse(image_url)
    ext = os.path.splitext(parsed.path)[1] or ".png"
    filename = f"{item_key}{ext}"
    filepath = ITEMS_DIR / filename

    if filepath.exists():
        return True  # Already downloaded

    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        print(f"  [OK] Downloaded image: {filename}")
        return True
    except Exception as e:
        print(f"  [ERR] Failed to download image {filename}: {e}")
        return False


def parse_value(raw_value):
    """Parse an attribute value string into the appropriate type.

    Returns:
        - int for numeric values
        - int for percentage values (just the number, no % sign)
        - str for descriptive text (like Unique Attribute)
    """
    raw_value = raw_value.strip()

    # Try to extract percentage value
    pct_match = re.match(r'^[+\-]?\s*(\d+(?:\.\d+)?)\s*%$', raw_value)
    if pct_match:
        return int(float(pct_match.group(1)))

    # Try to extract numeric value (with optional + prefix)
    num_match = re.match(r'^[+\-]?\s*(\d+(?:\.\d+)?)$', raw_value)
    if num_match:
        return int(float(num_match.group(1)))

    # Keep as string for descriptive text
    return raw_value


def parse_infobox(soup):
    """Parse item detail page infobox to extract metadata."""
    infobox = soup.find("div", class_="fo-nttax-infobox")
    if not infobox:
        return None, None

    data = {}
    attributes = {}

    # Iterate direct children of the infobox — each child is a wrapper <div>
    # containing either a header, a data row, or the center image.
    children = infobox.find_all("div", recursive=False)
    capturing_attrs = False

    for child in children:
        # Check if this child contains a section header
        header = child.find("div", class_="infobox-header")
        if header:
            header_text = header.get_text(strip=True)

            if not capturing_attrs and "Attributes" in header_text:
                capturing_attrs = True
                continue
            elif capturing_attrs:
                # Next header after Attributes signals end of attributes section
                break

        if capturing_attrs:
            cell = child.find("div", class_="infobox-cell-2")
            if cell:
                label = cell.get_text(strip=True).rstrip(":")
                value_div = cell.find_next_sibling("div")
                if value_div:
                    raw_value = value_div.get_text(strip=True)
                    key = ATTRIBUTE_MAP.get(label)
                    if key:
                        attributes[key] = parse_value(raw_value)

        # Also collect general metadata (Cost, Tier, Category, etc.)
        if not capturing_attrs:
            cell = child.find("div", class_="infobox-cell-2")
            if cell:
                label = cell.get_text(strip=True).rstrip(":")
                value_div = cell.find_next_sibling("div")
                if value_div:
                    value = value_div.get_text(strip=True)
                    data[label] = value

    return data, attributes


def get_h1_title(soup):
    """Get the H1 title from the detail page."""
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return None


def normalize_name(name):
    """Convert item name to a snake_case key.

    Examples:
        "Dominance Ice" -> "dominance_ice"
        "Berserker's Fury" -> "berserker_s_fury"
        "Athena's Shield" -> "athena_s_shield"
        "Haas's Claws" -> "haas_s_claws"
        "Steel Legplates" -> "steel_legplates"
    """
    name = name.replace("'", "_").replace("'", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name.lower()


def crawl_portal_page():
    """Crawl the equipment portal page to get list of all items, with proper category tracking."""
    print(f"Fetching portal page: {PORTAL_URL}")
    soup = fetch_page(PORTAL_URL)

    content = soup.find("div", class_="mw-parser-output")
    if not content:
        print("[ERR] Could not find mw-parser-output")
        return []

    items = []
    seen_names = set()
    current_category = "Unknown"

    # Walk through children of mw-parser-output to track categories
    for child in content.children:
        if not isinstance(child, Tag):
            continue

        # Track category from h2 headings
        if child.name == "div" and "mw-heading2" in child.get("class", []):
            h2 = child.find("h2")
            if h2:
                span = h2.find("span", class_="mw-headline")
                if span:
                    headline = span.get_text(strip=True)
                    current_category = re.sub(r'\s*\(\d+\)', '', headline)
            continue

        # Find item images within this category's section
        if child.name in ["div", "p", "ul"]:
            for img in child.find_all("img"):
                alt = img.get("alt", "").strip()
                src = img.get("src", "")

                # Skip non-item images
                if not alt or "Item_" not in src:
                    continue

                # Find the parent <a> tag to get the detail page URL
                parent_a = img.find_parent("a")
                if not parent_a:
                    continue

                href = parent_a.get("href", "")
                if not href:
                    continue

                # Skip already seen names (items can appear in multiple categories)
                if alt in seen_names:
                    continue
                seen_names.add(alt)

                final_name = NAME_OVERRIDES.get(alt, alt)

                # Also track override name so the bad name doesn't sneak in later
                if alt != final_name:
                    seen_names.add(alt)

                items.append({
                    "name": final_name,
                    "url": fix_item_url(href),
                    "category": current_category,
                    "image_src": src
                })

    print(f"Found {len(items)} unique items on portal page")
    return items


def crawl_item_detail(item):
    """Crawl an item's detail page for tier, price, attributes, etc."""
    detail_url = f"{BASE_URL}{item['url']}"

    try:
        time.sleep(REQUEST_DELAY)
        soup = fetch_page(detail_url)
        infobox_data, attributes = parse_infobox(soup)

        if not infobox_data:
            print(f"  [WARN] No infobox found for {item['name']} at {detail_url}")
            return None

        # Get the authoritative name from the page title
        page_title = get_h1_title(soup)
        if page_title:
            item["name"] = page_title  # Override with authoritative name

        # Extract relevant fields
        price_text = infobox_data.get("Cost", "0")
        price = int(re.sub(r'[^0-9]', '', price_text)) if price_text else 0

        tier_text = infobox_data.get("Tier", "")
        tier = int(re.sub(r'[^0-9]', '', tier_text)) if tier_text else 0

        # Category from detail page overrides portal category
        category = infobox_data.get("Category", item.get("category", "Unknown"))

        return {
            "price": price,
            "tier": tier,
            "category": category,
            "attributes": attributes or {},
        }
    except Exception as e:
        print(f"  [ERR] Failed to crawl detail for {item['name']}: {e}")
        return None


def main():
    ensure_dirs()

    # Step 1: Get all items from portal page
    portal_items = crawl_portal_page()

    # Step 2: Crawl detail pages and build complete data
    complete_items = []
    errors = []

    for idx, item in enumerate(portal_items, 1):
        name = item["name"]
        key = normalize_name(name)

        print(f"[{idx}/{len(portal_items)}] Processing: {name}")

        # Get detail info
        detail = crawl_item_detail(item)

        if detail:
            category = detail["category"]
            tier = detail["tier"]
            price = detail["price"]
            attributes = detail["attributes"]
        else:
            # Fallback to portal info
            category = item["category"]
            tier = 0
            price = 0
            attributes = {}
            errors.append(name)

        # Download image
        full_img_url = extract_full_image_url(item["image_src"])
        download_image(key, full_img_url)

        entry = {
            "id": idx,
            "key": key,
            "name": item["name"],  # Use possibly-updated name from H1
            "tier": tier,
            "category": category,
            "price": price,
        }

        # Only add attributes if there are any
        if attributes:
            entry["attributes"] = attributes

        complete_items.append(entry)

    # Step 3: Save to JSON
    ITEMS_JSON.write_text(json.dumps(complete_items, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(complete_items)} items to {ITEMS_JSON}")

    # Summary
    categories = {}
    total_with_attrs = 0
    total_attr_count = 0

    for item in complete_items:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
        if "attributes" in item and item["attributes"]:
            total_with_attrs += 1
            total_attr_count += len(item["attributes"])

    for cat in sorted(categories.keys()):
        print(f"   {cat}: {categories[cat]} items")

    print(f"\n📊 Attributes: {total_with_attrs}/{len(complete_items)} items have attributes ({total_attr_count} total attribute entries)")

    if errors:
        print(f"\n⚠️  {len(errors)} items had errors:")
        for err in errors:
            print(f"   - {err}")


if __name__ == "__main__":
    main()

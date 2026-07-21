"""
Mobile Legends Creeps Crawler (via Fandom API)
- Crawls creep data from mobile-legends.fandom.com
- Downloads creep images
- Saves structured data to creeps.json

Usage:
    python3 crawlings/crawl_creeps.py
"""

import os
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API_URL = "https://mobile-legends.fandom.com/api.php"
HEADERS = {
    "User-Agent": "MLBB-Auto-Crawler/1.0 (educational project)"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREEPS_JSON = PROJECT_ROOT / "assets" / "databases" / "creeps.json"
CREEPS_DIR = PROJECT_ROOT / "assets" / "creeps"

REQUEST_DELAY = 0.3

# Creep pages to crawl with their types (from wiki categories)
CREEPS = [
    # (page_name, display_name, type)
    ("Lithowanderer", "Lithowanderer", "Common"),
    ("Crab", "Crab", "Common"),
    ("Fire_Beetle", "Fire Beetle", "Common"),
    ("Horned_Lizard", "Horned Lizard", "Common"),
    ("Lava_Golem", "Lava Golem", "Common"),
    ("Molten_Fiend", "Molten Fiend", "Elite"),
    ("Thunder_Fenrir", "Thunder Fenrir", "Elite"),
    ("Lord", "Lord", "Legend"),
    ("Turtle", "Turtle", "Legend"),
]


def ensure_dirs():
    CREEPS_DIR.mkdir(parents=True, exist_ok=True)
    CREEPS_JSON.parent.mkdir(parents=True, exist_ok=True)


def api_call(params):
    """Make a MediaWiki API call to Fandom."""
    params["format"] = "json"
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_page_html(page_title):
    """Fetch parsed HTML of a wiki page via API."""
    data = api_call({
        "action": "parse",
        "page": page_title,
        "prop": "text",
    })
    if "parse" in data and "text" in data["parse"]:
        return data["parse"]["text"]["*"]
    return None


def fetch_full_image_url(filename):
    """Get the full-size URL for an image file."""
    data = api_call({
        "action": "query",
        "prop": "imageinfo",
        "titles": f"File:{filename}",
        "iiprop": "url",
    })
    pages = data.get("query", {}).get("pages", {})
    for pid, pdata in pages.items():
        if "imageinfo" in pdata:
            return pdata["imageinfo"][0]["url"]
    return None


def normalize_name(name):
    """Convert name to snake_case key."""
    name = name.replace("'", "_").replace("'", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip("_").lower()


def download_image(creep_key, image_url):
    """Download creep image."""
    if not image_url:
        return False
    parsed = urllib.parse.urlparse(image_url)
    ext = os.path.splitext(parsed.path)[1] or ".jpg"
    filename = f"{creep_key}{ext}"
    filepath = CREEPS_DIR / filename

    if filepath.exists():
        return True

    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        print(f"  [OK] Downloaded image: {filename}")
        return True
    except Exception as e:
        print(f"  [ERR] Failed to download image: {e}")
        return False


def parse_aside(soup):
    """Parse the aside (infobox) for metadata."""
    aside = soup.find("aside")
    if not aside:
        return {}, None

    data = {}
    image_filename = None

    for child in aside.find_all(["div", "figure"]):
        # Get image
        if child.name == "figure":
            img = child.find("img")
            if img and not image_filename:
                src = img.get("src", "")
                # Extract filename from URL
                match = re.search(r'/images/[^/]+/[^/]+/([^/]+)', src)
                if match:
                    image_filename = match.group(1)

        # Get metadata
        data_source = child.get("data-source")
        if data_source:
            val_div = child.find("div")
            if val_div:
                value = val_div.get_text(strip=True)
                data[data_source] = value

    return data, image_filename


def parse_attributes(soup):
    """Parse the Attributes section for creep stats."""
    attrs = {}

    for h2 in soup.find_all("h2"):
        if "Attributes" in h2.get_text():
            for sib in h2.find_next_siblings():
                if sib.name == "h2":
                    break
                table = sib.find("table")
                if table:
                    rows = table.find_all("tr")
                    headers = []
                    for i, row in enumerate(rows):
                        cells = row.find_all(["th", "td"])
                        vals = [c.get_text(strip=True) for c in cells]
                        if i == 0:
                            headers = vals
                        else:
                            row_name = vals[0].strip() if vals else ""
                            if row_name:
                                row_data = {}
                                for j, val in enumerate(vals[1:], 1):
                                    if j < len(headers) and headers[j]:
                                        # Parse numeric values
                                        try:
                                            num_val = float(val.replace(",", "")) if val and val != "-" else None
                                        except ValueError:
                                            num_val = val if val and val != "-" else None
                                        header_key = headers[j].lower().replace(" ", "_").replace("/", "_per_")
                                        row_data[header_key] = num_val
                                if row_data:
                                    attrs[row_name.lower()] = row_data
    return attrs


def parse_seconds(text):
    """Parse various time formats into seconds (integer).

    Handles formats like:
        "35s" -> 35
        "2 mins (120s)" -> 120
        "1.17 mins (70s)" -> 70
        "3min,Little Crab: 42s" -> 180 (main value only, ignores sub-types)
        "120s (2 min)" -> 120
        "2 minutes after the last turtle" -> 120
        "3 min2.5 min (after 18 min)" -> 180 (first value)
        "39s" -> 39
    """
    if not text:
        return None

    # Strategy: extract the FIRST time value in the text,
    # ignoring sub-values after commas or after the main value.

    # Take only the first part before any comma or newline
    first_part = text.split(",")[0].split(";")[0].strip()

    # Look for seconds in parentheses: "(NNNs)" - this is often the canonical value
    paren_sec = re.search(r'\((\d+)\s*s\)', first_part)
    if paren_sec:
        return int(paren_sec.group(1))

    # Try decimal minutes: "X.Y mins" or "X.Y min"
    dec_min = re.search(r'([\d.]+)\s*min(?:s|ute)?', first_part)
    if dec_min:
        val = float(dec_min.group(1))
        return int(round(val * 60))

    # Try simple "NNNs" in the first part
    sec_val = re.search(r'(\d+)\s*s(?:ec)?', first_part)
    if sec_val:
        return int(sec_val.group(1))

    # Try whole minutes: "N min" or "N minute"
    whole_min = re.search(r'(\d+)\s*min(?:s|ute)?', first_part)
    if whole_min:
        return int(whole_min.group(1)) * 60

    return None


def parse_ability_sections(soup):
    """Parse ability descriptions from the page sections."""
    abilities = []

    for h2 in soup.find_all("h2"):
        if "Ability" in h2.get_text():
            for sib in h2.find_next_siblings():
                if sib.name == "h2":
                    break
                txt = sib.get_text(strip=True)
                if txt:
                    abilities.append(txt[:300])

    return abilities


def main():
    ensure_dirs()

    all_creeps = []

    for idx, (page_name, display_name, creep_type) in enumerate(CREEPS, 1):
        key = normalize_name(page_name)
        print(f"[{idx}/{len(CREEPS)}] Processing: {display_name} ({creep_type})")

        time.sleep(REQUEST_DELAY)
        html = fetch_page_html(page_name)
        if not html:
            print(f"  [ERR] Failed to fetch page {page_name}")
            all_creeps.append({
                "id": idx,
                "key": key,
                "name": display_name,
                "type": creep_type,
            })
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Parse aside metadata
        aside_data, image_filename = parse_aside(soup)

        # Parse attributes
        attributes = parse_attributes(soup)

        # Parse ability descriptions
        ability_texts = parse_ability_sections(soup)

        # Build entry
        entry = {
            "id": idx,
            "key": key,
            "name": display_name,
            "type": creep_type,
        }

        # Add aside metadata with seconds conversion for time fields
        meta = {}
        if aside_data.get("first_appear"):
            meta["first_appear"] = aside_data["first_appear"]
            seconds = parse_seconds(aside_data["first_appear"])
            if seconds is not None:
                meta["first_appear_seconds"] = seconds
        if aside_data.get("refresh"):
            meta["refresh"] = aside_data["refresh"]
            seconds = parse_seconds(aside_data["refresh"])
            if seconds is not None:
                meta["refresh_seconds"] = seconds
        if aside_data.get("loots"):
            meta["loots"] = aside_data["loots"]
        if aside_data.get("ability"):
            meta["ability"] = aside_data["ability"]
        if aside_data.get("rarity"):
            meta["rarity"] = aside_data["rarity"]
        if ability_texts:
            meta["ability_details"] = ability_texts

        if meta:
            entry["meta"] = meta

        # Add attributes
        if attributes:
            entry["attributes"] = attributes

        # Download image
        if image_filename:
            full_url = fetch_full_image_url(image_filename)
            if full_url:
                download_image(key, full_url)

        all_creeps.append(entry)

    # Save to JSON
    CREEPS_JSON.write_text(json.dumps(all_creeps, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(all_creeps)} creeps to {CREEPS_JSON}")

    # Summary
    types = {}
    for c in all_creeps:
        t = c.get("type", "Unknown")
        types[t] = types.get(t, 0) + 1
    for t in sorted(types.keys()):
        print(f"   {t}: {types[t]} creeps")

    with_attrs = sum(1 for c in all_creeps if "attributes" in c)
    print(f"   With attributes: {with_attrs}/{len(all_creeps)}")


if __name__ == "__main__":
    main()

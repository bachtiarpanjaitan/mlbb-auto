"""
Mobile Legends Battle Spells Crawler (via Fandom API)
- Crawls battle spells from mobile-legends.fandom.com
- Downloads spell images
- Saves structured data with level scaling (1-15) to spells.json

Usage:
    python3 crawlings/crawl_spells.py
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
SPELLS_JSON = PROJECT_ROOT / "assets" / "databases" / "spells.json"
SPELLS_DIR = PROJECT_ROOT / "assets" / "spells"

REQUEST_DELAY = 0.3


def ensure_dirs():
    SPELLS_DIR.mkdir(parents=True, exist_ok=True)
    SPELLS_JSON.parent.mkdir(parents=True, exist_ok=True)


def api_call(params):
    """Make a MediaWiki API call to Fandom."""
    params["format"] = "json"
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def normalize_name(name):
    """Convert name to snake_case key."""
    name = name.replace("'", "_").replace("'", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip("_").lower()


def fetch_full_image_url(filename):
    """Get full-size URL for an image file."""
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


def download_image(spell_key, image_url):
    """Download spell image."""
    if not image_url:
        return False
    parsed = urllib.parse.urlparse(image_url)
    ext = os.path.splitext(parsed.path)[1] or ".png"
    filename = f"{spell_key}{ext}"
    filepath = SPELLS_DIR / filename

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


def parse_level_values(text):
    """Parse level-scaled values like '110 / 120 / 130 / ...' into an array of 15 numbers.

    Returns array of 15 floats/ints, or None if not a level scaling value.
    """
    if not text:
        return None

    # Split by "/" and clean each value
    parts = [p.strip() for p in text.split("/") if p.strip()]

    # Filter to only numeric-looking values
    values = []
    for p in parts:
        try:
            # Strip trailing non-numeric text like "(Nearby allies)" or "(Nearby"
            p_clean = re.sub(r'\s*\([^)]*\)\s*$', '', p).strip()
            p_clean = p_clean.replace("%", "").replace(",", "")
            if not p_clean:
                continue
            values.append(float(p_clean))
        except ValueError:
            continue

    if len(values) >= 2:  # At least 2 values means it's level-scaled
        return values
    return None


def extract_cell_value(cell):
    """Extract value text from a table cell, splitting by <br/> for multi-line values.

    Returns list of text segments (one per line).
    """
    segments = []
    for content in cell.contents:
        if content.name == "br":
            continue
        if hasattr(content, "get_text"):
            txt = content.get_text(strip=True)
        else:
            txt = str(content).strip()
        if txt:
            segments.append(txt)
    return segments


def extract_filename_from_url(url):
    """Extract filename from a Fandom image URL."""
    match = re.search(r'/images/[^/]+/[^/]+/([^/]+)', url)
    if match:
        return urllib.parse.unquote(match.group(1))
    return None


def main():
    ensure_dirs()

    # Fetch the page
    print("Fetching Battle Spells page...")
    data = api_call({
        "action": "parse",
        "page": "Battle_spells",
        "prop": "text",
    })

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    spells = []
    spell_buffer = {}  # Buffer to accumulate data for current spell

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        caption = table.find("caption")
        is_collapsible = "mw-collapsible" in table.get("class", [])

        first_cells = rows[0].find_all(["th", "td"]) if rows else []

        if not is_collapsible and first_cells:
            # Non-collapsible table: extract spell name + image from first row
            # First cell typically contains the spell icon (with lazy loading via data-src)
            img = first_cells[0].find("img")
            if img and img.get("alt"):
                img_alt = img["alt"].strip()

                # Skip non-spell images (like formula images)
                if not re.match(r'^[A-Za-z\s\']+$', img_alt):
                    continue

                # Save previous spell
                if spell_buffer.get("name"):
                    _save_spell(spells, spell_buffer)

                # Get actual image URL - Fandom uses lazy loading (data-src)
                img_src = img.get("data-src") or img.get("src", "")

                spell_buffer = {
                    "name": img_alt,
                    "image_file": extract_filename_from_url(img_src),
                    "description": "",
                    "scaling": {},
                }

                # Get description from second cell
                if len(first_cells) >= 2:
                    desc = first_cells[1].get_text(strip=True)
                    for known in [img_alt, "Level Scaling"]:
                        if desc.startswith(known):
                            desc = desc[len(known):]
                    spell_buffer["description"] = desc[:500]

        # Parse data rows - only from collapsible tables (clean row structure)
        if is_collapsible:
            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue

                label = cells[0].get_text(strip=True).strip()

                # Skip non-data rows
                if not label or label in ["Level Scaling", "", "Calculation"]:
                    continue

                # Skip unlocked at level only
                if label.lower().startswith("unlocked"):
                    continue

                # Get value segments (handles <br/> split for multi-line values)
                value_segments = extract_cell_value(cells[1])

                # ── Cooldown: capture sebagai scalar (nilai tunggal, bukan level-scaling) ──
                if "cooldown" in label.lower():
                    for segment in value_segments:
                        nums = re.findall(r'\d+\.?\d*', segment)
                        if nums:
                            spell_buffer["cooldown"] = float(nums[0])
                    continue

                attr_key = label.lower().replace(" ", "_").replace("-", "_").replace("%", "pct")

                for segment in value_segments:
                    level_vals = parse_level_values(segment)
                    if level_vals and spell_buffer.get("name"):
                        # If this key already exists from a previous segment,
                        # it's a sub-effect (e.g., "Shield" for nearby allies)
                        if attr_key in spell_buffer["scaling"]:
                            extra_key = f"{attr_key}_nearby_allies"
                            spell_buffer["scaling"][extra_key] = level_vals
                        else:
                            spell_buffer["scaling"][attr_key] = level_vals

    # Save last spell
    if spell_buffer.get("name"):
        _save_spell(spells, spell_buffer)

    # Save to JSON
    SPELLS_JSON.write_text(json.dumps(spells, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(spells)} spells to {SPELLS_JSON}")

    for s in spells:
        attrs = list(s.get("level_scaling", {}).keys())
        print(f"   {s['name']:20s} | {attrs}")

    # Download images
    print()
    for s in spells:
        img_file = s.get("image_file")
        if img_file:
            # Try exact filename first, then common fallbacks
            urls_to_try = [fetch_full_image_url(img_file)]

            # Some spell icons on Fandom use naming like Battle_spell_<name>.png
            name_snake = s["key"]
            fallbacks = [
                f"Battle_spell_{name_snake}.png",
                f"{name_snake.replace('_', ' ')}.png",
                f"Spell_{name_snake}.png",
            ]
            for fb in fallbacks:
                urls_to_try.append(fetch_full_image_url(fb))

            downloaded = False
            for url in urls_to_try:
                if url and download_image(s["key"], url):
                    downloaded = True
                    break
            if not downloaded:
                print(f"  [WARN] Could not download image for {s['name']}")


def _save_spell(spells, buffer):
    """Save a completed spell entry from buffer dict."""
    if not buffer.get("name"):
        return

    key = normalize_name(buffer["name"])
    entry = {
        "id": len(spells) + 1,
        "key": key,
        "name": buffer["name"],
    }
    if buffer.get("description"):
        entry["description"] = buffer["description"]
    if buffer.get("image_file"):
        entry["image_file"] = buffer["image_file"]
    if buffer.get("cooldown"):
        entry["cooldown"] = int(buffer["cooldown"])
    if buffer.get("scaling"):
        entry["level_scaling"] = buffer["scaling"]

    spells.append(entry)


if __name__ == "__main__":
    main()

"""
Mobile Legends Heroes Crawler
- Crawls Liquipedia heroes portal for all heroes
- Downloads hero portraits
- Saves structured data with skills, general info, and base stats

Usage:
    python3 crawlings/crawl_heroes.py
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
PORTAL_URL = f"{BASE_URL}/mobilelegends/Portal:Heroes"
HEADERS = {
    "User-Agent": "MLBB-Auto-Crawler/1.0 (educational project)"
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEROES_JSON = PROJECT_ROOT / "assets" / "databases" / "heroes.json"
HEROES_DIR = PROJECT_ROOT / "assets" / "heroes"

REQUEST_DELAY = 0.6  # seconds between requests


def ensure_dirs():
    HEROES_DIR.mkdir(parents=True, exist_ok=True)
    HEROES_JSON.parent.mkdir(parents=True, exist_ok=True)


def fetch_page(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_full_image_url(thumb_src):
    """Convert a thumbnail URL to the full-size image URL."""
    if "/thumb/" in thumb_src:
        full = thumb_src.replace("/thumb/", "/", 1)
        full = re.sub(r"/\d+px-[^/]+$", "", full)
        return full
    return thumb_src


def download_image(hero_key, image_url):
    """Download hero portrait."""
    if not image_url.startswith("http"):
        image_url = f"{BASE_URL}{image_url}"
    parsed = urllib.parse.urlparse(image_url)
    ext = os.path.splitext(parsed.path)[1] or ".png"
    filename = f"{hero_key}{ext}"
    filepath = HEROES_DIR / filename

    if filepath.exists():
        return True

    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        print(f"  [OK] Downloaded portrait: {filename}")
        return True
    except Exception as e:
        print(f"  [ERR] Failed to download portrait {filename}: {e}")
        return False


def normalize_name(name):
    """Convert hero name to snake_case key."""
    name = name.replace("'", "_").replace("'", "_")
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip("_").lower()


# ── Portal Page ─────────────────────────────────────────────────


def crawl_portal_page():
    """Get list of all heroes from the portal page."""
    print(f"Fetching portal page: {PORTAL_URL}")
    soup = fetch_page(PORTAL_URL)
    content = soup.find("div", class_="mw-parser-output")

    heroes = []
    seen = set()

    # The portal page is a gallery of hero thumbnails.
    # Find all thumb images with hero icons.
    for img in content.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "").strip()
        if not alt or "ML_icon_" not in src:
            continue

        parent_a = img.find_parent("a")
        if not parent_a:
            continue
        href = parent_a.get("href", "")
        if not href:
            continue

        if alt in seen:
            continue
        seen.add(alt)

        heroes.append({
            "name": alt,
            "url": href,
            "image_src": src,
        })

    print(f"Found {len(heroes)} unique heroes on portal page")
    return heroes


# ── Infobox Parser ─────────────────────────────────────────────


def parse_infobox(soup):
    """Parse hero infobox. Returns (general_info, base_stats)."""
    infobox = soup.find("div", class_="fo-nttax-infobox")
    if not infobox:
        return {}, {}

    general = {}
    base_stats = {}
    current_section = None

    for child in infobox.find_all("div", recursive=False):
        text = child.get_text(strip=True)

        # Check for section header
        header = child.find("div", class_="infobox-header-2")
        if header:
            hdr_text = header.get_text(strip=True)
            if "General Information" in hdr_text:
                current_section = "general"
            elif "Base Statistics" in hdr_text:
                current_section = "stats"
            elif "Esports Statistics" in hdr_text:
                current_section = None
            continue

        # Parse key-value pairs
        if current_section == "general":
            parsed = _parse_general_row(text, child)
            if parsed:
                key, value = parsed
                general[key] = value
        elif current_section == "stats":
            parsed = _parse_stat_row(text)
            if parsed:
                key, value = parsed
                base_stats[key] = value

    return general, base_stats


def _parse_general_row(text, child):
    """Parse a general information row from the infobox."""
    # Format: "Key:Value" - but values with multiple parts (like Price) need care
    if ":" not in text:
        return None

    # Find cell-2 (label) and its sibling (value)
    cell = child.find("div", class_="infobox-cell-2")
    if cell:
        label = cell.get_text(strip=True).rstrip(":")
        value_div = cell.find_next_sibling("div")
        if value_div:
            value = value_div.get_text(strip=True)
            return label, value

    # Fallback: split on first colon
    key, _, val = text.partition(":")
    return key.strip(), val.strip()


def _parse_stat_row(text):
    """Parse a base statistics row."""
    # Format: "StatName:Value" (e.g., "HP:2455")
    if ":" not in text:
        return None

    # Split on first colon
    key, _, val = text.partition(":")
    key = key.strip()
    val = val.strip()

    if not key or not val:
        return None

    # Try to parse numeric value
    num_match = re.match(r'^[+\-]?\s*(\d+(?:\.\d+)?)\s*$', val)
    if num_match:
        return key, float(num_match.group(1))

    # Handle percentage values
    pct_match = re.match(r'^(\d+(?:\.\d+)?)\s*%$', val)
    if pct_match:
        return key, float(pct_match.group(1))

    return key, val


def parse_general_info(general):
    """Normalize general information fields."""
    result = {}

    for key, val in general.items():
        k = key.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if "region" in k:
            result["region"] = val
        elif "city" in k:
            result["city"] = val
        elif "role" in k:
            result["role"] = val
        elif "lane" in k:
            result["lane"] = val
        elif "release" in k and "date" in k:
            result["release_date"] = val
        elif "price" in k:
            # Price field contains both BP and Diamonds concatenated like "32000599"
            if val.isdigit():
                if len(val) >= 6:
                    result["price_bp"] = int(val[:-3]) if val[:-3] else 0
                    result["price_diamonds"] = int(val[-3:]) if val[-3:] else 0
                else:
                    result["price_bp"] = int(val)
            else:
                result["price_bp"] = val
        elif "specialty" in k:
            # Specialty like "ChaseMagic Damage" -> ["Chase", "Magic Damage"]
            parts = re.findall(r'[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*', val)
            if parts:
                result["specialty"] = [p.strip() for p in parts if p.strip()]
            else:
                result["specialty"] = [val]
        elif "resource" in k:
            result["resource_bar"] = val
        elif "voice" in k:
            result["voice_actor"] = val
        elif "title" in k:
            result["title"] = val

    return result


def parse_base_stats(base_stats):
    """Normalize base statistics field names."""
    mapping = {
        "HP": "hp",
        "HP Regen": "hp_regen",
        "Mana": "mana",
        "Mana Regen": "mana_regen",
        "Physical Attack": "physical_attack",
        "Physical Defense": "physical_defense",
        "Magic Power": "magic_power",
        "Magic Defense": "magic_defense",
        "Attack Speed": "attack_speed",
        "Attack Speed Ratio": "attack_speed_ratio",
        "Movement Speed": "movement_speed",
    }

    result = {}
    for key, val in base_stats.items():
        norm_key = mapping.get(key, key.lower().replace(" ", "_"))
        result[norm_key] = val
    return result


# ── Skills Parser ──────────────────────────────────────────────


def parse_skill_heading(heading_text):
    """Convert a heading (like 'Skill 1', 'Ultimate', 'Passive') to a slug key."""
    h = heading_text.lower().replace(" ", "_")
    if "skill" in h:
        # "skill_1", "skill_2"
        return h
    return h  # "passive", "ultimate", "special_skill"


def parse_skill_table(table_div):
    """Parse a skill's per-level table div.

    Returns a dict mapping column headers to lists of values per level.
    """
    table = table_div.find("table")
    if not table:
        return {}

    result = {}
    rows = table.find_all("tr")

    # First row is header
    headers = []
    for th in rows[0].find_all(["th", "td"]):
        headers.append(th.get_text(strip=True))

    # Remaining rows are data
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        row_name = cells[0].get_text(strip=True)
        values = []
        for cell in cells[1:]:
            val = cell.get_text(strip=True)
            # Try to convert to number
            try:
                values.append(float(val) if "." in val else int(val))
            except ValueError:
                values.append(val)
        result[row_name] = values

    return result


def extract_skill_metadata(text):
    """Extract metadata (CD, Mana Cost, Spell Vamp) from skill text."""
    meta = {}

    # Cooldown: "CD:7.0" or "CD:50.0|"
    cd_match = re.search(r'CD\s*:\s*([\d.]+)', text)
    if cd_match:
        meta["cooldown"] = float(cd_match.group(1))

    # Mana Cost: "Mana Cost:40" or "Skill Cost:40"
    cost_match = re.search(r'(?:Mana|Skill)\s*Cost\s*:\s*(\d+)', text)
    if cost_match:
        meta["skill_cost"] = int(cost_match.group(1))

    # Spell Vamp Ratio
    sv_match = re.search(r'Spell\s*Vamp\s*Ratio\s*:\s*(\d+)%', text)
    if sv_match:
        meta["spell_vamp_ratio"] = int(sv_match.group(1))

    return meta


def parse_skill_container(skill_div):
    """Parse a skill container div (spellcard or tabs-dynamic).

    Returns (skill_name, labels, description, metadata, table_data)
    """
    skill_name = ""
    labels = []
    description = ""
    metadata = {}
    table_data = {}

    # Try to get skill name from sapphire-bg span (spellcard)
    saph = skill_div.find("span", class_="sapphire-bg")
    if saph:
        # Skill name
        name_span = saph.find(
            "span",
            style=lambda s: s and "font-weight:bold" in s and "font-size:16px" in s,
        )
        if name_span:
            skill_name = name_span.get_text(strip=True)
        # Labels (badges)
        labels = [
            ls.get_text(strip=True)
            for ls in saph.find_all("span", class_="white-text")
        ]
        # Metadata from the sapphire-bg text
        meta_text = saph.get_text()
        metadata = extract_skill_metadata(meta_text)
    else:
        # tabs-dynamic — try to get info from the div text directly
        all_text = skill_div.get_text()

        # Sometimes first part before specific words is the name
        # Look for known patterns
        cd_idx = all_text.find("CD:")
        if cd_idx > 0:
            # Name is before CD
            # Try to find the name by looking for labels
            possible_name = all_text[:cd_idx].strip()
            # Filter out common words
            for known_label in [
                "Basic",
                "Enhanced",
                "Damage",
                "Buff",
                "Slow",
                "Mobility",
                "CC",
                "Morph",
                "Burst",
                "Camouflage",
                "AoE",
            ]:
                possible_name = possible_name.replace(known_label, "")
            possible_name = possible_name.strip()
            if possible_name:
                skill_name = possible_name

            # Extract labels from known badge patterns in tabs-dynamic
            # The text has format: "BasicEnhancedNameLabel1Label2CD:X.X|..."
            meta_text = all_text[cd_idx:] if cd_idx >= 0 else all_text
            metadata.update(extract_skill_metadata(meta_text))

            # Labels are between the name and CD
            between = all_text[:cd_idx].replace(possible_name, "").strip()
            # Split CamelCase labels
            label_parts = re.findall(r'[A-Z][a-z]+(?:[\s/][A-Z][a-z]+)*', between)
            for part in label_parts:
                p = part.strip()
                if p and p not in ["Basic", "Enhanced"]:
                    labels.append(p)

    # Description (first text div after sapphire-bg)
    for c in skill_div.children:
        if isinstance(c, Tag) and c.name == "div" and "table2" not in c.get("class", []):
            description = c.get_text(strip=True)
            break

    # Per-level table
    table_div = skill_div.find("div", class_="table2")
    if not table_div:
        table_div = skill_div.find("div", class_=lambda c: c and "table2" in str(c))
    if table_div:
        table_data = parse_skill_table(table_div)

    return skill_name, labels, description, metadata, table_data


def find_skills(content):
    """Find all skill containers in the abilities section.

    Returns a list of (heading_slug, skill_data) tuples.
    """
    skills = []
    current_heading = ""
    in_abilities = False

    for child in content.children:
        if not isinstance(child, Tag):
            continue

        # Track section boundaries
        if child.name == "div" and "mw-heading" in child.get("class", []):
            inner = child.find(["h2", "h3", "h4"])
            if inner:
                # Heading text is directly in the <h2>/<h3> tag, not in a span
                heading = inner.get_text(strip=True)

                if heading == "Abilities":
                    in_abilities = True
                    continue

                if in_abilities and inner.name in ("h3", "h4"):
                    if "Video" in heading or "Music" in heading:
                        # Skip video sections
                        continue
                    current_heading = heading
                    continue

                if inner.name == "h2" and heading not in (
                    "Abilities",
                    "Contents",
                    "Additional Content",
                ):
                    in_abilities = False
                    continue

        if not in_abilities:
            continue

        # Check for skill container: spellcard or tabs-dynamic
        is_skill = False
        if child.name == "div":
            cls = child.get("class", [])
            if "spellcard" in cls:
                is_skill = True
            elif any("tabs-dynamic" in c for c in cls):
                is_skill = True

        if is_skill and current_heading:
            slug = parse_skill_heading(current_heading)
            name, labels, desc, meta, table = parse_skill_container(child)

            skill_data = {
                "name": name,
                "description": desc[:500] if desc else "",
            }
            if labels:
                skill_data["labels"] = labels
            if meta:
                skill_data.update(meta)
            if table:
                # Normalize table keys to snake_case
                norm_table = {}
                for k, v in table.items():
                    nk = k.lower().replace(" ", "_")
                    norm_table[nk] = v
                skill_data["levels"] = norm_table

            skills.append((slug, skill_data))
            current_heading = ""

    return skills


# ── Main ───────────────────────────────────────────────────────


def crawl_hero_detail(hero):
    """Fetch hero detail page and parse all data."""
    detail_url = f"{BASE_URL}{hero['url']}"

    try:
        time.sleep(REQUEST_DELAY)
        soup = fetch_page(detail_url)

        # Parse infobox
        general, base_stats = parse_infobox(soup)

        # Parse skills
        content = soup.find("div", class_="mw-parser-output")
        if not content:
            print(f"  [WARN] No content for {hero['name']}")
            return None

        skills = find_skills(content)

        return {
            "general": parse_general_info(general),
            "base_stats": parse_base_stats(base_stats),
            "skills": skills,
        }
    except Exception as e:
        print(f"  [ERR] Failed to crawl detail for {hero['name']}: {e}")
        return None


def main():
    ensure_dirs()

    # Step 1: Get all heroes from portal
    portal_heroes = crawl_portal_page()

    # Step 2: Crawl details for each hero
    all_heroes = []
    errors = []
    stats = {"with_portrait": 0, "with_skills": 0}

    for idx, hero in enumerate(portal_heroes, 1):
        name = hero["name"]
        key = normalize_name(name)

        print(f"[{idx}/{len(portal_heroes)}] Processing: {name}")

        detail = crawl_hero_detail(hero)

        if detail is None:
            errors.append(name)
            all_heroes.append({
                "id": idx,
                "key": key,
                "name": name,
            })
            continue

        # Build output entry
        entry = {
            "id": idx,
            "key": key,
            "name": name,
        }

        # General info
        gi = detail["general"]
        if gi.get("title"):
            entry["title"] = gi["title"]
        if gi.get("role"):
            entry["role"] = gi["role"]
        if gi.get("lane"):
            entry["lane"] = gi["lane"]
        if gi.get("region"):
            entry["region"] = gi["region"]

        # Full general_information
        entry["general_information"] = gi

        # Base statistics
        if detail["base_stats"]:
            entry["base_statistics"] = detail["base_stats"]

        # Skills
        skill_objects = {}
        for slug, skill_data in detail["skills"]:
            # Tambah unlock_level
            if slug in ("skill_1", "passive", "special_skill", "battle_spell"):
                skill_data["unlock_level"] = 1
            elif slug == "skill_2":
                skill_data["unlock_level"] = 2
            elif slug == "ultimate":
                cd_levels = skill_data.get("levels", {}).get("cooldown", [])
                skill_data["unlock_level"] = 4 if len(cd_levels) in (0, 3) else 1
            else:
                skill_data["unlock_level"] = 1
            skill_objects[slug] = skill_data

        if skill_objects:
            entry["skills"] = skill_objects
            stats["with_skills"] += 1

        # Download image from portal thumbnail (convert to full size)
        thumb_url = hero["image_src"]
        if thumb_url:
            full_url = extract_full_image_url(thumb_url)
            if download_image(key, full_url):
                stats["with_portrait"] += 1

        all_heroes.append(entry)

    # Step 3: Save to JSON
    HEROES_JSON.write_text(json.dumps(all_heroes, indent=2, ensure_ascii=False))
    print(f"\n✅ Saved {len(all_heroes)} heroes to {HEROES_JSON}")

    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total heroes: {len(all_heroes)}")
    print(f"   With portrait: {stats['with_portrait']}")
    print(f"   With skills: {stats['with_skills']}")

    # Count skill types
    skill_counts = {}
    for h in all_heroes:
        if "skills" in h:
            for sk in h["skills"]:
                skill_counts[sk] = skill_counts.get(sk, 0) + 1
    print(f"\n📊 Skill types found:")
    for sk in sorted(skill_counts.keys()):
        print(f"   {sk}: {skill_counts[sk]} heroes")

    if errors:
        print(f"\n⚠️  {len(errors)} heroes had errors:")
        for err in errors:
            print(f"   - {err}")


if __name__ == "__main__":
    main()

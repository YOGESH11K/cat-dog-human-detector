"""
Download free cat and dog photos from Wikimedia Commons into:
    dataset/cat/
    dataset/dog/

Uses only Python's built-in urllib. Re-run to get more images.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "class4-catdog-dataset/1.0 (educational project; contact: local)",
}

SUBCS = {
    "cat": [
        "domestic cat photograph",
        "cat portrait photograph",
        "house cat photograph",
        "kitten photograph",
    ],
    "dog": [
        "dog photograph",
        "dog portrait photograph",
        "pet dog photograph",
        "puppy photograph",
    ],
    "human": [
        "person portrait photograph",
        "man face close up photograph",
        "woman face photograph",
        "human face photograph",
        "child face portrait photograph",
    ],
}

TARGET = 400
WANT_EXT = (".jpg", ".jpeg", ".png", ".webp")
SKIP_EXT = (".gif", ".tiff", ".tif", ".svg", ".bmp")
MIN_DIM = 300


def api_search(query, limit=60):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query + " filetype:bitmap",
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|size|mime",
        "iiurlwidth": "400",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    return data


def candidates(data):
    pages = data.get("query", {}).get("pages", {})
    items = []
    for pid, page in pages.items():
        lower = page["title"].lower()
        if not lower.endswith(WANT_EXT) or lower.endswith(SKIP_EXT):
            continue
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "").split("/")[-1]
        if mime not in ("jpeg", "png", "webp"):
            continue
        width = info.get("width", 0)
        height = info.get("height", 0)
        if width < MIN_DIM or height < MIN_DIM:
            continue
        items.append((info.get("thumburl") or info.get("url"), width, height))
    return items


def download_to(url, path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def fill_class(folder, queries, want):
    os.makedirs(folder, exist_ok=True)
    existing = [n for n in os.listdir(folder) if n.lower().endswith(WANT_EXT)]
    have = len(existing)
    base = have + 1
    missing = want - have
    if missing <= 0:
        print(f"{folder}: already has {have} images, nothing to do.")
        return

    print(f"{folder}: downloading {missing} more images...")
    seen = set(existing)
    tried = 0
    for query in queries:
        if missing <= 0:
            break
        try:
            data = api_search(query)
        except Exception as exc:
            print(f"  search '{query}' failed: {exc}")
            continue
        for url, w, h in candidates(data):
            if missing <= 0:
                break
            if url in seen:
                continue
            ext = os.path.splitext(url)[1].lower() or ".jpg"
            if not ext.endswith(WANT_EXT):
                ext = ".jpg"
            name = f"img_{base:03d}{ext}"
            dest = os.path.join(folder, name)
            try:
                size = download_to(url, dest)
                if size < 1000:
                    os.remove(dest)
                    continue
                seen.add(url)
                base += 1
                missing -= 1
                tried = 0
                print(f"  saved {name}  ({w}x{h}, {size // 1024} KB)")
            except Exception as exc:
                tried += 1
                if os.path.exists(dest):
                    os.remove(dest)
                if tried > 15:
                    print(f"  too many failures for {folder}, giving up: {exc}")
                    break
            time.sleep(0.4)
    if missing > 0:
        print(f"  warning: only got {want - missing} of {want} images for {os.path.basename(folder)}")
    else:
        print(f"  done: {folder} now has {want} images.")


def main():
    total = 0
    for cls, queries in SUBCS.items():
        folder = os.path.join(DATASET_DIR, cls)
        if cls == "cat":
            fill_class(folder, queries, TARGET)
        else:
            fill_class(folder, queries, TARGET)
        n = len([x for x in os.listdir(folder) if x.lower().endswith(WANT_EXT)])
        total += n
        print(f"  -> total in {folder}: {n}\n")
    print(f"DONE. Dataset now has {total} images.")
    print("Next step:  python train_model.py")


if __name__ == "__main__":
    main()
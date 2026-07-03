"""Download TFT set 16 champion, item and trait icons from Community Dragon.

Icons and a ``catalog.json`` are written into the ``src/web`` frontend package so
it can be served fully offline afterwards. The catalog is the single
payload the UI needs to render the unit / item / trait pickers: it maps every
apiName the models understand to its display name, cost, traits and local icon.

Run: ``trp fetch-assets``  (or ``python -m src.api.fetch_assets``)
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from src.api import config
from src.training.utils.static_data import EMBLEMS, TRAITS, UNITS
from src.training.utils.vocab import load_vocab

if TYPE_CHECKING:
    from pathlib import Path

CDRAGON_JSON = "https://raw.communitydragon.org/latest/cdragon/tft/en_us.json"
CDRAGON_GAME = "https://raw.communitydragon.org/latest/game/"
UA = {"User-Agent": "Mozilla/5.0 (tft-round-prediction asset fetcher)"}
SET_NUMBER = "16"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _icon_url(cdragon_path: str) -> str:
    """Map a Community Dragon asset path to its rendered PNG URL."""
    p = cdragon_path.lower().replace(".tex", ".png").replace(".dds", ".png")
    return CDRAGON_GAME + p


def _download(url: str, dest: Path) -> bool:
    try:
        data = _get(url)
    except Exception as exc:  # noqa: BLE001 - best-effort, report and continue
        print(f"  ! failed {dest.name}: {exc}")
        return False
    dest.write_bytes(data)
    return True


def fetch_assets(max_workers: int = 16) -> None:
    """Fetch all icons and build the catalog. Idempotent (overwrites)."""
    print("Fetching Community Dragon TFT data ...")
    data = json.loads(_get(CDRAGON_JSON))
    set_data = data["sets"][SET_NUMBER]
    champions = {c["apiName"]: c for c in set_data["champions"]}
    traits_meta = {t["name"]: t for t in set_data["traits"]}
    items = {it["apiName"]: it for it in data.get("items", [])}

    units_dir = config.ASSET_DIR / "units"
    items_dir = config.ASSET_DIR / "items"
    traits_dir = config.ASSET_DIR / "traits"
    for d in (units_dir, items_dir, traits_dir):
        d.mkdir(parents=True, exist_ok=True)

    unit_vocab = load_vocab(
        str(config.resolve("data/set16/static/vocabulary/unit_vocab.json"))
    )
    item_list = json.loads(config.resolve("data/set16/static/item.json").read_text())

    jobs: list[tuple[str, Path]] = []  # (url, dest)
    unit_catalog: list[dict] = []
    for api_name in unit_vocab:
        champ = champions.get(api_name)
        info = UNITS.get(api_name, {})
        icon_rel = f"assets/units/{api_name}.png"
        dest = units_dir / f"{api_name}.png"
        if champ:
            icon_src = champ.get("tileIcon") or champ.get("squareIcon")
            if icon_src:
                jobs.append((_icon_url(icon_src), dest))
        unit_catalog.append(
            {
                "apiName": api_name,
                "name": (champ or {}).get("name") or api_name,
                "cost": (champ or {}).get("cost", info.get("cost")),
                "traits": info.get("traits", (champ or {}).get("traits", [])),
                "icon": icon_rel,
                "hasIcon": bool(champ),
            }
        )

    item_catalog: list[dict] = []
    for api_name in item_list:
        it = items.get(api_name)
        dest = items_dir / f"{api_name}.png"
        if it and it.get("icon"):
            jobs.append((_icon_url(it["icon"]), dest))
        item_catalog.append(
            {
                "apiName": api_name,
                "name": (it or {}).get("name") or api_name,
                "icon": f"assets/items/{api_name}.png",
                "isEmblem": api_name in EMBLEMS,
                "emblemTrait": EMBLEMS.get(api_name),
                "hasIcon": bool(it and it.get("icon")),
            }
        )

    trait_catalog: list[dict] = []
    for name, bps in TRAITS.items():
        meta = traits_meta.get(name)
        safe = name.replace(" ", "_").replace("'", "")
        dest = traits_dir / f"{safe}.png"
        if meta and meta.get("icon"):
            jobs.append((_icon_url(meta["icon"]), dest))
        trait_catalog.append(
            {
                "name": name,
                "breakpoints": sorted(bps),
                "icon": f"assets/traits/{safe}.png",
                "hasIcon": bool(meta and meta.get("icon")),
            }
        )

    print(f"Downloading {len(jobs)} icons with {max_workers} workers ...")
    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_download, url, dest): dest for url, dest in jobs}
        for fut in as_completed(futures):
            ok += bool(fut.result())
    print(f"Downloaded {ok}/{len(jobs)} icons.")

    catalog = {
        "set": SET_NUMBER,
        "units": sorted(unit_catalog, key=lambda u: (u["cost"] or 99, u["name"])),
        "items": item_catalog,
        "traits": trait_catalog,
    }
    config.CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote catalog -> {config.CATALOG_PATH}")


if __name__ == "__main__":
    fetch_assets()

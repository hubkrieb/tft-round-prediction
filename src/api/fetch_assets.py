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

# Community Dragon item tags. Most are hashed; these were identified by checking
# which items carry them (every Ornn/Darkin/Shimmerscale artifact has the
# artifact tag, every TFT5_*Radiant item has the radiant one).
TAG_COMPONENT = "component"
TAG_ARTIFACT = "{44ace175}"
TAG_RADIANT = "{6ef5c598}"

# In the UI catalog: placeholder "items" the game uses internally.
HIDDEN_ITEMS = {"TFT_Item_Blank", "TFT_Item_EmptyBag"}
# Non-placeable board entities the picker should not offer at all.
HIDDEN_UNITS = {"TFT_ElderDragon", "TFT9_SLIME_Crab", "TFT16_MalzaharVoidling"}
# Placeable but cost-less special units: listed after the regular roster.
SPECIAL_UNITS = {
    "TFT_BlueGolem",
    "TFT_TrainingDummy",
    "TFT16_Atakhan",
    "TFT16_FreljordProp",
    "TFT16_AnnieTibbers",
    "TFT16_AzirUltSoldier",
    "TFT16_PiltoverInvention",
}

# Display order of the item categories in the picker's "All" pane: normal
# (full) items first, then components, emblems, artifacts, bilgewater, radiant.
ITEM_CATEGORY_ORDER = (
    "normal",
    "component",
    "emblem",
    "artifact",
    "bilgewater",
    "radiant",
)


def _item_category(api_name: str, tags: list[str]) -> str:
    """Classify an item for the UI picker (component/normal/emblem/...)."""
    if api_name in EMBLEMS:
        return "emblem"
    if TAG_COMPONENT in tags:
        return "component"
    if api_name.startswith("TFT16_Item_Bilgewater_"):
        return "bilgewater"
    if TAG_RADIANT in tags or api_name.endswith("Radiant"):
        return "radiant"
    if TAG_ARTIFACT in tags or "_Artifact_" in api_name:
        return "artifact"
    return "normal"


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
        if api_name in HIDDEN_UNITS:
            continue
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
                "special": api_name in SPECIAL_UNITS,
            }
        )

    item_catalog: list[dict] = []
    for api_name in item_list:
        if api_name in HIDDEN_ITEMS:
            continue
        it = items.get(api_name)
        dest = items_dir / f"{api_name}.png"
        if it and it.get("icon"):
            jobs.append((_icon_url(it["icon"]), dest))
        item_catalog.append(
            {
                "apiName": api_name,
                "name": (it or {}).get("name") or api_name,
                "icon": f"assets/items/{api_name}.png",
                "category": _item_category(api_name, (it or {}).get("tags") or []),
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
                # Single-breakpoint traits are the "unique" ones (1-of / duo traits).
                "unique": len(bps) == 1,
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
        # Regular roster by cost, special (cost-less) units at the end.
        "units": sorted(
            unit_catalog, key=lambda u: (u["special"], u["cost"] or 99, u["name"])
        ),
        # Grouped by category so the picker's "All" pane reads component ->
        # normal -> emblem -> artifact -> bilgewater -> radiant.
        "items": sorted(
            item_catalog,
            key=lambda i: (ITEM_CATEGORY_ORDER.index(i["category"]), i["name"]),
        ),
        "traits": trait_catalog,
    }
    config.CATALOG_PATH.write_text(json.dumps(catalog, indent=2))
    print(f"Wrote catalog -> {config.CATALOG_PATH}")


if __name__ == "__main__":
    fetch_assets()

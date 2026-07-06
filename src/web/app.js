"use strict";

/* ------------------------------------------------------------------ *
 * TFT Round Predictor — board builder front-end.
 * State lives in `state.boards`; the DOM is re-rendered from it.
 * ------------------------------------------------------------------ */

const ROWS = 4;
const COLS = 7;
const MAX_ITEMS = 3;
const MAX_STARS = 4;
const COST_COLORS = { 1: "--c1", 2: "--c2", 3: "--c3", 4: "--c4", 5: "--c5", 6: "--c6" };

// Pointer type of the most recent pointerdown ("mouse" | "touch" | "pen").
// Click events don't carry it reliably across browsers, so track it here.
let lastPointerType = "mouse";
window.addEventListener("pointerdown", (e) => (lastPointerType = e.pointerType || "mouse"), true);
const isTouchEvent = () => lastPointerType !== "mouse";

const state = {
  boards: { player: {}, opponent: {} }, // key "r_c" -> {unit, tier, items:[]}
  model: "vit",
  autoPredict: false,
  loadedModels: new Set(), // backends already loaded server-side
};

let catalog = null;
let unitsByApi = {};
let itemsByApi = {};
let traitsByName = {};

const $ = (sel) => document.querySelector(sel);
const cellKey = (r, c) => `${r}_${c}`;
const costVar = (cost) => `var(${COST_COLORS[Math.min(cost || 1, 6)]})`;
// Special units (dummy, Atakhan, …) have no meaningful cost: neutral colour.
const unitCostVar = (u) => (u.special ? "var(--c1)" : costVar(u.cost));

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */
async function boot() {
  catalog = await fetch("catalog.json").then((r) => r.json());
  catalog.units.forEach((u) => (unitsByApi[u.apiName] = u));
  catalog.items.forEach((i) => (itemsByApi[i.apiName] = i));
  catalog.traits.forEach((t) => (traitsByName[t.name] = t));

  buildHexBoards();
  buildModelSelect();
  buildCostFilter();
  buildItemFilter();
  renderUnitGrid();
  renderItemGrid();
  renderTraits();
  wireEvents();
}

/* ------------------------------------------------------------------ *
 * Hex boards
 * ------------------------------------------------------------------ */
function buildHexBoards() {
  for (const side of ["player", "opponent"]) {
    const board = $(`#board-${side}`);
    board.innerHTML = "";
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const hex = document.createElement("div");
        hex.className = "hex";
        hex.dataset.side = side;
        hex.dataset.r = r;
        hex.dataset.c = c;
        board.appendChild(hex);
      }
    }
  }
  layoutHexes();
}

// Pointy-top honeycomb geometry. The hex width is sized to the board column so
// the grid scales responsively; height follows from the regular-hexagon ratio.
function hexWidthForContainer() {
  const area = $(".board-area");
  const avail = (area ? area.clientWidth : 480) - 8;
  return Math.max(38, Math.min(66, Math.floor(avail / 7.5)));
}

function layoutHexes() {
  const hexW = hexWidthForContainer();
  const hexH = hexW * 1.1547;
  document.documentElement.style.setProperty("--hexW", `${hexW}px`);
  document.querySelectorAll(".hexboard .hex").forEach((hex) => {
    const r = +hex.dataset.r;
    const c = +hex.dataset.c;
    hex.style.left = `${c * hexW + (r % 2) * (hexW / 2) + 4}px`;
    hex.style.top = `${r * hexH * 0.75 + 4}px`;
  });
}

function renderBoards() {
  for (const side of ["player", "opponent"]) {
    const board = $(`#board-${side}`);
    board.querySelectorAll(".hex").forEach((hex) => {
      const key = cellKey(hex.dataset.r, hex.dataset.c);
      const cell = state.boards[side][key];
      hex.classList.toggle("occupied", !!cell);
      hex.innerHTML = "";
      if (!cell) {
        hex.classList.remove("stars-open");
        return;
      }
      const u = unitsByApi[cell.unit] || {};
      hex.style.setProperty("--cost-color", unitCostVar(u));

      const portrait = document.createElement("img");
      portrait.className = "unit-portrait";
      portrait.src = u.icon || "";
      portrait.alt = u.name || cell.unit;
      portrait.draggable = false;
      const frame = document.createElement("div");
      frame.className = "unit-frame";
      // Persistent star pips (hidden while the hover selector is showing).
      const stars = document.createElement("div");
      stars.className = "stars";
      stars.textContent = "★".repeat(cell.tier);

      // Hover star selector: click a star to set the level (1-4); hovering a
      // star previews the fill up to it.
      const starSelect = document.createElement("div");
      starSelect.className = "star-select";
      const starEls = [];
      const paint = (n) => starEls.forEach((el, i) => el.classList.toggle("on", i < n));
      for (let t = 1; t <= MAX_STARS; t++) {
        const s = document.createElement("span");
        s.className = "star";
        s.textContent = "★";
        s.title = `${t}-star`;
        s.addEventListener("mouseenter", () => paint(t));
        // pointerup, not click: touch pipelines don't reliably synthesize
        // clicks on custom elements. pointerdown is already excluded from the
        // board's drag handler for the star selector.
        s.addEventListener("pointerup", (e) => {
          e.stopPropagation();
          cell.tier = t;
          afterEdit();
        });
        s.addEventListener("click", (e) => e.stopPropagation());
        starEls.push(s);
        starSelect.appendChild(s);
      }
      paint(cell.tier);
      starSelect.addEventListener("mouseleave", () => paint(cell.tier));

      const items = document.createElement("div");
      items.className = "items";
      cell.items.forEach((it, idx) => {
        const im = document.createElement("img");
        im.src = (itemsByApi[it] || {}).icon || "";
        im.title = `${(itemsByApi[it] || {}).name || it} — drag to move, drop outside to remove`;
        im.draggable = false;
        // Drag the item to another unit to move it, or off the boards to
        // discard it. stopPropagation keeps the unit itself from dragging.
        im.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          beginDrag(e, "board-item", { from: { side, key, idx } }, im.src);
        });
        im.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          e.stopPropagation(); // don't also remove the unit
          if (isTouchEvent()) return; // touch long-press is not a remove
          cell.items.splice(idx, 1);
          afterEdit();
        });
        items.appendChild(im);
      });
      hex.append(portrait, frame, stars, starSelect, items);
    });
  }
}

/* ------------------------------------------------------------------ *
 * Placement / interaction
 * ------------------------------------------------------------------ */
function placeUnit(side, r, c, apiName) {
  state.boards[side][cellKey(r, c)] = { unit: apiName, tier: 1, items: [] };
  afterEdit();
}

// First empty cell of a board in reading order (frontline first).
function firstFreeCell(side) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (!state.boards[side][cellKey(r, c)]) return { r, c };
    }
  }
  return null;
}

// First placed unit (reading order) that still has a free item slot.
function firstFreeItemSlot(side) {
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = state.boards[side][cellKey(r, c)];
      if (cell && cell.items.length < MAX_ITEMS) return cellKey(r, c);
    }
  }
  return null;
}

function moveUnit(from, to) {
  const src = state.boards[from.side][from.key];
  if (!src || (from.side === to.side && from.key === to.key)) return;
  const dst = state.boards[to.side][to.key];
  state.boards[to.side][to.key] = src;
  if (dst) state.boards[from.side][from.key] = dst; // swap
  else delete state.boards[from.side][from.key];
  afterEdit();
}

function equipItem(side, key, apiName) {
  const cell = state.boards[side][key];
  if (!cell) return toast("Items go on a placed champion.");
  if (cell.items.length >= MAX_ITEMS) return toast("Unit already has 3 items.");
  cell.items.push(apiName);
  afterEdit();
}

function moveItem(from, to) {
  const src = state.boards[from.side][from.key];
  const dst = state.boards[to.side][to.key];
  if (!src || !dst || src === dst) return;
  if (dst.items.length >= MAX_ITEMS) return toast("Unit already has 3 items.");
  dst.items.push(src.items.splice(from.idx, 1)[0]);
  afterEdit();
}

/* ------------------------------------------------------------------ *
 * Pointer-events drag engine (works for both mouse and touch).
 * Picker entries drag with a mouse only (touch uses tap-to-place so the
 * grids stay scrollable); board units and their items drag with any
 * pointer, and dropping them outside the two boards discards them.
 * ------------------------------------------------------------------ */
let dragCtx = null;

// `onTap` runs when the pointer goes down and up without moving. Taps are
// handled here at the pointer level rather than with click listeners because
// touch pipelines don't reliably synthesize click events on custom elements.
function beginDrag(e, kind, payload, iconSrc, onTap) {
  if (dragCtx || (e.pointerType === "mouse" && e.button !== 0)) return;
  dragCtx = { pointerId: e.pointerId, kind, payload, iconSrc, onTap, sx: e.clientX, sy: e.clientY, started: false, ghost: null };
  window.addEventListener("pointermove", onDragMove);
  window.addEventListener("pointerup", onDragEnd);
  window.addEventListener("pointercancel", onDragCancel);
}

// Tap detection for elements that are not drag sources (picker entries on
// touch, empty hexes): pointerdown..pointerup with no movement in between.
// A scroll or gesture takeover fires pointercancel and voids the tap.
function watchTap(e, cb) {
  const { pointerId, clientX: sx, clientY: sy } = e;
  const up = (ev) => {
    if (ev.pointerId !== pointerId) return;
    cleanup();
    if (Math.hypot(ev.clientX - sx, ev.clientY - sy) < 10) cb(ev);
  };
  const cancel = (ev) => ev.pointerId === pointerId && cleanup();
  const cleanup = () => {
    window.removeEventListener("pointerup", up);
    window.removeEventListener("pointercancel", cancel);
  };
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", cancel);
}

function onDragMove(e) {
  if (!dragCtx || e.pointerId !== dragCtx.pointerId) return;
  if (!dragCtx.started) {
    if (Math.hypot(e.clientX - dragCtx.sx, e.clientY - dragCtx.sy) < 6) return;
    dragCtx.started = true;
    dragCtx.ghost = document.createElement("img");
    dragCtx.ghost.className = "drag-ghost";
    dragCtx.ghost.src = dragCtx.iconSrc || "";
    document.body.appendChild(dragCtx.ghost);
  }
  dragCtx.ghost.style.left = `${e.clientX}px`;
  dragCtx.ghost.style.top = `${e.clientY}px`;
  document.querySelectorAll(".hex.drop-target").forEach((h) => h.classList.remove("drop-target"));
  const hex = document.elementFromPoint(e.clientX, e.clientY)?.closest(".hex");
  if (hex) hex.classList.add("drop-target");
}

function onDragCancel(e) {
  if (!dragCtx || e.pointerId !== dragCtx.pointerId) return;
  teardownDrag();
}

function onDragEnd(e) {
  if (!dragCtx || e.pointerId !== dragCtx.pointerId) return;
  const { kind, payload, started, onTap } = dragCtx;
  teardownDrag();
  if (!started) {
    onTap?.(e); // the pointer never moved: this was a tap/click on the source
    return;
  }

  const under = document.elementFromPoint(e.clientX, e.clientY);
  const hex = under?.closest(".hex");
  const onBoard = !!under?.closest(".hexboard");
  const at = hex && { side: hex.closest(".hexboard").dataset.side, key: cellKey(hex.dataset.r, hex.dataset.c) };

  if (kind === "pick-unit" && hex) {
    placeUnit(at.side, +hex.dataset.r, +hex.dataset.c, payload.apiName);
  } else if (kind === "pick-item" && hex) {
    equipItem(at.side, at.key, payload.apiName);
  } else if (kind === "board-unit") {
    if (hex) moveUnit(payload.from, at);
    else if (!onBoard) {
      delete state.boards[payload.from.side][payload.from.key];
      afterEdit();
    }
  } else if (kind === "board-item") {
    if (hex) moveItem(payload.from, at);
    else if (!onBoard) {
      const cell = state.boards[payload.from.side][payload.from.key];
      if (cell) {
        cell.items.splice(payload.from.idx, 1);
        afterEdit();
      }
    }
  }
}

function teardownDrag() {
  window.removeEventListener("pointermove", onDragMove);
  window.removeEventListener("pointerup", onDragEnd);
  window.removeEventListener("pointercancel", onDragCancel);
  dragCtx?.ghost?.remove();
  document.querySelectorAll(".hex.drop-target").forEach((h) => h.classList.remove("drop-target"));
  dragCtx = null;
}

// A tap on a board hex (touch/pen only — no hover): toggle the star selector
// on a placed unit; a tap on an empty hex closes any open selector.
function boardTap(ev, hex) {
  if (ev.pointerType === "mouse") return;
  const open = hex.classList.contains("stars-open");
  document.querySelectorAll(".hex.stars-open").forEach((h) => h.classList.remove("stars-open"));
  if (hex.classList.contains("occupied")) hex.classList.toggle("stars-open", !open);
}

function wireEvents() {
  // Kill the native HTML5 drag (image ghosting): all dragging is pointer-based.
  document.addEventListener("dragstart", (e) => e.preventDefault());

  for (const side of ["player", "opponent"]) {
    const board = $(`#board-${side}`);

    // Right-click an occupied hex removes the unit (desktop shortcut; on touch
    // a long-press fires contextmenu too, but there it starts a drag instead).
    board.addEventListener("contextmenu", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex) return;
      e.preventDefault();
      if (isTouchEvent()) return; // touch long-press is not a remove
      delete state.boards[side][cellKey(hex.dataset.r, hex.dataset.c)];
      afterEdit();
    });

    // All board interaction starts at pointerdown: occupied hexes become drag
    // sources (with the no-movement case handled as a tap), empty hexes only
    // watch for a tap. Item icons and the star selector have their own
    // handlers and are excluded here.
    board.addEventListener("pointerdown", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex || e.target.closest(".star-select")) return;
      const key = cellKey(hex.dataset.r, hex.dataset.c);
      const cell = state.boards[side][key];
      const tap = (ev) => boardTap(ev, hex);
      if (cell && !e.target.closest(".items")) {
        beginDrag(e, "board-unit", { from: { side, key } }, (unitsByApi[cell.unit] || {}).icon, tap);
      } else if (!cell) {
        watchTap(e, tap);
      }
    });
  }

  // Tabs
  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      $("#tab-units").classList.toggle("hidden", tab.dataset.tab !== "units");
      $("#tab-items").classList.toggle("hidden", tab.dataset.tab !== "items");
    })
  );

  $("#unit-search").addEventListener("input", renderUnitGrid);
  $("#item-search").addEventListener("input", renderItemGrid);
  $("#predict-btn").addEventListener("click", predict);
  $("#auto-predict").addEventListener("change", (e) => {
    state.autoPredict = e.target.checked;
    if (state.autoPredict) predict();
  });
  $("#random-board").addEventListener("click", loadRandomBoard);
  $("#clear-all").addEventListener("click", () => {
    state.boards = { player: {}, opponent: {} };
    afterEdit();
  });
  $("#swap-boards").addEventListener("click", () => {
    // Swapping perspectives rotates the matchup 180°: each unit keeps its
    // battlefield position (frontline stays next to the divider, left becomes
    // right), so the cells must be point-mirrored, not copied as-is.
    const mirror = (board) => {
      const out = {};
      for (const [key, cell] of Object.entries(board)) {
        const [r, c] = key.split("_").map(Number);
        out[cellKey(ROWS - 1 - r, COLS - 1 - c)] = cell;
      }
      return out;
    };
    state.boards = {
      player: mirror(state.boards.opponent),
      opponent: mirror(state.boards.player),
    };
    afterEdit();
  });
  document.querySelectorAll("[data-clear]").forEach((b) =>
    b.addEventListener("click", () => {
      state.boards[b.dataset.clear] = {};
      afterEdit();
    })
  );

  // Re-flow the hex grids to the new container width on resize (debounced).
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(layoutHexes, 120);
  });
}

/* ------------------------------------------------------------------ *
 * Pickers
 * ------------------------------------------------------------------ */
function buildCostFilter() {
  const wrap = $("#cost-filter");
  ["all", 1, 2, 3, 4, 5].forEach((c) => {
    const chip = document.createElement("div");
    chip.className = "cost-chip" + (c === "all" ? " active" : "");
    chip.textContent = c === "all" ? "All" : `${c}★`;
    chip.dataset.cost = c;
    chip.addEventListener("click", () => {
      wrap.querySelectorAll(".cost-chip").forEach((x) => x.classList.remove("active"));
      chip.classList.add("active");
      renderUnitGrid();
    });
    wrap.appendChild(chip);
  });
}

function renderUnitGrid() {
  const grid = $("#unit-grid");
  const q = $("#unit-search").value.toLowerCase();
  const costSel = $("#cost-filter .cost-chip.active")?.dataset.cost ?? "all";
  grid.innerHTML = "";
  catalog.units
    .filter((u) => u.hasIcon)
    .filter((u) => (costSel === "all" ? true : !u.special && String(u.cost) === costSel))
    .filter((u) => (u.name || u.apiName).toLowerCase().includes(q))
    .forEach((u) => grid.appendChild(makeUnitPick(u)));
}

function makeUnitPick(u) {
  const el = document.createElement("div");
  el.className = "pick";
  el.style.setProperty("--cost-color", unitCostVar(u));
  const traits = (u.traits || []).join(", ");
  el.title = u.special ? u.name : `${u.name} · ${u.cost}★${traits ? ` · ${traits}` : ""}`;
  const costBadge = u.special ? "" : `<span class="pick-cost">${u.cost}</span>`;
  el.innerHTML = `<img src="${u.icon}" alt="${u.name}" draggable="false" />${costBadge}`;
  wirePick(el, "unit", u.apiName, u.icon);
  return el;
}

// Picker entries: drag with a mouse; on touch, tapping a champion drops it on
// the first free cell of the player board, and tapping an item equips the
// first unit with a free item slot (drag afterwards to reposition/move). Taps
// are detected at the pointer level — no click listener — so they work on
// every touch pipeline.
function wirePick(el, kind, apiName, iconSrc) {
  el.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse") return beginDrag(e, `pick-${kind}`, { apiName }, iconSrc);
    watchTap(e, () => {
      if (kind === "unit") {
        const free = firstFreeCell("player");
        if (!free) return toast("Your board is full.");
        placeUnit("player", free.r, free.c, apiName);
      } else {
        const key = firstFreeItemSlot("player");
        if (!key) return toast("No unit with a free item slot.");
        equipItem("player", key, apiName);
      }
    });
  });
}

function buildItemFilter() {
  const wrap = $("#item-filter");
  const cats = [
    ["all", "All"],
    ["component", "Components"],
    ["normal", "Items"],
    ["emblem", "Emblems"],
    ["artifact", "Artifacts"],
    ["bilgewater", "Bilgewater"],
    ["radiant", "Radiant"],
  ];
  cats.forEach(([key, label], i) => {
    const chip = document.createElement("div");
    chip.className = "cost-chip" + (i === 0 ? " active" : "");
    chip.textContent = label;
    chip.dataset.cat = key;
    chip.addEventListener("click", () => {
      wrap.querySelectorAll(".cost-chip").forEach((x) => x.classList.remove("active"));
      chip.classList.add("active");
      renderItemGrid();
    });
    wrap.appendChild(chip);
  });
}

// The catalog carries the category (classified from Community Dragon tags);
// the regexes are only a fallback for a stale catalog.json.
function itemCategory(it) {
  if (it.category) return it.category;
  if (it.isEmblem) return "emblem";
  if (/Artifact|Ornn|Shimmerscale|TheDarkin/i.test(it.apiName)) return "artifact";
  if (/Radiant$/i.test(it.apiName)) return "radiant";
  if (/Bilgewater_/i.test(it.apiName)) return "bilgewater";
  return "normal";
}

function renderItemGrid() {
  const grid = $("#item-grid");
  const q = $("#item-search").value.toLowerCase();
  const cat = $("#item-filter .cost-chip.active")?.dataset.cat ?? "all";
  grid.innerHTML = "";
  catalog.items
    .filter((i) => i.hasIcon)
    .filter((i) => (cat === "all" ? true : itemCategory(i) === cat))
    .filter((i) => (i.name || i.apiName).toLowerCase().includes(q))
    .forEach((it) => {
      const el = document.createElement("div");
      el.className = "pick item";
      el.title = it.name;
      el.innerHTML = `<img src="${it.icon}" alt="${it.name}" draggable="false" />`;
      wirePick(el, "item", it.apiName, it.icon);
      grid.appendChild(el);
    });
}

/* ------------------------------------------------------------------ *
 * Trait computation (display only — mirrors the emblem-aware count)
 * ------------------------------------------------------------------ */
function computeTraits(side) {
  const counts = {};
  const seenUnits = new Set();
  for (const cell of Object.values(state.boards[side])) {
    // Each distinct champion contributes 1 to each of its traits.
    if (!seenUnits.has(cell.unit)) {
      seenUnits.add(cell.unit);
      (unitsByApi[cell.unit]?.traits || []).forEach((t) => (counts[t] = (counts[t] || 0) + 1));
    }
    // Emblem items add a bonus to their trait.
    for (const it of cell.items) {
      const trait = itemsByApi[it]?.emblemTrait;
      if (trait) counts[trait] = (counts[trait] || 0) + 1;
    }
  }
  const rows = [];
  for (const [name, count] of Object.entries(counts)) {
    const bps = traitsByName[name]?.breakpoints || [];
    const reached = bps.filter((b) => count >= b).length;
    // Next breakpoint to aim for; sticks to the last one once maxed out.
    const next = bps.find((b) => count < b) ?? bps[bps.length - 1] ?? count;
    rows.push({
      name,
      count,
      next,
      tierIndex: reached,
      maxTiers: bps.length,
      active: reached > 0,
      unique: !!traitsByName[name]?.unique || bps.length === 1,
    });
  }
  // Active first (unique traits on top, then by tier/count); inactive after.
  rows.sort(
    (a, b) =>
      b.active - a.active ||
      (b.active && b.unique) - (a.active && a.unique) ||
      b.tierIndex - a.tierIndex ||
      b.count - a.count ||
      a.name.localeCompare(b.name)
  );
  return rows;
}

function tierClass(t) {
  if (!t.active) return "tier-none";
  if (t.unique) return "tier-unique"; // unique traits render gold, not prismatic
  if (t.tierIndex >= t.maxTiers) return "tier-prism";
  const frac = t.tierIndex / t.maxTiers;
  if (frac >= 0.66) return "tier-gold";
  if (frac >= 0.33) return "tier-silver";
  return "tier-bronze";
}

function renderTraits() {
  for (const side of ["player", "opponent"]) {
    const ul = $(`#traits-${side}`);
    const rows = computeTraits(side);
    ul.innerHTML = "";
    if (rows.length === 0) {
      ul.innerHTML = `<li class="trait-empty">No traits yet</li>`;
      continue;
    }
    rows.forEach((t) => {
      const li = document.createElement("li");
      li.className = "trait-row " + tierClass(t);
      const icon = traitsByName[t.name]?.icon;
      li.innerHTML = `${icon ? `<img src="${icon}" alt="" />` : ""}
        <span class="trait-name">${t.name}</span>
        <span class="trait-count">${t.count}/${t.next}</span>`;
      ul.appendChild(li);
    });
  }
}

/* ------------------------------------------------------------------ *
 * Model select + prediction
 * ------------------------------------------------------------------ */
async function buildModelSelect() {
  const wrap = $("#model-select");
  let avail = { vit: true, cnn: true, xgboost: true };
  try {
    const info = await fetch("api/models").then((r) => r.json());
    avail = info.available;
    state.loadedModels = new Set(info.loaded || []);
  } catch {}
  const models = [
    ["vit", "ViT"],
    ["cnn", "CNN"],
    ["xgboost", "XGBoost"],
  ];
  // Default to the best available model (ViT > CNN > XGBoost).
  state.model = models.map((m) => m[0]).find((k) => avail[k]) || "vit";
  wrap.innerHTML = "";
  models.forEach(([key, label]) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.disabled = !avail[key];
    btn.classList.toggle("active", key === state.model);
    btn.title = avail[key] ? `Predict with ${label}` : `${label} weights not found on server`;
    btn.addEventListener("click", () => {
      state.model = key;
      wrap.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (state.autoPredict) predict();
    });
    wrap.appendChild(btn);
  });
}

/* ------------------------------------------------------------------ *
 * Pre-saved real boards (sample_boards.json, extracted from actual
 * games with `trp extract-sample-boards`) — lets the models be tested
 * on realistic positions in one click.
 * ------------------------------------------------------------------ */
let sampleBoards = null; // lazy-fetched on first use
let lastSampleIdx = -1;

// Inverse of serializeBoard: sample units are stored in each side's own frame
// (row 0 = frontline), so the opponent side is point-mirrored back into
// screen coordinates.
function deserializeBoard(side, units) {
  const flip = side === "opponent";
  const out = {};
  for (const u of units) {
    const r = flip ? ROWS - 1 - u.row : u.row;
    const c = flip ? COLS - 1 - u.col : u.col;
    out[cellKey(r, c)] = { unit: u.unit, tier: u.tier, items: [...(u.items || [])] };
  }
  return out;
}

async function loadRandomBoard() {
  if (!sampleBoards) {
    try {
      sampleBoards = (await fetch("data/sample_boards.json").then((r) => r.json())).boards;
    } catch {
      return toast("No sample boards found — run `trp extract-sample-boards`.");
    }
  }
  if (!sampleBoards || sampleBoards.length === 0) {
    return toast("No sample boards found — run `trp extract-sample-boards`.");
  }
  let idx = Math.floor(Math.random() * sampleBoards.length);
  if (sampleBoards.length > 1 && idx === lastSampleIdx) {
    idx = (idx + 1) % sampleBoards.length; // don't serve the same board twice in a row
  }
  lastSampleIdx = idx;
  const b = sampleBoards[idx];
  state.boards = {
    player: deserializeBoard("player", b.player),
    opponent: deserializeBoard("opponent", b.opponent),
  };
  afterEdit();
}

// The API expects each board in its OWN frame (row 0 = that side's frontline,
// matching the raw-data "A1".."D7" locs). The player board is displayed in its
// own frame already (top row = frontline, next to the divider), but the
// opponent board is displayed from the player's point of view — rotated 180° —
// so its screen coordinates must be point-mirrored back into its frame.
function serializeBoard(side) {
  const flip = side === "opponent";
  return Object.entries(state.boards[side]).map(([key, cell]) => {
    const [r, c] = key.split("_").map(Number);
    return {
      unit: cell.unit,
      tier: cell.tier,
      items: cell.items,
      row: flip ? ROWS - 1 - r : r,
      col: flip ? COLS - 1 - c : c,
    };
  });
}

async function predict() {
  const player = serializeBoard("player");
  const opponent = serializeBoard("opponent");
  if (player.length === 0 && opponent.length === 0) {
    return toast("Place some units first.");
  }
  const btn = $("#predict-btn");
  const model = state.model;
  const firstLoad = !state.loadedModels.has(model);
  btn.disabled = true;
  if (firstLoad) {
    // The server loads model weights lazily: the first prediction with a
    // backend also pays the load cost, so tell the user what is going on.
    btn.textContent = "Loading model…";
    toast(`Loading the ${model.toUpperCase()} model — the first prediction can take a moment.`);
  } else {
    btn.textContent = "Predicting…";
  }
  try {
    const res = await fetch("api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, player, opponent }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    state.loadedModels.add(model);
    showResult(await res.json());
  } catch (e) {
    toast("Prediction failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict outcome";
  }
}

function showResult(res) {
  const pct = res.win_probability * 100;
  const win = res.win_probability >= 0.5;
  $("#result").classList.remove("hidden");
  const verdict = $("#result-verdict");
  verdict.textContent = win ? "Victory" : "Defeat";
  verdict.className = "verdict " + (win ? "win" : "loss");
  $("#result-model").textContent = res.model;
  $("#winbar-fill").style.width = `${pct}%`;
  $("#winbar-label").textContent = `${pct.toFixed(1)}%`;
}

/* ------------------------------------------------------------------ *
 * Helpers
 * ------------------------------------------------------------------ */
let toastTimer = null;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 2600);
}

function afterEdit() {
  renderBoards();
  renderTraits();
  if (state.autoPredict) predict();
}

window.__tft = state; // exposed for debugging / automated tests
boot();

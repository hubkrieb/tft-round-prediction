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

const state = {
  boards: { player: {}, opponent: {} }, // key "r_c" -> {unit, tier, items:[]}
  model: "vit",
  autoPredict: false,
};

let catalog = null;
let unitsByApi = {};
let itemsByApi = {};
let traitsByName = {};

const $ = (sel) => document.querySelector(sel);
const cellKey = (r, c) => `${r}_${c}`;
const costVar = (cost) => `var(${COST_COLORS[Math.min(cost || 1, 6)]})`;

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
      hex.draggable = !!cell;
      if (!cell) return;
      const u = unitsByApi[cell.unit] || {};
      const cost = u.cost || 1;
      hex.style.setProperty("--cost-color", costVar(cost));

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
        s.addEventListener("click", (e) => {
          e.stopPropagation();
          cell.tier = t;
          afterEdit();
        });
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
        im.title = `${(itemsByApi[it] || {}).name || it} — right-click to remove`;
        im.draggable = false;
        im.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          e.stopPropagation(); // don't also remove the unit
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

function wireEvents() {
  // Hex clicks: place selected unit, or cycle star on occupied.
  for (const side of ["player", "opponent"]) {
    const board = $(`#board-${side}`);

    // Right-click an occupied hex to remove the unit. (Right-clicking an item
    // icon removes just that item; its handler stops propagation.) Star level is
    // set via the hover star selector; placement is drag-and-drop only.
    board.addEventListener("contextmenu", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex) return;
      e.preventDefault();
      delete state.boards[side][cellKey(hex.dataset.r, hex.dataset.c)];
      afterEdit();
    });

    // Drag & drop targets
    board.addEventListener("dragover", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex) return;
      e.preventDefault();
      hex.classList.add("drop-target");
    });
    board.addEventListener("dragleave", (e) => {
      const hex = e.target.closest(".hex");
      if (hex) hex.classList.remove("drop-target");
    });
    board.addEventListener("drop", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex) return;
      e.preventDefault();
      hex.classList.remove("drop-target");
      handleDrop(side, +hex.dataset.r, +hex.dataset.c, e.dataTransfer);
    });

    // Dragging a placed unit to move it.
    board.addEventListener("dragstart", (e) => {
      const hex = e.target.closest(".hex");
      if (!hex || !hex.classList.contains("occupied")) return;
      e.dataTransfer.setData(
        "application/json",
        JSON.stringify({ kind: "move", from: { side, key: cellKey(hex.dataset.r, hex.dataset.c) } })
      );
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
  $("#clear-all").addEventListener("click", () => {
    state.boards = { player: {}, opponent: {} };
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

function handleDrop(side, r, c, dt) {
  let payload;
  try {
    payload = JSON.parse(dt.getData("application/json"));
  } catch {
    return;
  }
  const key = cellKey(r, c);
  if (payload.kind === "unit") {
    placeUnit(side, r, c, payload.apiName);
  } else if (payload.kind === "move") {
    const src = state.boards[payload.from.side][payload.from.key];
    if (!src) return;
    const dst = state.boards[side][key];
    state.boards[side][key] = src;
    if (dst) state.boards[payload.from.side][payload.from.key] = dst;
    else delete state.boards[payload.from.side][payload.from.key];
    afterEdit();
  } else if (payload.kind === "item") {
    const cell = state.boards[side][key];
    if (!cell) return toast("Drop items onto a champion.");
    if (cell.items.length >= MAX_ITEMS) return toast("Unit already has 3 items.");
    cell.items.push(payload.apiName);
    afterEdit();
  }
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
    .filter((u) => (costSel === "all" ? true : String(u.cost) === costSel))
    .filter((u) => (u.name || u.apiName).toLowerCase().includes(q))
    .forEach((u) => grid.appendChild(makeUnitPick(u)));
}

function makeUnitPick(u) {
  const el = document.createElement("div");
  el.className = "pick";
  el.style.setProperty("--cost-color", costVar(u.cost));
  el.title = `${u.name} · ${u.cost}★ · ${(u.traits || []).join(", ")}`;
  el.draggable = true;
  el.innerHTML = `<img src="${u.icon}" alt="${u.name}" draggable="false" /><span class="pick-cost">${u.cost}</span>`;
  el.addEventListener("dragstart", (e) =>
    e.dataTransfer.setData("application/json", JSON.stringify({ kind: "unit", apiName: u.apiName }))
  );
  return el;
}

function buildItemFilter() {
  const wrap = $("#item-filter");
  const cats = [
    ["all", "All"],
    ["normal", "Items"],
    ["emblem", "Emblems"],
    ["artifact", "Artifacts"],
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

function itemCategory(it) {
  if (it.isEmblem) return "emblem";
  if (/Artifact/i.test(it.apiName)) return "artifact";
  if (/Radiant/i.test(it.apiName)) return "radiant";
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
      el.draggable = true;
      el.innerHTML = `<img src="${it.icon}" alt="${it.name}" draggable="false" />`;
      el.addEventListener("dragstart", (e) =>
        e.dataTransfer.setData("application/json", JSON.stringify({ kind: "item", apiName: it.apiName }))
      );
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
  const active = [];
  for (const [name, count] of Object.entries(counts)) {
    const bps = traitsByName[name]?.breakpoints || [];
    const reached = bps.filter((b) => count >= b);
    if (reached.length > 0) {
      active.push({ name, count, tierIndex: reached.length, maxTiers: bps.length });
    }
  }
  active.sort((a, b) => b.tierIndex - a.tierIndex || b.count - a.count);
  return active;
}

function tierClass(t) {
  const frac = t.tierIndex / t.maxTiers;
  if (t.tierIndex >= t.maxTiers) return "tier-prism";
  if (frac >= 0.66) return "tier-gold";
  if (frac >= 0.33) return "tier-silver";
  return "tier-bronze";
}

function renderTraits() {
  for (const side of ["player", "opponent"]) {
    const ul = $(`#traits-${side}`);
    const active = computeTraits(side);
    ul.innerHTML = "";
    if (active.length === 0) {
      ul.innerHTML = `<li class="trait-empty">No active traits</li>`;
      continue;
    }
    active.forEach((t) => {
      const li = document.createElement("li");
      li.className = "trait-row " + tierClass(t);
      const icon = traitsByName[t.name]?.icon;
      li.innerHTML = `${icon ? `<img src="${icon}" alt="" />` : ""}
        <span class="trait-name">${t.name}</span>
        <span class="trait-count">${t.count}</span>`;
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

function serializeBoard(side) {
  return Object.entries(state.boards[side]).map(([key, cell]) => {
    const [r, c] = key.split("_").map(Number);
    return { unit: cell.unit, tier: cell.tier, items: cell.items, row: r, col: c };
  });
}

async function predict() {
  const player = serializeBoard("player");
  const opponent = serializeBoard("opponent");
  if (player.length === 0 && opponent.length === 0) {
    return toast("Place some units first.");
  }
  const btn = $("#predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  try {
    const res = await fetch("api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: state.model, player, opponent }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
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

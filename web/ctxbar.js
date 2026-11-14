// Context 面板：勾选框由文档里的 `contexts` 生成，规则按 `kind` 套用。
//   trigger → 任一选中即触发
//   veto    → 命中则撤销当天信号
//   boost   → 命中则补一个信号
// 三个页面共用这一份。页面只要提供文档和「重画箭头」的回调。
(function () {
  const REQUIRED = (data) => data.contexts.filter((item) => item.required).map((item) => item.key);
  const DEFAULT = (data) => ["composite", "cal_cpi", "cal_nfp", "fomc_day", "news_bearish"]
    .filter((key) => data.contexts.some((item) => item.key === key));

  let data = null;
  let selected = new Set();
  let onFlags = () => {};

  const pct = (value) => (value === null || value === undefined ? "—" : (value * 100).toFixed(1) + "%");

  function applyContexts(picked) {
    const c = data.components;
    const active = new Set(picked);
    data.contexts.filter((item) => item.required).forEach((item) => active.add(item.key));
    const contexts = data.contexts.filter((item) => active.has(item.key));
    const triggers = contexts.filter((item) => item.kind === "trigger");
    const vetoes = contexts.filter((item) => item.kind === "veto");
    const boosts = contexts.filter((item) => item.kind === "boost");
    const flags = new Array(data.time.length).fill(0);
    for (let i = 0; i < data.time.length; i += 1) {
      let flag = 0;
      for (const item of triggers) if (c[item.key][i]) { flag = 1; break; }
      if (!flag) for (const item of boosts) if (c[item.key][i]) { flag = 1; break; }
      if (flag) {
        for (const item of vetoes) {
          const value = c[item.key][i];
          const hit = item.key === "riskoff_hits" ? value >= 2 : Boolean(value);
          if (hit) { flag = 0; break; }
        }
      }
      flags[i] = flag;
    }
    return flags;
  }

  function summarise(flags) {
    let signals = 0, hits = 0, hit3 = 0, hit45 = 0, held = 0, heldHits = 0;
    const holdout = data.summary.holdout_from || "2015-01-01";
    data.time.forEach((day, index) => {
      const outcome = data.outcome[index];
      if (outcome === null || outcome === undefined || !flags[index]) return;
      signals += 1;
      hits += outcome;
      if (data.hit3[index] === 1) hit3 += 1;
      else if (outcome === 1) hit45 += 1;
      if (day >= holdout) { held += 1; heldHits += outcome; }
    });
    return {
      signals,
      precision: signals ? hits / signals : null,
      precision3: signals ? hit3 / signals : null,
      precision45: signals ? hit45 / signals : null,
      holdoutSignals: held,
      holdoutPrecision: held ? heldHits / held : null,
    };
  }

  function row(table, values, header) {
    const tr = document.createElement("tr");
    values.forEach((value, index) => {
      const cell = document.createElement(header || index === 0 ? "th" : "td");
      if (header) cell.scope = "col";
      else if (index === 0) cell.scope = "row";
      cell.textContent = value;
      tr.appendChild(cell);
    });
    table.appendChild(tr);
  }

  function render() {
    const flags = applyContexts(selected);
    onFlags(flags);
    const now = summarise(flags);
    const baseline = summarise(applyContexts(new Set(["composite"])));

    const table = document.querySelector("#ctx-table");
    table.textContent = "";
    row(table, ["", "This selection", "Composite only"], true);
    row(table, ["Signals", String(now.signals), String(baseline.signals)]);
    row(table, ["Hit within 3 sessions", pct(now.precision3), pct(baseline.precision3)]);
    row(table, ["Hit only on day 4-5", pct(now.precision45), pct(baseline.precision45)]);
    row(table, ["Precision (5 sessions)", pct(now.precision), pct(baseline.precision)]);
    row(table, ["Out-of-sample signals", String(now.holdoutSignals), String(baseline.holdoutSignals)]);
    row(table, ["Out-of-sample precision", pct(now.holdoutPrecision), pct(baseline.holdoutPrecision)]);

    const rule = document.querySelector("#rule");
    if (rule) {
      rule.textContent = now.signals + " signals · 1-3 days " + pct(now.precision3) +
        " · 4-5 days " + pct(now.precision45) +
        " · " + selected.size + " of " + data.contexts.length + " contexts";
    }

    const active = data.contexts.filter((item) => selected.has(item.key) || item.required);
    document.querySelector("#ctx-status").textContent =
      active.length + "/" + data.contexts.length + " contexts · " + now.signals + " signals";
    document.querySelector("#ctx-note").textContent = active.length
      ? "Using: " + active.map((item) => item.label).join(" · ") +
        " | any random day " + pct(data.summary.base_rate ?? (data.summary.all || {}).base_rate) +
        (now.signals < 60 ? " | fewer than 60 signals: treat the precision as noise" : "")
      : "Nothing selected: no signals at all.";
  }

  function buildPicker() {
    const holder = document.querySelector("#ctx-picker");
    holder.textContent = "";
    data.contexts.forEach((item) => {
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = item.required || selected.has(item.key);
      box.disabled = Boolean(item.required);
      if (item.required) label.classList.add("required");
      box.onchange = () => {
        if (box.checked) selected.add(item.key); else selected.delete(item.key);
        render();
      };
      const text = document.createElement("span");
      text.innerHTML = item.label + ' <span class="kind">· ' + item.kind +
        (item.required ? " · always on (base trigger)" : "") + " · " + item.note + "</span>";
      label.appendChild(box);
      label.appendChild(text);
      holder.appendChild(label);
    });
  }

  window.initCtxbar = function (document_, callback) {
    data = document_;
    onFlags = callback;
    selected = new Set(DEFAULT(data));
    REQUIRED(data).forEach((key) => selected.add(key));
    buildPicker();
    document.querySelector("#ctx-reset").onclick = () => {
      selected = new Set(["composite"]);
      buildPicker();
      render();
    };
    document.querySelector("#ctx-all").onclick = () => {
      selected = new Set(data.contexts.map((item) => item.key));
      buildPicker();
      render();
    };
    document.querySelector("#ctx-toggle").onclick = (event) => {
      const panel = document.querySelector(".ctxbar");
      const hidden = panel.classList.toggle("collapsed");
      event.target.textContent = hidden ? "\u25BC" : "\u25B2";
      event.target.title = hidden ? "Show the factor panel" : "Collapse the factor panel";
      event.target.setAttribute("aria-expanded", String(!hidden));
    };
    render();
  };
})();

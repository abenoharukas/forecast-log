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
    const statusElem = document.querySelector("#ctx-status");
    if (statusElem) {
      statusElem.textContent = active.length + "/" + data.contexts.length + " contexts · " + now.signals + " signals";
    }
    const noteElem = document.querySelector("#ctx-note");
    if (noteElem) {
      noteElem.textContent = active.length
        ? "Using: " + active.map((item) => item.label).join(" · ") +
          " | any random day " + pct(data.summary.base_rate ?? (data.summary.all || {}).base_rate) +
          (now.signals < 60 ? " | fewer than 60 signals: treat the precision as noise" : "")
        : "Nothing selected: no signals at all.";
    }

    const panel = document.querySelector(".ctxbar");
    const isOpen = panel && !panel.classList.contains("collapsed");
    updateToggleUI(isOpen);
  }

  function updateToggleUI(isOpen) {
    const toggleBtn = document.querySelector("#ctx-toggle");
    if (!toggleBtn || !data) return;
    const active = data.contexts.filter((item) => selected.has(item.key) || item.required);
    toggleBtn.textContent = "factors (" + active.length + "/" + data.contexts.length + ") " + (isOpen ? "\u25B4" : "\u25BE");
    toggleBtn.setAttribute("aria-expanded", String(isOpen));
    toggleBtn.classList.toggle("active", isOpen);
    toggleBtn.title = isOpen ? "Close the factors panel" : "Configure model context factors";
  }

  function setOpen(open) {
    const panel = document.querySelector(".ctxbar");
    if (!panel) return;
    panel.classList.toggle("collapsed", !open);
    updateToggleUI(open);
  }

  function buildPicker() {
    const holder = document.querySelector("#ctx-picker");
    holder.textContent = "";

    function option(item) {
      const label = document.createElement("label");
      label.className = "ctx-item";
      const box = document.createElement("input");
      const required = Boolean(item.required);
      box.type = item.group ? "radio" : "checkbox";
      if (item.group) box.name = "group-" + item.group;
      box.checked = required || selected.has(item.key);
      box.disabled = required;
      if (required) label.classList.add("required");
      box.onchange = () => {
        if (item.group) {
          groups[item.group].forEach((key) => selected.delete(key));
          if (box.checked) selected.add(item.key);
          buildPicker();
        } else if (box.checked) {
          selected.add(item.key);
        } else {
          selected.delete(item.key);
        }
        render();
      };
      const text = document.createElement("span");
      text.innerHTML = item.label + ' <span class="kind">· ' + item.kind +
        (required ? ' <span class="badge">always on</span>' : "") + " · " + item.note + "</span>";
      label.appendChild(box);
      label.appendChild(text);
      return label;
    }

    // 同一个 group 里的选项互斥：新闻源只能选一个，或一个都不选。
    const groups = {};
    data.contexts.filter((item) => item.group).forEach((item) => {
      (groups[item.group] = groups[item.group] || []).push(item.key);
    });

    const standalone = document.createElement("div");
    standalone.className = "ctx-standalone";
    data.contexts.filter((item) => !item.group).forEach((item) => standalone.appendChild(option(item)));
    holder.appendChild(standalone);

    Object.entries(groups).forEach(([name, keys]) => {
      const members = data.contexts.filter((item) => item.group === name);
      const field = document.createElement("fieldset");
      field.className = "ctx-group";
      const legend = document.createElement("legend");
      legend.textContent = (members[0].group_label || name) + " · 三选一";
      field.appendChild(legend);

      const offLabel = document.createElement("label");
      offLabel.className = "ctx-item";
      const offBox = document.createElement("input");
      offBox.type = "radio";
      offBox.name = "group-" + name;
      offBox.checked = keys.every((key) => !selected.has(key));
      offBox.onchange = () => {
        keys.forEach((key) => selected.delete(key));
        buildPicker();
        render();
      };
      const offText = document.createElement("span");
      offText.innerHTML = 'None <span class="kind">· off · 不撤任何信号</span>';
      offLabel.appendChild(offBox);
      offLabel.appendChild(offText);
      field.appendChild(offLabel);

      members.forEach((item) => field.appendChild(option(item)));
      holder.appendChild(field);
    });
  }

  window.initCtxbar = function (document_, callback) {
    data = document_;
    onFlags = callback;
    selected = new Set(DEFAULT(data));
    REQUIRED(data).forEach((key) => selected.add(key));
    buildPicker();

    const resetBtn = document.querySelector("#ctx-reset");
    if (resetBtn) {
      resetBtn.onclick = () => {
        selected = new Set(["composite"]);
        buildPicker();
        render();
      };
    }

    const allBtn = document.querySelector("#ctx-all");
    if (allBtn) {
      allBtn.onclick = () => {
        selected = new Set(data.contexts.map((item) => item.key));
        buildPicker();
        render();
      };
    }

    const toggleBtn = document.querySelector("#ctx-toggle");
    if (toggleBtn) {
      toggleBtn.onclick = (e) => {
        e.stopPropagation();
        const panel = document.querySelector(".ctxbar");
        const isOpen = panel && !panel.classList.contains("collapsed");
        setOpen(!isOpen);
      };
    }

    const closeBtn = document.querySelector("#ctx-close");
    if (closeBtn) {
      closeBtn.onclick = (e) => {
        e.stopPropagation();
        setOpen(false);
      };
    }

    const ruleElem = document.querySelector("#rule");
    if (ruleElem) {
      ruleElem.onclick = (e) => {
        e.stopPropagation();
        const panel = document.querySelector(".ctxbar");
        const isOpen = panel && !panel.classList.contains("collapsed");
        setOpen(!isOpen);
      };
    }

    // Light dismiss: clicking outside the floating panel closes it
    document.addEventListener("click", (e) => {
      const panel = document.querySelector(".ctxbar");
      if (!panel || panel.classList.contains("collapsed")) return;
      if (panel.contains(e.target)) return;
      if (toggleBtn && toggleBtn.contains(e.target)) return;
      if (ruleElem && ruleElem.contains(e.target)) return;
      setOpen(false);
    });

    // Escape key dismisses the panel
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const panel = document.querySelector(".ctxbar");
        if (panel && !panel.classList.contains("collapsed")) {
          setOpen(false);
          if (toggleBtn) toggleBtn.focus();
        }
      }
    });

    render();
  };
})();

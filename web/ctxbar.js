// Context 面板：勾选框由文档里的 `contexts` 动态生成，规则按 `kind` 与用户选择的判断函数实时计算。
//   trigger → 触发条件（支持任一命中 OR / 多因子共振 ≥2 / 全票通过 AND）
//   boost   → 增强信号（在未触发时按触发逻辑补信号）
//   veto    → 否决条件（支持一票否决 OR / 双重确认否决 ≥2）
// 统一页面共用这一份，按 marketKey 记忆每个市场的勾选与判断规则配置。
(function () {
  const DEFAULT_KEYS = ["composite", "cal_cpi", "cal_nfp", "fomc_day", "news_bearish"];
  const marketStates = new Map();

  let data = null;
  let currentMarket = "default";
  let selected = new Set();
  let ruleConfig = { triggerMode: "any", vetoMode: "any" };
  let onFlags = () => {};
  let eventsBound = false;

  const pct = (value) => (value === null || value === undefined ? "—" : (value * 100).toFixed(1) + "%");
  const requiredKeys = (doc) => doc.contexts.filter((item) => item.required).map((item) => item.key);
  const defaultKeys = (doc) => {
    const keys = DEFAULT_KEYS.filter((key) => doc.contexts.some((item) => item.key === key));
    requiredKeys(doc).forEach((key) => { if (!keys.includes(key)) keys.push(key); });
    return keys;
  };

  function isHit(item, rawValue) {
    if (item.threshold !== undefined && item.threshold !== null) {
      return Number(rawValue) >= Number(item.threshold);
    }
    return item.key === "riskoff_hits" ? Number(rawValue) >= 2 : Boolean(rawValue);
  }

  function applyContexts(picked, rules) {
    const c = data.components;
    const active = new Set(picked);
    requiredKeys(data).forEach((key) => active.add(key));

    const contexts = data.contexts.filter((item) => active.has(item.key) && Array.isArray(c[item.key]));
    const triggers = contexts.filter((item) => item.kind === "trigger");
    const boosts = contexts.filter((item) => item.kind === "boost");
    const vetoes = contexts.filter((item) => item.kind === "veto");

    const triggerMode = (rules && rules.triggerMode) || "any";
    const vetoMode = (rules && rules.vetoMode) || "any";
    const flags = new Array(data.time.length).fill(0);

    for (let i = 0; i < data.time.length; i += 1) {
      let flag = 0;
      if (triggerMode === "all") {
        flag = triggers.length > 0 && triggers.every((item) => isHit(item, c[item.key][i])) ? 1 : 0;
        if (!flag && boosts.length > 0) {
          for (const item of boosts) {
            if (isHit(item, c[item.key][i])) { flag = 1; break; }
          }
        }
      } else if (triggerMode === "consensus") {
        let count = 0;
        for (const item of triggers) if (isHit(item, c[item.key][i])) count += 1;
        for (const item of boosts) if (isHit(item, c[item.key][i])) count += 1;
        const needed = Math.min(2, triggers.length + boosts.length);
        flag = needed > 0 && count >= needed ? 1 : 0;
      } else {
        for (const item of triggers) {
          if (isHit(item, c[item.key][i])) { flag = 1; break; }
        }
        if (!flag) {
          for (const item of boosts) {
            if (isHit(item, c[item.key][i])) { flag = 1; break; }
          }
        }
      }

      if (flag && vetoes.length > 0) {
        if (vetoMode === "consensus" && vetoes.length >= 2) {
          let vetoCount = 0;
          for (const item of vetoes) {
            if (isHit(item, c[item.key][i])) vetoCount += 1;
          }
          if (vetoCount >= 2) flag = 0;
        } else {
          for (const item of vetoes) {
            if (isHit(item, c[item.key][i])) { flag = 0; break; }
          }
        }
      }
      flags[i] = flag;
    }
    return flags;
  }

  function applySellContexts(picked, buyFlags) {
    const c = data.components || {};
    const baseSell = Array.isArray(c.sell_composite)
      ? c.sell_composite
      : (Array.isArray(data.sell_flag) ? data.sell_flag : null);
    const sellFlags = new Array(data.time.length).fill(0);
    if (!baseSell) return sellFlags;

    const active = new Set(picked);
    const useFomc = active.has("fomc_day") && Array.isArray(c.fomc_day);
    const useBullVeto = active.has("news_bullish") && Array.isArray(c.news_bullish);

    for (let i = 0; i < data.time.length; i += 1) {
      if (buyFlags && buyFlags[i]) continue;
      let flag = baseSell[i] ? 1 : 0;
      if (flag && useFomc && c.fomc_day[i]) flag = 0;
      if (flag && useBullVeto && c.news_bullish[i]) flag = 0;
      sellFlags[i] = flag;
    }
    return sellFlags;
  }

  function summarise(flags) {
    let signals = 0, hits = 0, hit3 = 0, hit45 = 0, held = 0, heldHits = 0;
    const holdout = (data.summary && data.summary.holdout_from) || "2015-01-01";
    data.time.forEach((day, index) => {
      const outcome = data.outcome[index];
      if (outcome === null || outcome === undefined || !flags[index]) return;
      signals += 1;
      hits += outcome;
      if (data.hit3 && data.hit3[index] === 1) hit3 += 1;
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

  function summariseSell(sellFlags) {
    let signals = 0, hits = 0, hit3 = 0, hit45 = 0, held = 0, heldHits = 0;
    const holdout = (data.summary && data.summary.holdout_from) || "2015-01-01";
    data.time.forEach((day, index) => {
      if (!sellFlags[index]) return;
      const base = data.base_close ? data.base_close[index] : null;
      const c5 = data.close5 ? data.close5[index] : null;
      const c3 = data.close3 ? data.close3[index] : null;
      if (base === null || base === undefined || c5 === null || c5 === undefined) return;
      const down5 = c5 < base ? 1 : 0;
      const down3 = c3 !== null && c3 !== undefined && c3 < base ? 1 : 0;
      signals += 1;
      hits += down5;
      if (down3) hit3 += 1;
      else if (down5) hit45 += 1;
      if (day >= holdout) { held += 1; heldHits += down5; }
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

  function summariseActual() {
    let totalUp = 0, up3 = 0, up45 = 0, up5 = 0, heldUp = 0, heldUp5 = 0;
    let totalDown = 0, down3 = 0, down45 = 0, down5 = 0, heldDown = 0, heldDown5 = 0;
    const holdout = (data.summary && data.summary.holdout_from) || "2015-01-01";
    data.time.forEach((day, index) => {
      const outcome = data.outcome[index];
      if (outcome !== null && outcome !== undefined) {
        totalUp += 1;
        up5 += outcome;
        if (data.hit3 && data.hit3[index] === 1) up3 += 1;
        else if (outcome === 1) up45 += 1;
        if (day >= holdout) { heldUp += 1; heldUp5 += outcome; }
      }
      const base = data.base_close ? data.base_close[index] : null;
      const c5 = data.close5 ? data.close5[index] : null;
      const c3 = data.close3 ? data.close3[index] : null;
      if (base !== null && base !== undefined && c5 !== null && c5 !== undefined) {
        const d5 = c5 < base ? 1 : 0;
        const d3 = c3 !== null && c3 !== undefined && c3 < base ? 1 : 0;
        totalDown += 1;
        down5 += d5;
        if (d3) down3 += 1;
        else if (d5) down45 += 1;
        if (day >= holdout) { heldDown += 1; heldDown5 += d5; }
      }
    });
    return {
      up: {
        total: totalUp,
        precision: totalUp ? up5 / totalUp : null,
        precision3: totalUp ? up3 / totalUp : null,
        precision45: totalUp ? up45 / totalUp : null,
        holdoutPrecision: heldUp ? heldUp5 / heldUp : null,
      },
      down: {
        total: totalDown,
        precision: totalDown ? down5 / totalDown : null,
        precision3: totalDown ? down3 / totalDown : null,
        precision45: totalDown ? down45 / totalDown : null,
        holdoutPrecision: heldDown ? heldDown5 / heldDown : null,
      },
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

  function saveMarketState() {
    marketStates.set(currentMarket, {
      selected: new Set(selected),
      ruleConfig: { ...ruleConfig },
    });
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

  function render() {
    if (!data) return;
    saveMarketState();
    const flags = applyContexts(selected, ruleConfig);
    const sellFlags = applySellContexts(selected, flags);
    onFlags(flags, sellFlags);
    const now = summarise(flags);
    const sellNow = summariseSell(sellFlags);
    const actual = summariseActual();
    const latestIdx = data.time.length - 1;
    const latestFlag = flags[latestIdx];
    const latestSellFlag = sellFlags[latestIdx];

    const table = document.querySelector("#ctx-table");
    if (table) {
      table.textContent = "";
      row(table, ["", "预测上涨 (BUY)", "真实上涨 (基率)", "预测下跌 (SELL)", "真实下跌 (基率)"], true);
      row(table, ["样本 / 信号数", String(now.signals), String(actual.up.total), String(sellNow.signals), String(actual.down.total)]);
      row(table, ["1–3d 命中率", pct(now.precision3), pct(actual.up.precision3), pct(sellNow.precision3), pct(actual.down.precision3)]);
      row(table, ["4–5d 命中率", pct(now.precision45), pct(actual.up.precision45), pct(sellNow.precision45), pct(actual.down.precision45)]);
      row(table, ["5d 收盘命中率", pct(now.precision), pct(actual.up.precision), pct(sellNow.precision), pct(actual.down.precision)]);
      row(table, ["2015+ 样本外", pct(now.holdoutPrecision) + " (" + now.holdoutSignals + ")", pct(actual.up.holdoutPrecision), pct(sellNow.holdoutPrecision) + " (" + sellNow.holdoutSignals + ")", pct(actual.down.holdoutPrecision)]);
      row(table, [
        "最新交易日",
        latestFlag ? "BUY" : "—",
        "—",
        latestSellFlag ? "SELL" : "—",
        "—",
      ]);
    }

    const active = data.contexts.filter((item) => selected.has(item.key) || item.required);
    const starts = active.map((item) => item.available_from || "0000-00-00").sort();
    const effectiveFrom = starts.length ? starts[starts.length - 1] : "—";

    const ruleLabel =
      (ruleConfig.triggerMode === "consensus" ? "consensus ≥2" : ruleConfig.triggerMode === "all" ? "strict AND" : "any OR") +
      (ruleConfig.vetoMode === "consensus" ? " · veto ≥2" : "");

    const rule = document.querySelector("#rule");
    if (rule) {
      rule.textContent = "";
      const buyLine = document.createElement("span");
      buyLine.className = "rule-line";
      buyLine.textContent =
        "▲ 预测上涨 " + now.signals +
        " · 1-3d " + pct(now.precision3) + " vs 真实 " + pct(actual.up.precision3) +
        " · 4-5d " + pct(now.precision45) + " vs 真实 " + pct(actual.up.precision45);
      const sellLine = document.createElement("span");
      sellLine.className = "rule-line";
      sellLine.textContent =
        "▼ 预测下跌 " + sellNow.signals +
        " · 1-3d " + pct(sellNow.precision3) + " vs 真实 " + pct(actual.down.precision3) +
        " · 4-5d " + pct(sellNow.precision45) + " vs 真实 " + pct(actual.down.precision45) +
        " · " + active.length + "/" + data.contexts.length + " factors (" + ruleLabel + ")";
      rule.appendChild(buyLine);
      rule.appendChild(sellLine);
    }

    const statusElem = document.querySelector("#ctx-status");
    if (statusElem) {
      statusElem.textContent = active.length + "/" + data.contexts.length + " contexts · BUY " +
        now.signals + " / SELL " + sellNow.signals + " · from " + effectiveFrom;
    }

    const baseRate = data.summary.base_rate ?? (data.summary.all || {}).base_rate;
    const noteElem = document.querySelector("#ctx-note");
    if (noteElem) {
      noteElem.textContent = active.length
        ? "Using: " + active.map((item) => item.label).join(" · ") +
          " | any random day " + pct(baseRate) +
          (now.signals < 60 ? " | fewer than 60 signals: treat precision as noise" : "")
        : "Nothing selected: no signals at all.";
    }

    const panel = document.querySelector(".ctxbar");
    const isOpen = panel && !panel.classList.contains("collapsed");
    updateToggleUI(isOpen);
  }

  function buildPicker() {
    const holder = document.querySelector("#ctx-picker");
    if (!holder || !data) return;
    holder.textContent = "";

    const groups = {};
    data.contexts.filter((item) => item.group).forEach((item) => {
      (groups[item.group] = groups[item.group] || []).push(item.key);
    });

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

    const standalone = document.createElement("div");
    standalone.className = "ctx-standalone";
    data.contexts.filter((item) => !item.group).forEach((item) => standalone.appendChild(option(item)));
    holder.appendChild(standalone);

    Object.entries(groups).forEach(([name, keys]) => {
      const members = data.contexts.filter((item) => item.group === name);
      const field = document.createElement("fieldset");
      field.className = "ctx-group";
      const legend = document.createElement("legend");
      legend.textContent = (members[0].group_label || name) + " · select one";
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
      offText.innerHTML = 'None <span class="kind">· off · no news veto</span>';
      offLabel.appendChild(offBox);
      offLabel.appendChild(offText);
      field.appendChild(offLabel);

      members.forEach((item) => field.appendChild(option(item)));
      holder.appendChild(field);
    });

    // 判断函数规则配置：让用户不再受限于写死的单一布尔规则
    const ruleBar = document.createElement("div");
    ruleBar.className = "ctx-rule-bar";

    const trigField = document.createElement("div");
    trigField.className = "ctx-rule-field";
    const trigLabel = document.createElement("label");
    trigLabel.htmlFor = "ctx-trigger-mode";
    trigLabel.textContent = "Trigger rule (判断函数 · 触发)";
    const trigSelect = document.createElement("select");
    trigSelect.id = "ctx-trigger-mode";
    [
      ["any", "Any selected trigger (OR)"],
      ["consensus", "Multi-factor consensus (≥ 2)"],
      ["all", "All selected triggers (AND)"],
    ].forEach(([val, text]) => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = text;
      opt.selected = ruleConfig.triggerMode === val;
      trigSelect.appendChild(opt);
    });
    trigSelect.onchange = () => {
      ruleConfig.triggerMode = trigSelect.value;
      render();
    };
    trigField.appendChild(trigLabel);
    trigField.appendChild(trigSelect);

    const vetoField = document.createElement("div");
    vetoField.className = "ctx-rule-field";
    const vetoLabel = document.createElement("label");
    vetoLabel.htmlFor = "ctx-veto-mode";
    vetoLabel.textContent = "Veto tolerance (判断函数 · 否决)";
    const vetoSelect = document.createElement("select");
    vetoSelect.id = "ctx-veto-mode";
    [
      ["any", "Strict veto (any 1 hit cancels)"],
      ["consensus", "Double-confirm veto (≥ 2 hits)"],
    ].forEach(([val, text]) => {
      const opt = document.createElement("option");
      opt.value = val;
      opt.textContent = text;
      opt.selected = ruleConfig.vetoMode === val;
      vetoSelect.appendChild(opt);
    });
    vetoSelect.onchange = () => {
      ruleConfig.vetoMode = vetoSelect.value;
      render();
    };
    vetoField.appendChild(vetoLabel);
    vetoField.appendChild(vetoSelect);

    ruleBar.appendChild(trigField);
    ruleBar.appendChild(vetoField);
    holder.appendChild(ruleBar);
  }

  function bindGlobalEvents() {
    if (eventsBound) return;
    eventsBound = true;

    const resetBtn = document.querySelector("#ctx-reset");
    if (resetBtn) {
      resetBtn.onclick = () => {
        selected = new Set(requiredKeys(data));
        ruleConfig = { triggerMode: "any", vetoMode: "any" };
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

    document.addEventListener("click", (e) => {
      const panel = document.querySelector(".ctxbar");
      if (!panel || panel.classList.contains("collapsed")) return;
      if (panel.contains(e.target)) return;
      if (toggleBtn && toggleBtn.contains(e.target)) return;
      if (ruleElem && ruleElem.contains(e.target)) return;
      setOpen(false);
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        const panel = document.querySelector(".ctxbar");
        if (panel && !panel.classList.contains("collapsed")) {
          setOpen(false);
          if (toggleBtn) toggleBtn.focus();
        }
      }
    });
  }

  window.explainCtxAt = function (index, direction) {
    if (!data || index == null || index < 0 || index >= data.time.length) return null;
    const c = data.components || {};
    const active = new Set(selected);
    requiredKeys(data).forEach((key) => active.add(key));

    if (direction === "SELL") {
      const baseSell = Array.isArray(c.sell_composite)
        ? Boolean(c.sell_composite[index])
        : Boolean(data.sell_flag && data.sell_flag[index]);
      const checks = [
        {
          key: "sell_composite",
          label: "跨资产复合因子 ≤ 历史 10% 分位",
          kind: "trigger",
          hit: baseSell,
          passed: baseSell,
        },
      ];
      if (active.has("fomc_day") && Array.isArray(c.fomc_day)) {
        const isFomc = Boolean(c.fomc_day[index]);
        checks.push({
          key: "fomc_day",
          label: "FOMC 决议日否决检查",
          kind: "veto",
          hit: isFomc,
          passed: !isFomc,
        });
      }
      if (active.has("news_bullish") && Array.isArray(c.news_bullish)) {
        const isBull = Boolean(c.news_bullish[index]);
        checks.push({
          key: "news_bullish",
          label: "强利多意外否决做空检查",
          kind: "veto",
          hit: isBull,
          passed: !isBull,
        });
      }
      return {
        direction: "SELL",
        ruleSummary: "复合因子 ≤ 10% 分位 + 否决门控通过",
        checks,
      };
    }

    const contexts = data.contexts.filter((item) => active.has(item.key) && Array.isArray(c[item.key]));
    const triggers = contexts.filter((item) => item.kind === "trigger" || item.kind === "boost");
    const vetoes = contexts.filter((item) => item.kind === "veto");
    const checks = [];
    triggers.forEach((item) => {
      const hit = isHit(item, c[item.key][index]);
      checks.push({
        key: item.key,
        label: item.label,
        kind: item.kind,
        hit,
        passed: hit,
        raw: c[item.key][index],
      });
    });
    vetoes.forEach((item) => {
      const vetoed = isHit(item, c[item.key][index]);
      checks.push({
        key: item.key,
        label: item.label,
        kind: "veto",
        hit: vetoed,
        passed: !vetoed,
        raw: c[item.key][index],
      });
    });
    const trigLabel =
      ruleConfig.triggerMode === "consensus"
        ? "多因子共振 ≥2"
        : ruleConfig.triggerMode === "all"
          ? "全部触发 AND"
          : "任一触发 OR";
    const vetoLabel = ruleConfig.vetoMode === "consensus" ? "双重否决 ≥2" : "一票否决";
    return {
      direction: "BUY",
      ruleSummary: "触发规则：" + trigLabel + " · 否决规则：" + vetoLabel,
      checks,
    };
  };

  window.initCtxbar = function (document_, callback, options) {
    data = document_;
    onFlags = callback;
    currentMarket = (options && options.marketKey) || document_.instrument || "default";

    const saved = marketStates.get(currentMarket);
    if (saved) {
      selected = new Set(
        [...saved.selected].filter((key) => data.contexts.some((item) => item.key === key))
      );
      requiredKeys(data).forEach((key) => selected.add(key));
      ruleConfig = { ...saved.ruleConfig };
    } else {
      selected = new Set(defaultKeys(data));
      ruleConfig = { triggerMode: "any", vetoMode: "any" };
    }

    bindGlobalEvents();
    buildPicker();
    render();
  };
})();

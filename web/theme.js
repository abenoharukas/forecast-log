// CSS owns the palette; canvas does not read CSS variables. This bridge hands
// the resolved tokens to LightweightCharts and repaints when the OS scheme
// changes, so a dark-mode switch cannot leave a bright chart inside a dark page.
//
// Read the -l / -d token pairs rather than --paper itself: a custom property
// holding light-dark(...) is substituted, not computed, so getComputedStyle
// would hand the chart the literal "light-dark(#f7f3ea, #1d1a12)" string.
(function () {
  const dark = window.matchMedia("(prefers-color-scheme: dark)");
  const listeners = new Set();

  function palette() {
    const style = getComputedStyle(document.documentElement);
    const suffix = dark.matches ? "d" : "l";
    const read = (base) => style.getPropertyValue("--" + base + "-" + suffix).trim();
    return {
      paper: read("paper"),
      sheet: read("sheet"),
      ink: read("ink"),
      muted: read("muted"),
      rule: read("rule"),
      grid: read("grid"),
      blue: read("blue"),
      red: read("red"),
      green: read("green"),
      ochre: read("ochre"),
      amber: read("amber"),
      grey: read("grey"),
      dark: dark.matches,
    };
  }

  function chartOptions(p) {
    return {
      layout: { background: { type: "solid", color: p.sheet }, textColor: p.muted },
      grid: { vertLines: { color: p.grid }, horzLines: { color: p.grid } },
      rightPriceScale: { borderColor: p.rule },
      timeScale: { borderColor: p.rule },
      crosshair: { vertLine: { color: p.rule }, horzLine: { color: p.rule } },
    };
  }

  // Runs immediately with the current palette, then again on every OS change.
  function subscribe(callback) {
    listeners.add(callback);
    callback(palette());
    return () => listeners.delete(callback);
  }

  dark.addEventListener("change", () => {
    listeners.forEach((callback) => callback(palette()));
  });

  window.fcTheme = { palette, chartOptions, subscribe };
})();

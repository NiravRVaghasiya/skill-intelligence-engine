"""Artifacts for benchmarks/: marker-delimited RESULTS.md blocks and the results chart.

Colors follow the dataviz reference palette (categorical slots 2 and 1), validated with
its palette checker in both modes: orange = keyword router, blue = SIE.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "keyword": "#eb6834", "sie": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "keyword": "#d95926", "sie": "#3987e5"},
}


@dataclass
class Bar:
    """One system's row in the chart; `values` is metric -> (mean, ci_lo, ci_hi) or None."""
    label: str
    family: str                      # "keyword" | "sie"
    values: dict[str, tuple[float, float, float]] | None
    labelled: bool = False           # direct-label this row's values
    note: str = ""                   # shown instead of a bar when values is None


def replace_block(path: Path, name: str, body: str) -> None:
    """Replace the text between <!-- BEGIN:name --> and <!-- END:name --> (append if absent)."""
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{begin}\n{body.strip()}\n{end}"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def md_table(header: list[str], rows: list[list[str]], align: str = "") -> str:
    align = align or "l" + "c" * (len(header) - 1)
    sep = ["---" if a == "l" else ":---:" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _bar(ax, y: float, width: float, height: float, color: str, radius: float) -> None:
    """Horizontal bar, square at the baseline with a rounded data-end."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    if width <= 0:
        return
    r = min(radius, width / 2, height / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, linewidth=0, facecolor=color,
                                boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=1))
    ax.add_patch(Rectangle((0, y - height / 2), max(width - r, 0), height, linewidth=0, facecolor=color))


def _panel(ax, bars: list[Bar], metric: str, t: dict) -> None:
    """One metric for one query set: a horizontal bar per system, CI whiskers, sparse labels."""
    n = len(bars)
    ys = list(range(n))[::-1]
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 1.16)                      # headroom right of 1.0 for value labels
    ax.set_ylim(-0.7, n - 0.3)
    for y, bar in zip(ys, bars):
        if bar.values is None:
            ax.text(0.01, y, bar.note or "pending", va="center", ha="left",
                    color=t["muted"], fontsize=8, style="italic")
            continue
        mean, lo, hi = bar.values[metric]
        _bar(ax, y, mean, 0.46, t[bar.family], radius=0.012)
        ax.plot([lo, hi], [y, y], color=t["ink2"], linewidth=1, solid_capstyle="butt", zorder=3)
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.1, y + 0.1], color=t["ink2"], linewidth=1, zorder=3)
        if bar.labelled:
            ax.text(hi + 0.02, y, f"{mean:.2f}", va="center", ha="left",
                    color=t["ink"], fontsize=8.5, fontweight="bold")
    ax.set_title(metric, loc="left", color=t["ink"], fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"], color=t["muted"], fontsize=8)
    ax.grid(axis="x", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["axis"])
    ax.set_yticks(ys)
    ax.set_yticklabels([b.label for b in bars], color=t["ink2"], fontsize=9)


def render_chart(rows: list[tuple[str, list[Bar]]], metrics: list[str], title: str,
                 subtitle: str, path: Path, theme: str = "light") -> None:
    """Small multiples: one row per query set, one column per metric."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
                         "text.color": t["ink"]})
    heights = [0.42 * len(bars) + 0.7 for _, bars in rows]
    fig_h = sum(heights) + 1.1
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=(10.5, fig_h), dpi=200,
                             sharey="row", squeeze=False, gridspec_kw={"height_ratios": heights})
    fig.patch.set_facecolor(t["surface"])
    for r, (_, bars) in enumerate(rows):
        for c, metric in enumerate(metrics):
            _panel(axes[r][c], bars, metric, t)
    fig.text(0.01, 1 - 0.12 / fig_h, title, ha="left", va="top", fontsize=12,
             fontweight="bold", color=t["ink"])
    fig.text(0.01, 1 - 0.42 / fig_h, subtitle, ha="left", va="top", fontsize=8.5, color=t["ink2"])
    handles = [Patch(facecolor=t["keyword"], label="Keyword router (source repo)"),
               Patch(facecolor=t["sie"], label="SIE (this repo)")]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1 - 0.08 / fig_h),
                     ncol=2, frameon=False, fontsize=8.5, handlelength=1.0, handleheight=0.8)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig_h), h_pad=2.6)
    for r, (row_title, _) in enumerate(rows):   # query-set name above each row of panels
        top = axes[r][0].get_position().y1
        fig.text(0.01, top + 0.34 / fig_h, row_title, ha="left", va="bottom", fontsize=10,
                 fontweight="bold", color=t["ink2"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=t["surface"], metadata={"Software": None})
    plt.close(fig)

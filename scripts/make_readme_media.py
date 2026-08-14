"""Regenerate the README's demo GIFs and comparison figures.

Reads every run directory under backend/runs/ for the fleet-wide charts, and the
eval videos of a chosen few for the GIFs. Writes into docs/assets/. Re-run after
adding runs; nothing here is hand-edited afterwards.

    uv run --python 3.12 --with matplotlib --with numpy --with pillow --with scipy \
        python scripts/make_readme_media.py

Run it through `uv run --with ...`, NOT the app venv: matplotlib is deliberately
absent from .venv (see docs/architecture_notes.md) and this is a docs tool, not
app code -- it has no business adding runtime dependencies.

Needs ffmpeg/ffprobe on PATH for the GIF pass. Pass --figures-only to skip it.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "backend" / "runs"
OUT = REPO / "docs" / "assets"

EPISODE_CAP = 1000  # Isaac episode length limit; episode_length_mean tops out here.

# ── Palette. Categorical slots 1-3 of the reference data-viz palette, validated
#    for the light chart surface under --pairs all: worst pair CVD deltaE 9.2
#    (deutan), normal-vision 24.0. Aqua sits at 2.74:1 contrast, below the 3:1
#    bar, so every chart using it carries visible value labels (the documented
#    relief). Do not swap these for "nicer" hues without re-validating.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"  # status colour, reserved: only marks a failure event

ALGO_COLOUR = {"PPO": "#2a78d6", "SAC": "#eb6834", "TD3": "#1baf7a"}
ALGO_ORDER = ["PPO", "SAC", "TD3"]

# GIF encode settings, tuned by measurement. The source frame is mostly empty
# floor and the camera pans, so every pixel changes every frame -- worst case for
# GIF. Cropping to the robot and capping the palette does far more for size than
# shrinking the output: a 640 px full-frame encode lands at 13 MB, a cropped
# 480 px one at ~3 MB with the robot rendered LARGER. dither=none is safe on this
# flat-grey scene; bayer_scale=5 is the fallback for gradient-heavy footage.
GIF_SECONDS = 5.0
GIF_WIDTH = 480
GIF_FPS = 12
GIF_COLOURS = 64


@dataclass
class Clip:
    """A run whose eval video becomes a GIF.

    crop_h is the crop height in SOURCE pixels -- the zoom level. It is set per
    robot class rather than derived, because the subject detector reliably finds
    the robot's centre but can under-measure its extent when the silhouette
    breaks into separate bright blobs (a humanoid's head and torso, say). Centre
    is detected; zoom is declared.
    """

    algo: str
    robot: str
    run_id: str
    clip_start: float
    crop_h: int
    caption: str

    @property
    def dir(self) -> Path:
        return RUNS / f"{self.run_id}-isaac"

    @property
    def slug(self) -> str:
        return f"{self.robot.lower()}_{self.algo.lower()}"


CLIPS = [
    Clip("SAC", "Go1", "20260812-151442", 4.0, 280, "reward 2,250 -- best quadruped result"),
    Clip("PPO", "Go1", "20260801-185125", 4.0, 280, "reward 2,074"),
    Clip("SAC", "Valkyrie", "20260812-165024", 3.0, 420, "reward 1,284 -- best humanoid result"),
    Clip("TD3", "Valkyrie", "20260812-181127", 3.0, 420, "reward 287, later diverged"),
]

# The two runs the head-to-head figures compare: same robot, same reward
# expression, same 40M-step budget, different learner.
COMPARE = [
    ("SAC", "20260812-165024", "#2a78d6"),
    ("TD3", "20260812-181127", "#eb6834"),
]


# ── Data ─────────────────────────────────────────────────────────────────────


def robot_name(urdf: str) -> str:
    """Group URDF variants under one robot name (valkyrie_A/B/C are one robot)."""
    stem = Path(urdf).stem.lower()
    for prefix, name in (("valkyrie", "Valkyrie"), ("go1", "Go1"), ("a1", "A1"), ("anymal", "ANYmal")):
        if stem.startswith(prefix):
            return name
    return stem or "unknown"


@dataclass
class Series:
    """Telemetry of one run."""

    algo: str = ""
    robot: str = ""
    run_id: str = ""
    colour: str = ""
    num_envs: int = 0
    steps: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    ep_len: list[float] = field(default_factory=list)
    fps: list[float] = field(default_factory=list)
    terms: dict[str, list[float]] = field(default_factory=dict)
    term_steps: list[int] = field(default_factory=list)
    failure_step: int | None = None
    failure_message: str | None = None

    @property
    def best_index(self) -> int:
        return max(range(len(self.reward)), key=lambda i: self.reward[i])

    @property
    def best_reward(self) -> float:
        return self.reward[self.best_index]

    @property
    def best_ep_len(self) -> float:
        return self.ep_len[self.best_index]

    @property
    def peak_fps(self) -> float:
        clean = [v for v in self.fps if v == v]
        return max(clean) if clean else 0.0


def read_run(run_dir: Path) -> Series | None:
    """Parse one run directory, or None if it has no usable telemetry."""
    config_path, telemetry_path = run_dir / "config.json", run_dir / "telemetry.jsonl"
    if not (config_path.exists() and telemetry_path.exists()):
        return None
    config = json.loads(config_path.read_text(encoding="utf8"))
    urdf = str(config.get("config", {}).get("urdf_path") or "")
    s = Series(algo=config.get("algorithm", "?"), robot=robot_name(urdf),
               run_id=run_dir.name.replace("-isaac", ""))
    for line in telemetry_path.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("message", "")).startswith("isaac training failed"):
            s.failure_step = s.steps[-1] if s.steps else None
            s.failure_message = row["message"]
        if not isinstance(row.get("reward_mean"), (int, float)):
            continue
        s.steps.append(row["timestep"])
        s.reward.append(row["reward_mean"])
        s.ep_len.append(row.get("episode_length_mean", float("nan")))
        s.fps.append(row.get("fps", float("nan")))
        s.num_envs = row.get("num_envs", s.num_envs) or config.get("num_envs", 0)
        if isinstance(row.get("reward_terms"), dict):
            s.term_steps.append(row["timestep"])
            for name, value in row["reward_terms"].items():
                s.terms.setdefault(name, []).append(value)
    return s if s.reward else None


def survey() -> list[Series]:
    """Every run with usable telemetry, for the fleet-wide charts."""
    found = [read_run(Path(d)) for d in sorted(glob.glob(str(RUNS / "*-isaac")))]
    return [s for s in found if s]


def smooth(values: list[float], window: int = 15) -> list[float]:
    """Centred rolling mean. Plotted over the raw trace, not instead of it."""
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
        chunk = [v for v in values[lo:hi] if v == v]  # drop NaN
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


# ── Chart chrome ─────────────────────────────────────────────────────────────


def millions(value: float, _pos: int = 0) -> str:
    return f"{value / 1e6:g}M"


def thousands(value: float, _pos: int = 0) -> str:
    return f"{value / 1000:g}k"


def figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def style(ax, title: str, subtitle: str, xlabel: str, ylabel: str, x_millions: bool = True) -> None:
    """Recessive chrome: solid hairline grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if x_millions:
        ax.xaxis.set_major_formatter(FuncFormatter(millions))
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def legend(ax, **kwargs) -> None:
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_2)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(f"  {path.relative_to(REPO)}  ({path.stat().st_size / 1024:.0f} KB)")


# ── Fleet-wide figures ───────────────────────────────────────────────────────


def fig_overview(runs: list[Series]) -> None:
    """What the platform achieved per robot per learner.

    Two panels on one figure because reward alone is not the result: a humanoid
    can farm a stay-alive term without walking. Episode length as a fraction of
    the cap is the reality check, so the two sit side by side and are read
    together.
    """
    robots = sorted({s.robot for s in runs},
                    key=lambda r: -max(s.best_reward for s in runs if s.robot == r))
    best: dict[tuple[str, str], Series] = {}
    for s in runs:
        key = (s.robot, s.algo)
        if key not in best or s.best_reward > best[key].best_reward:
            best[key] = s

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    height = 0.26
    for panel, (ax, extract, fmt, label) in enumerate([
        (axes[0], lambda s: s.best_reward, lambda v: f"{v:,.0f}", "Best mean episode reward"),
        (axes[1], lambda s: 100 * min(s.best_ep_len, EPISODE_CAP) / EPISODE_CAP,
         lambda v: f"{v:.0f}%", "Episode length, % of the 1000-step cap"),
    ]):
        # Each robot's bars are centred as a group on its tick, so a PPO-only
        # robot reads as one bar against its label rather than sitting off to one
        # side of an invisible three-slot grid.
        for slot, algo in enumerate(ALGO_ORDER):
            ys, vs = [], []
            for i, robot in enumerate(robots):
                s = best.get((robot, algo))
                if not s:
                    continue
                present = [a for a in ALGO_ORDER if (robot, a) in best]
                ys.append(i + (present.index(algo) - (len(present) - 1) / 2) * height)
                vs.append(extract(s))
            if not ys:
                continue
            ax.barh(ys, vs, height=height - 0.06, color=ALGO_COLOUR[algo],
                    label=algo if panel == 0 else None)
            # Visible value labels: the documented relief for aqua's sub-3:1
            # contrast, and they let the reader skip the axis entirely.
            for y, v in zip(ys, vs):
                ax.text(v, y, f"  {fmt(v)}", va="center", ha="left", color=INK_2, fontsize=7.5)
        ax.set_yticks(range(len(robots)), robots)
        ax.invert_yaxis()
        style(ax, "", "", label, "", x_millions=False)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.17)
        ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK_2)
    axes[1].set_xlim(0, 100)

    fig.text(0.006, 1.0, "Every robot, every learner", color=INK, fontsize=13.5,
             fontweight="bold", ha="left", va="top")
    fig.text(0.006, 0.955, f"Best of {len(runs)} Isaac runs. Bars are each robot's best run per "
                           "algorithm -- gaps mean that pairing was never run (SAC and TD3 were "
                           "only tried on Go1 and Valkyrie).",
             color=INK_2, fontsize=9, ha="left", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, labelcolor=INK_2,
               loc="upper right", bbox_to_anchor=(0.997, 1.005), ncols=3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "overview_results.png")


def fig_scaling(runs: list[Series]) -> None:
    """Peak throughput against parallel environment count, faceted by robot.

    Faceted rather than colour-coded because there are four robots and the
    reference palette only validates three slots for all-pairs separation (slot 4
    puts yellow beside orange, which fails); the documented alternative to a
    fourth hue is small multiples. It also happens to show the real finding
    better: env count sets the floor, robot cost sets the ceiling.
    """
    robots = sorted({s.robot for s in runs},
                    key=lambda r: -max(s.peak_fps for s in runs if s.robot == r))
    fig, axes = plt.subplots(1, len(robots), figsize=(11.4, 3.3), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ceiling = max(s.peak_fps for s in runs)
    for ax, robot in zip(axes, robots):
        mine = [s for s in runs if s.robot == robot]
        ax.scatter([s.num_envs for s in mine], [s.peak_fps for s in mine], s=38,
                   color=ALGO_COLOUR["PPO"], alpha=0.75, edgecolor=SURFACE,
                   linewidth=1.3, zorder=3)
        top = max(mine, key=lambda s: s.peak_fps)
        ax.annotate(f"{top.peak_fps / 1000:.1f}k", (top.num_envs, top.peak_fps),
                    textcoords="offset points", xytext=(-7, 3), ha="right",
                    color=INK_2, fontsize=8.5, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks([256, 1024, 4096])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.set_ylim(0, ceiling * 1.15)
        ax.set_title(f"{robot}  ({len(mine)} runs)", color=INK, fontsize=10,
                     fontweight="bold", loc="left", pad=6)
        style(ax, "", "", "Parallel envs (log)", "", x_millions=False)
    axes[0].set_ylabel("Peak steps / second", color=INK_2, fontsize=9)
    fig.text(0.006, 1.0, "Throughput against parallel environments", color=INK,
             fontsize=13.5, fontweight="bold", ha="left", va="top")
    fig.text(0.006, 0.945, f"Peak simulation rate of all {len(runs)} runs on one RTX 5000. A1 and Go1 "
                           "reach 30k+; Valkyrie never clears 15k at any batch size -- a humanoid "
                           "costs more to simulate. ANYmal's ceiling is not comparable: those were "
                           "early runs, almost all under 2,048 envs.",
             color=INK_2, fontsize=9, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "overview_scaling.png")


# ── Head-to-head figures ─────────────────────────────────────────────────────


def fig_reward(runs: list[Series]) -> None:
    """Mean episode reward against environment steps, for the compared pair.

    The y-range is set from the smoothed traces, not the raw ones. Raw reward
    spikes to -3,800 on single evaluation windows, and letting those set the
    limits squashes the entire 0-1,300 story into the top quarter of the plot.
    The raw trace is still drawn (it just clips), and the subtitle says so.
    """
    fig, ax = figure(8.4, 4.3)
    lo, hi = 0.0, 0.0
    for run in runs:
        trace = smooth(run.reward)
        lo, hi = min(lo, min(trace)), max(hi, max(trace))
        ax.plot(run.steps, run.reward, color=run.colour, linewidth=0.8, alpha=0.18)
        ax.plot(run.steps, trace, color=run.colour, linewidth=1.8, label=run.algo)
        i = run.best_index
        ax.plot(run.steps[i], run.reward[i], "o", color=run.colour, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        # The upper series' best sits near the top of the range, so its label goes
        # beside the marker rather than above it -- above collides with the subtitle.
        upper = run.best_reward >= max(r.best_reward for r in runs)
        ax.annotate(f"best {run.reward[i]:,.0f}", (run.steps[i], run.reward[i]),
                    textcoords="offset points", xytext=(-10, 0 if upper else -18),
                    ha="right", va="center" if upper else "top", color=INK_2, fontsize=8.5)
    span = hi - lo
    ax.set_ylim(lo - span * 0.12, hi + span * 0.2)
    for run in runs:
        if run.failure_step:
            ax.axvline(run.failure_step, color=CRITICAL, linewidth=1.2, zorder=1)
            ax.annotate(f"{run.algo} diverged (NaN)", (run.failure_step, ax.get_ylim()[0]),
                        textcoords="offset points", xytext=(-7, 9), ha="right", va="bottom",
                        color=CRITICAL, fontsize=8.5, fontweight="bold")
    style(ax, "Mean episode reward",
          "Valkyrie humanoid, identical reward expression and 40M-step budget. "
          "Bold = rolling mean, faint = raw (clipped).",
          "Environment steps", "Reward (mean over envs)")
    legend(ax, loc="upper left")
    save(fig, "compare_reward.png")


def fig_episode_length(runs: list[Series]) -> None:
    """How long the humanoid stays up. 1000 steps is the episode cap."""
    fig, ax = figure(8.4, 4.0)
    ax.axhline(EPISODE_CAP, color=AXIS, linewidth=0.9)
    # Right-aligned: the legend owns the upper-left corner.
    ax.text(0.997, EPISODE_CAP, f"episode cap ({EPISODE_CAP})  ", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", color=MUTED, fontsize=8)
    for run in runs:
        ax.plot(run.steps, run.ep_len, color=run.colour, linewidth=0.8, alpha=0.18)
        ax.plot(run.steps, smooth(run.ep_len), color=run.colour, linewidth=1.8, label=run.algo)
        ax.annotate(f"{run.best_ep_len:,.0f}", (run.steps[run.best_index], run.best_ep_len),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    color=run.colour, fontsize=9, fontweight="bold")
    style(ax, "Episode length",
          "Steps survived before termination -- the honest measure of whether it walks",
          "Environment steps", "Steps per episode")
    legend(ax, loc="upper left")
    save(fig, "compare_episode_length.png")


def fig_terms_at_best(runs: list[Series]) -> None:
    """Grouped bars: each run's reward terms at its own best-reward step."""
    names = list(runs[0].terms.keys())
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    height = 0.36
    positions = range(len(names))
    for offset, run in zip((height / 2, -height / 2), runs):
        i = min(run.best_index, len(run.term_steps) - 1)
        values = [run.terms[n][i] for n in names]
        ys = [p + offset for p in positions]
        ax.barh(ys, values, height=height - 0.04, color=run.colour,
                label=f"{run.algo}  (reward {run.best_reward:,.0f})")
        for y, value in zip(ys, values):
            ax.text(value + (0.02 if value >= 0 else -0.02), y, f"{value:+.2f}",
                    va="center", ha="left" if value >= 0 else "right",
                    color=INK_2, fontsize=7.5)
    ax.axvline(0, color=AXIS, linewidth=0.9)
    ax.set_yticks(list(positions), names)
    ax.invert_yaxis()
    style(ax, "Where the reward came from",
          "Per-term contribution PER STEP at each run's own best-reward checkpoint -- note how close "
          "these are. SAC's 4.5x total-reward lead is mostly episode length, not a better step.",
          "Contribution to reward per step", "", x_millions=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.13)
    legend(ax, loc="lower right")
    save(fig, "compare_reward_terms_best.png")


# ── GIFs ─────────────────────────────────────────────────────────────────────


def robot_centre(video: Path, start: float, duration: float, samples: int = 11) -> tuple[int, int]:
    """Locate the robot across a clip window and return the centre to crop around.

    The robot is the brightest compact object in the scene, but a plain
    brightness threshold also catches the sky gradient and the grid lines. So:
    take the brightest N pixels, then keep the largest connected blob among
    them. Count-based rather than value-based, so it adapts to clip exposure;
    connectivity is what rejects sky and grid.
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    boxes = []
    for i in range(samples):
        t = start + duration * i / (samples - 1)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video),
                              "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                             capture_output=True).stdout
        frame = np.array(Image.open(io.BytesIO(raw)).convert("L"))
        keep = min(1200, frame.size // 8)
        mask = frame >= np.partition(frame.ravel(), -keep)[-keep]
        labelled, count = ndimage.label(mask)
        if not count:
            continue
        sizes = ndimage.sum(mask, labelled, range(1, count + 1))
        ys, xs = np.where(labelled == 1 + int(np.argmax(sizes)))
        boxes.append((xs.min(), xs.max(), ys.min(), ys.max()))
    if not boxes:
        h, w = frame.shape
        return w // 2, h // 2
    x0 = min(b[0] for b in boxes); x1 = max(b[1] for b in boxes)
    y0 = min(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    return int((x0 + x1) / 2), int((y0 + y1) / 2)


def crop_filter(video: Path, clip: Clip) -> str:
    """A 4:3 crop of height clip.crop_h centred on the robot, clamped to frame."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True).stdout.strip().split(",")
    src_w, src_h = int(probe[0]), int(probe[1])
    cx, cy = robot_centre(video, clip.clip_start, GIF_SECONDS)
    h = min(clip.crop_h, src_h) // 2 * 2
    w = min(int(h * 4 / 3), src_w) // 2 * 2
    x = max(0, min(cx - w // 2, src_w - w)) // 2 * 2
    y = max(0, min(cy - h // 2, src_h - h)) // 2 * 2
    print(f"    subject centre ({cx},{cy}) -> crop {w}x{h}+{x}+{y}")
    return f"crop={w}:{h}:{x}:{y}"


def make_gif(clip: Clip) -> None:
    """Two-pass palette encode. One shared palette per clip keeps the grid floor
    from dithering into noise, which a per-frame palette does."""
    source = clip.dir / "eval_video.mp4"
    if not source.exists():
        print(f"  skipped {clip.slug}: no eval_video.mp4")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{clip.slug}.gif"
    palette = OUT / f".palette_{clip.slug}.png"
    chain = f"fps={GIF_FPS},{crop_filter(source, clip)},scale={GIF_WIDTH}:-1:flags=lanczos"
    window = ["-ss", str(clip.clip_start), "-t", str(GIF_SECONDS)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", *window, "-i", str(source), "-vf",
                    f"{chain},palettegen=max_colors={GIF_COLOURS}:stats_mode=diff", str(palette)],
                   check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *window, "-i", str(source), "-i", str(palette),
                    "-lavfi", f"{chain}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", str(target)], check=True)
    palette.unlink(missing_ok=True)
    print(f"  {target.relative_to(REPO)}  ({target.stat().st_size / 1024:.0f} KB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true", help="skip the ffmpeg GIF pass")
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]

    runs = survey()
    print(f"{len(runs)} runs with telemetry across {len({s.robot for s in runs})} robots")
    peak = max(runs, key=lambda s: s.peak_fps)
    reward = max(runs, key=lambda s: s.best_reward)
    print(f"  peak throughput {peak.peak_fps:,.0f} steps/s  ({peak.robot} {peak.algo}, "
          f"{peak.num_envs:,} envs)")
    print(f"  best reward     {reward.best_reward:,.0f}  ({reward.robot} {reward.algo}, "
          f"episode length {reward.best_ep_len:,.0f})")

    by_id = {s.run_id: s for s in runs}
    compare = []
    for algo, run_id, colour in COMPARE:
        s = by_id.get(run_id)
        if not s:
            raise SystemExit(f"compared run {run_id} not found")
        s.colour = colour
        compare.append(s)

    print("figures:")
    fig_overview(runs)
    fig_scaling(runs)
    fig_reward(compare)
    fig_episode_length(compare)
    fig_terms_at_best(compare)

    if not args.figures_only:
        if not shutil.which("ffmpeg"):
            print("gifs: skipped, ffmpeg not on PATH")
            return 0
        print("gifs:")
        for clip in CLIPS:
            make_gif(clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())

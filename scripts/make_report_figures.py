"""Regenerate every chart the report uses, from the Isaac run telemetry.

Nothing here is hand-drawn and nothing is quoted by hand: each figure is computed
from backend/runs/*/telemetry.jsonl, so re-running after adding runs updates the
report's evidence. Writes PNGs into report/media/, which scripts/write_report.py
then types into Word.

The plotting chrome, palette and telemetry reader are reused from
scripts/make_readme_media.py rather than duplicated -- that module is already the
project's single definition of what a chart looks like.

    py -m uv run --python 3.12 --with matplotlib --with numpy python \
        scripts/make_report_figures.py

Run it through `uv run --with ...`, NOT the app venv: matplotlib is deliberately
absent from .venv (see docs/architecture_notes.md) and this is a docs tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_readme_media as mrm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "report" / "media"

EPISODE_CAP = mrm.EPISODE_CAP
SURVIVED = 0.9 * EPISODE_CAP  # "reached the cap" threshold used throughout the report

ROBOT_ORDER = ["A1", "Go1", "ANYmal", "Valkyrie"]
ROBOT_COLOUR = {"A1": "#2a78d6", "Go1": "#eb6834", "ANYmal": "#1baf7a", "Valkyrie": "#8a5cd0"}
ALGO_MARKER = {"PPO": "o", "SAC": "s", "TD3": "^"}


# ── Selection helpers ────────────────────────────────────────────────────────


def best_per_pairing(runs):
    """Highest-reward run for each (robot, algorithm), the report's fleet summary rule."""
    best = {}
    for r in runs:
        key = (r.robot, r.algo)
        if key not in best or r.best_reward > best[key].best_reward:
            best[key] = r
    return best


def for_robot(runs, robot, algos=None):
    best = best_per_pairing(runs)
    picked = [best[k] for k in best if k[0] == robot and (algos is None or k[1] in algos)]
    return sorted(picked, key=lambda r: mrm.ALGO_ORDER.index(r.algo) if r.algo in mrm.ALGO_ORDER else 9)


def robust_limits(series_values, pad=0.08):
    """Y-limits from the smoothed traces, so single-step reward collapses to
    -7,000 do not flatten the trend the figure exists to show."""
    lows, highs = [], []
    for values in series_values:
        clean = [v for v in mrm.smooth(values) if v == v]
        if clean:
            lows.append(min(clean))
            highs.append(max(clean))
    if not lows:
        return None
    low, high = min(lows), max(highs)
    span = (high - low) or abs(high) or 1.0
    return low - span * pad, high + span * pad * 2.4


def tall_titles(ax, title, subtitle, xlabel, ylabel, x_millions=True):
    """style() anchors its subtitle at a fixed axes fraction, which on a tall
    figure rides over the fixed-point title pad. Place both in points instead."""
    mrm.style(ax, title, "", xlabel, ylabel, x_millions=x_millions)
    ax.set_title(title, color=mrm.INK, fontsize=12.5, fontweight="bold", loc="left", pad=26)
    ax.annotate(
        subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
        textcoords="offset points", color=mrm.INK_2, fontsize=9, va="bottom",
    )


def steps_to_survive(run):
    """Environment steps at which mean episode length first *holds* above SURVIVED.

    Measured on the smoothed trace on purpose: a single lucky logging step above
    the threshold is not the policy having learned to stay upright, and the raw
    trace produces exactly that false positive on the early ANYmal runs."""
    for step, length in zip(run.steps, mrm.smooth(run.ep_len)):
        if length == length and length >= SURVIVED:
            return step
    return None


# ── Figures ──────────────────────────────────────────────────────────────────


def fig_fleet_chronology(runs):
    """Every analysable run in date order: did the policy survive the episode?"""
    fig, ax = mrm.figure(7.6, 2.9)
    ordered = sorted(runs, key=lambda r: r.run_id)
    for robot in ROBOT_ORDER:
        xs = [i for i, r in enumerate(ordered) if r.robot == robot]
        if not xs:
            continue
        ax.scatter(
            xs,
            [100 * ordered[i].best_ep_len / EPISODE_CAP for i in xs],
            s=46,
            color=ROBOT_COLOUR[robot],
            edgecolor=mrm.SURFACE,
            linewidth=0.8,
            label=robot,
            zorder=3,
        )
    ax.axhline(95, color=mrm.MUTED, linewidth=0.9, linestyle="--", zorder=2)
    ax.text(len(ordered) - 0.5, 96.5, "95% of episode cap", color=mrm.MUTED, fontsize=8, ha="right")
    mrm.style(
        ax,
        "Every analysable run, in order",
        "%d Isaac runs with reward telemetry, 13 July to 18 August 2026" % len(ordered),
        "run (chronological)",
        "best mean episode length (% of cap)",
        x_millions=False,
    )
    ax.set_ylim(0, 108)
    mrm.legend(ax, loc="lower right", ncol=4)
    mrm.save(fig, "fleet_chronology.png")


def fig_fleet_survival(runs):
    """Best run per robot and algorithm, as a share of the episode cap."""
    best = best_per_pairing(runs)
    fig, ax = mrm.figure(7.6, 2.9)
    width = 0.26
    # Robots that were only ever run with one learner get a centred bar rather than
    # one parked in PPO's slot, which reads as a misaligned chart.
    present = {robot: [a for a in mrm.ALGO_ORDER if (robot, a) in best] for robot in ROBOT_ORDER}
    for algo in mrm.ALGO_ORDER:
        xs, ys = [], []
        for i, robot in enumerate(ROBOT_ORDER):
            run = best.get((robot, algo))
            if run is None:
                continue
            group = present[robot]
            offset = (group.index(algo) - (len(group) - 1) / 2) * width
            xs.append(i + offset)
            ys.append(100 * run.best_ep_len / EPISODE_CAP)
        if not xs:
            continue
        bars = ax.bar(xs, ys, width * 0.92, color=mrm.ALGO_COLOUR[algo], label=algo, zorder=3)
        for bar, value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.6,
                "%.0f" % value,
                ha="center",
                color=mrm.INK_2,
                fontsize=8,
            )
    mrm.style(
        ax,
        "The quadrupeds survive the episode; the humanoid mostly does not",
        "best observed run for each robot and algorithm pairing",
        "",
        "mean episode length (% of cap)",
        x_millions=False,
    )
    ax.set_xticks(range(len(ROBOT_ORDER)))
    ax.set_xticklabels(ROBOT_ORDER, color=mrm.INK_2, fontsize=9.5)
    ax.set_ylim(0, 116)
    mrm.legend(ax, loc="upper right", ncol=3)
    mrm.save(fig, "fleet_survival.png")


def fig_throughput(runs):
    """Peak throughput against parallel environment count."""
    fig, ax = mrm.figure(7.6, 2.9)
    for robot in ROBOT_ORDER:
        picked = [r for r in runs if r.robot == robot and r.num_envs and r.peak_fps]
        if not picked:
            continue
        ax.scatter(
            [r.num_envs for r in picked],
            [r.peak_fps for r in picked],
            s=44,
            color=ROBOT_COLOUR[robot],
            edgecolor=mrm.SURFACE,
            linewidth=0.8,
            label=robot,
            zorder=3,
        )
    peak = max(runs, key=lambda r: r.peak_fps)
    ax.annotate(
        "%s, %s: %s steps/s" % (peak.robot, peak.algo, "{:,.0f}".format(peak.peak_fps)),
        xy=(peak.num_envs, peak.peak_fps),
        xytext=(-12, -22),
        textcoords="offset points",
        color=mrm.INK_2,
        fontsize=8.5,
        ha="right",
        arrowprops=dict(arrowstyle="-", color=mrm.AXIS, linewidth=0.8),
    )
    mrm.style(
        ax,
        "Throughput follows the environment count, until the body costs too much",
        "peak simulation rate per run, one RTX 5000",
        "parallel environments",
        "peak steps/s",
        x_millions=False,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 128, 512, 2048, 4096, 8192])
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _p: "%d" % v))
    ax.yaxis.set_major_formatter(FuncFormatter(mrm.thousands))
    mrm.legend(ax, loc="upper left", ncol=4)
    mrm.save(fig, "throughput.png")


def fig_learning_curves(runs):
    """Best run per robot: reward against environment steps, one panel each."""
    best = best_per_pairing(runs)
    picked = []
    for robot in ROBOT_ORDER:
        candidates = [best[k] for k in best if k[0] == robot]
        if candidates:
            picked.append(max(candidates, key=lambda r: r.best_reward))
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.9), dpi=160)
    fig.patch.set_facecolor(mrm.SURFACE)
    for ax, run in zip(axes.flat, picked):
        ax.plot(run.steps, run.reward, color=ROBOT_COLOUR[run.robot], linewidth=0.7, alpha=0.35, zorder=2)
        ax.plot(run.steps, mrm.smooth(run.reward), color=ROBOT_COLOUR[run.robot], linewidth=1.8, zorder=3)
        mrm.style(
            ax,
            "%s – %s" % (run.robot, run.algo),
            "best reward %s, %.0f%% of episode cap"
            % ("{:,.0f}".format(run.best_reward), 100 * run.best_ep_len / EPISODE_CAP),
            "environment steps",
            "mean episode reward",
        )
        ax.title.set_fontsize(10.5)
    for ax in axes.flat[len(picked):]:
        ax.set_visible(False)
    fig.tight_layout(pad=1.4, h_pad=2.6, w_pad=2.2)
    mrm.save(fig, "learning_curves.png")


def _algorithm_panel(runs, robot, field, ylabel, title, subtitle, name,
                     cap_line=False, algos=None, legend_loc="lower right"):
    picked = for_robot(runs, robot, algos)
    if not picked:
        print("  skipped %s: no runs for %s" % (name, robot))
        return
    fig, ax = mrm.figure(7.6, 2.9)
    failures = []
    for run in picked:
        values = getattr(run, field)
        colour = mrm.ALGO_COLOUR.get(run.algo, mrm.INK_2)
        ax.plot(run.steps, values, color=colour, linewidth=0.7, alpha=0.3, zorder=2)
        ax.plot(
            run.steps,
            mrm.smooth(values),
            color=colour,
            linewidth=1.9,
            label="%s (best %s)" % (run.algo, "{:,.0f}".format(run.best_reward)),
            zorder=3,
        )
        if run.failure_step is not None:
            failures.append(run.failure_step)

    limits = robust_limits([getattr(r, field) for r in picked])
    if cap_line:
        limits = (min(limits[0], 0), max(limits[1], EPISODE_CAP * 1.12)) if limits else None
    if limits:
        ax.set_ylim(*limits)
    if cap_line:
        ax.axhline(EPISODE_CAP, color=mrm.MUTED, linewidth=0.9, linestyle="--", zorder=2)
        ax.text(0, EPISODE_CAP * 1.02, "episode cap", color=mrm.MUTED, fontsize=8)
    for step in failures:
        ax.axvline(step, color=mrm.CRITICAL, linewidth=1.0, linestyle=":", zorder=4)
        ax.text(step, ax.get_ylim()[1], " run failed", color=mrm.CRITICAL, fontsize=8, va="top")
    mrm.style(ax, title, subtitle, "environment steps", ylabel)
    mrm.legend(ax, loc=legend_loc)
    mrm.save(fig, name)


def fig_go1_algorithms(runs):
    _algorithm_panel(
        runs, "Go1", "reward", "mean episode reward",
        "On a quadruped, every learner gets there",
        "Unitree Go1, best run per algorithm, same reward expression",
        "go1_algorithms.png", legend_loc="upper left",
    )


def fig_go1_episode_length(runs):
    _algorithm_panel(
        runs, "Go1", "ep_len", "mean episode length (steps)",
        "All three Go1 policies hold the full episode",
        "reward separates the learners; survival does not",
        "go1_episode_length.png", cap_line=True,
    )


def fig_valkyrie_reward(runs):
    # SAC and TD3 only: those two share a robot, a reward expression and a 40M-step
    # budget, so they are the one controlled comparison the archive supports. The
    # Valkyrie PPO run went to 200M steps and belongs in its own discussion.
    _algorithm_panel(
        runs, "Valkyrie", "reward", "mean episode reward",
        "On a humanoid, the learner decides the outcome",
        "NASA Valkyrie, SAC against TD3, same reward expression and 40M-step budget",
        "valkyrie_reward.png", algos={"SAC", "TD3"}, legend_loc="upper left",
    )


def fig_valkyrie_episode_length(runs):
    _algorithm_panel(
        runs, "Valkyrie", "ep_len", "mean episode length (steps)",
        "SAC's advantage is staying upright, not scoring faster",
        "episode length tells the story that total reward hides",
        "valkyrie_episode_length.png", cap_line=True, algos={"SAC", "TD3"},
    )


def fig_reward_terms(runs):
    """What the winning policy was actually paid for, term by term."""
    candidates = [r for r in runs if r.robot == "Go1" and len(r.terms) >= 4]
    if not candidates:
        print("  skipped reward_terms.png: no Go1 run carries reward-term telemetry")
        return
    run = max(candidates, key=lambda r: r.best_reward)
    index = min(run.best_index, len(run.term_steps) - 1)
    pairs = []
    for name, values in run.terms.items():
        if index < len(values) and values[index] == values[index]:
            pairs.append((name.replace("_", " "), values[index]))
    pairs.sort(key=lambda kv: kv[1])
    fig, ax = mrm.figure(7.6, max(2.6, 0.34 * len(pairs) + 1.2))
    colours = [mrm.CRITICAL if value < 0 else mrm.ALGO_COLOUR["PPO"] for _, value in pairs]
    bars = ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=colours, zorder=3, height=0.66)
    span = max(abs(v) for _, v in pairs) or 1.0
    for bar, (_, value) in zip(bars, pairs):
        offset = span * 0.02
        ax.text(
            value + (offset if value >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            "%.3g" % value,
            va="center",
            ha="left" if value >= 0 else "right",
            color=mrm.INK_2,
            fontsize=8,
        )
    ax.axvline(0, color=mrm.AXIS, linewidth=0.9)
    # style() anchors its subtitle at a fixed axes fraction, which on a tall figure
    # rides over the fixed-point title pad. Place this one in points instead.
    tall_titles(
        ax,
        "What the best Go1 policy was paid for",
        "reward terms at the peak-reward step of run %s (%s)" % (run.run_id, run.algo),
        "term contribution", "", x_millions=False,
    )
    # Asymmetric limits: the penalties are three orders of magnitude smaller than
    # the rewards, so a symmetric axis would hide every one of them.
    low = min(v for _, v in pairs)
    high = max(v for _, v in pairs)
    margin = (high - low) * 0.22
    ax.set_xlim(min(low - margin, -margin * 0.5), high + margin)
    mrm.save(fig, "reward_terms.png")


def fig_sample_efficiency(runs):
    """Environment steps needed before a policy first holds 90% of the episode."""
    reached = [(r, steps_to_survive(r)) for r in runs]
    reached = [(r, s) for r, s in reached if s]
    if not reached:
        print("  skipped sample_efficiency.png: no run reached the survival threshold")
        return
    reached.sort(key=lambda rs: rs[1])
    fig, ax = mrm.figure(7.6, max(2.6, 0.32 * len(reached) + 1.1))
    labels = ["%s %s  %s" % (r.robot, r.algo, r.run_id) for r, _ in reached]
    bars = ax.barh(
        labels,
        [s for _, s in reached],
        color=[ROBOT_COLOUR[r.robot] for r, _ in reached],
        zorder=3,
        height=0.66,
    )
    for bar, (_, steps) in zip(bars, reached):
        ax.text(
            steps * 1.02,
            bar.get_y() + bar.get_height() / 2,
            "%.2fM" % (steps / 1e6),
            va="center",
            color=mrm.INK_2,
            fontsize=8,
        )
    ax.invert_yaxis()
    tall_titles(
        ax,
        "How long until the policy stops falling over",
        "environment steps before mean episode length first holds above %d" % SURVIVED,
        "environment steps", "",
    )
    ax.set_xlim(0, max(s for _, s in reached) * 1.16)
    ax.tick_params(axis="y", labelsize=8)
    mrm.save(fig, "sample_efficiency.png")


FIGURES = [
    fig_fleet_chronology,
    fig_fleet_survival,
    fig_throughput,
    fig_learning_curves,
    fig_go1_algorithms,
    fig_go1_episode_length,
    fig_valkyrie_reward,
    fig_valkyrie_episode_length,
    fig_reward_terms,
    fig_sample_efficiency,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", default="", help="substring filter on figure function names")
    args = ap.parse_args()

    mrm.OUT = args.out  # save() writes here instead of docs/assets
    runs = mrm.survey()
    if not runs:
        sys.exit("no runs with telemetry under %s" % mrm.RUNS)
    print("%d runs with reward telemetry under %s" % (len(runs), mrm.RUNS))

    for figure in FIGURES:
        if args.only and args.only not in figure.__name__:
            continue
        figure(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

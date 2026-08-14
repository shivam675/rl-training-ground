# EasyRTG Final Presentation — build record

**Goal:** a 14-slide, light-theme, 16:9 PowerPoint telling EasyRTG's final-submission story from
repository evidence and recorded robot-policy footage, sized for a 12–14 minute talk, no speaker
notes. Output: `EasyRTG_Final_Submission_Shivam_Chavan.pptx` at the repo root.

Deck design (slide-by-slide) is in
[`../specs/2026-08-06-easyrtg-final-presentation-design.md`](../specs/2026-08-06-easyrtg-final-presentation-design.md).

## Design constraints

- `Georgia Bold` titles, `Segoe UI` body, `Segoe UI Semibold` metrics.
- Palette: background `#F6F7F2`, ink `#18232F`, teal `#0B6B64`, rust `#B8502A`, slate `#5B6670`.
- Minimum sizes: 50 pt deck title, 35 pt slide titles, 24 pt subheads, 16 pt body.
- Real footage and repository evidence only — no synthetic robot imagery, no invented claims.

## Verified metrics (recomputed from telemetry, not quoted from memory)

Aggregated from the run directories' `telemetry.jsonl`; these are the exact values the slides show.

```text
Go1 run 20260802-191958-isaac: 40M configured steps; 6,400 envs; 30,138 average FPS; 31,631 peak
  FPS; final forward_speed 1.377 m/s; flight_phase 0.215; episode_length_mean 1,000; elapsed
  1,363.09 s.
Valkyrie run 20260803-233317-isaac: 200M configured steps; 5,800 envs; 14,005 average FPS;
  best-reward row at 31,041,600 steps has reward 276.1392, episode_length_mean 204.24,
  forward_velocity 0.721 m/s.
Valkyrie run 20260803-211052-isaac: verified peak throughput 14,555.7 FPS, rounded to 14.6k. A
  SEPARATE run from the selected policy evidence — do not merge the two.
CPU figures 240 FPS and approximately 1,750 FPS are presenter-provided historical measurements
  and are labeled as such on the slide.
```

Wording rule that came out of the consistency check: the deck says **"simulation/training
throughput"**, never "rendering frame rate", and every number carries its average/peak qualifier,
robot name and environment count.

## Asset provenance

Slide imagery is FFmpeg single-frame extractions from the two recorded evaluation videos —
`backend/runs/20260802-191958-isaac/eval_video.mp4` (Go1) and `X:\20260803-233317-isaac.mp4`
(Valkyrie) — sampled at 1.5 s / 3.5 s / 5.5 s at source resolution (960×540), then culled by eye:
keep frames that clearly show body posture and direction of travel, reject blurred, transitional,
obstructed or badly framed ones.

```powershell
ffmpeg -ss 3.5 -i 'backend\runs\20260802-191958-isaac\eval_video.mp4' -frames:v 1 -y 'assets\go1-2.png'
```

Slide 4's app screenshot is a genuine capture of `dist\EasyRTG\EasyRTG.exe` on its home screen
(allow ~60 s for first paint). If the window does not render, the slide falls back to a native
workflow composition rather than a fabricated screenshot.

## QA gates the deck had to pass

- Structural: exactly 14 slides, zero notes parts in the `.pptx` package.
- Mechanical: no text overflow, clipping, unresolved placeholders, or slide-count mismatch.
- Visual: every slide rendered to PNG and inspected full-size for title wrapping, body
  readability, image crops, chart labels, consistent page numbers, colour semantics and evidence
  footers. Any unintended overlap or weak crop is fixed and the deck re-rendered.
- Evidence: displayed metrics recomputed against the source run files before shipping.

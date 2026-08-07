# EasyRTG Final Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and visually verify a 14-slide, light-theme PowerPoint that tells EasyRTG's final-submission story using repository evidence and recorded robot-policy footage.

**Architecture:** A single JavaScript ES module will build the deck with `@oai/artifact-tool`. An isolated worktree build directory will hold extracted video frames, source notes, renders, and QA output; only the final `.pptx` will be copied to the main repository root.

**Tech Stack:** `@oai/artifact-tool`, bundled Node.js runtime, FFmpeg/FFprobe, LibreOffice-based slide renderer, PowerPoint OOXML inspection.

## Global Constraints

- Output: `X:\workspace\auto-rl\sxc1742\EasyRTG_Final_Submission_Shivam_Chavan.pptx`.
- Build directory: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final`.
- 16:9 format, 14 slides, designed for 12–14 minutes.
- No speaker notes.
- Use `Georgia Bold` for titles, `Segoe UI` for body, and `Segoe UI Semibold` for metrics.
- Palette: background `#F6F7F2`, ink `#18232F`, teal `#0B6B64`, rust `#B8502A`, slate `#5B6670`.
- Minimum sizes: 50 pt deck title, 35 pt slide titles, 24 pt subheads, 16 pt body.
- Use real EasyRTG footage and repository evidence; do not generate synthetic robot imagery or invent claims.
- Keep the user’s unrelated modified files untouched.

---

### Task 1: Verify metrics and prepare source records

**Files:**
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\source-notes.txt`
- Read: `EasyRTG_Project_Proposal.docx`
- Read: `backend/runs/20260802-191958-isaac/{config.json,telemetry.jsonl,play_log.txt}`
- Read: `backend/runs/20260803-233317-isaac/{config.json,telemetry.jsonl,play_log.txt}`
- Read: `docs/superpowers/plans/2026-07-29-expression-cutover.md`

**Interfaces:**
- Consumes: saved project evidence and the approved design spec.
- Produces: exact slide-ready metrics and a provenance ledger used by the deck module.

- [ ] **Step 1: Recompute run statistics from telemetry**

Run PowerShell JSONL aggregation to confirm average/peak FPS, final Go1 gait metrics, and Valkyrie’s best-reward telemetry row.

- [ ] **Step 2: Record the exact evidence**

Write `source-notes.txt` with these verified values:

```text
Go1 run 20260802-191958-isaac: 40M configured steps; 6,400 envs; 30,138 average FPS; 31,631 peak FPS; final forward_speed 1.377 m/s; flight_phase 0.215; episode_length_mean 1,000; elapsed 1,363.09 s.
Valkyrie run 20260803-233317-isaac: 200M configured steps; 5,800 envs; 14,005 average FPS; best-reward row at 31,041,600 steps has reward 276.1392, episode_length_mean 204.24, forward_velocity 0.721 m/s.
Valkyrie run 20260803-211052-isaac: verified peak throughput 14,555.7 FPS, rounded to 14.6k; this is a separate run from the selected policy evidence.
CPU figures 240 FPS and approximately 1,750 FPS are presenter-provided historical measurements and must be labeled as such.
```

- [ ] **Step 3: Check evidence consistency**

Confirm the deck will say “simulation/training throughput,” not rendering frame rate, and will label averages, peaks, robot names, and environment counts.

### Task 2: Extract real visual assets

**Files:**
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\go1-*.png`
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\valkyrie-*.png`
- Read: `backend/runs/20260802-191958-isaac/eval_video.mp4`
- Read: `X:\20260803-233317-isaac.mp4`

**Interfaces:**
- Consumes: the two recorded evaluation videos.
- Produces: sharp 16:9 PNG frames selected for the two policy-result slides and title/transition use.

- [ ] **Step 1: Extract candidate frames**

Use FFmpeg at several timestamps from each video, preserving the original 960×540 or source resolution.

```powershell
ffmpeg -ss 1.5 -i 'X:\workspace\auto-rl\sxc1742\backend\runs\20260802-191958-isaac\eval_video.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\go1-1.png'
ffmpeg -ss 3.5 -i 'X:\workspace\auto-rl\sxc1742\backend\runs\20260802-191958-isaac\eval_video.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\go1-2.png'
ffmpeg -ss 5.5 -i 'X:\workspace\auto-rl\sxc1742\backend\runs\20260802-191958-isaac\eval_video.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\go1-3.png'
ffmpeg -ss 1.5 -i 'X:\20260803-233317-isaac.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\valkyrie-1.png'
ffmpeg -ss 3.5 -i 'X:\20260803-233317-isaac.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\valkyrie-2.png'
ffmpeg -ss 5.5 -i 'X:\20260803-233317-isaac.mp4' -frames:v 1 -y 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\assets\valkyrie-3.png'
```

- [ ] **Step 2: Inspect every candidate at full size**

Keep frames that clearly show body posture and direction of travel; reject blurred, transitional, obstructed, or poorly framed images.

- [ ] **Step 3: Prepare a real app visual if available**

Launch `X:\workspace\auto-rl\sxc1742\dist\EasyRTG\EasyRTG.exe` and allow up to 60 seconds for the home screen. Capture one genuine interface screenshot if the window renders correctly. If it does not, omit the screenshot rather than fabricate one; slide 4 will use a simple native workflow composition instead.

### Task 3: Build the presentation

**Files:**
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\build_presentation.mjs`
- Create: `EasyRTG_Final_Submission_Shivam_Chavan.pptx`

**Interfaces:**
- Consumes: verified metrics, selected frames, proposal metadata, and the approved design spec.
- Produces: a 14-slide PowerPoint with no notes.

- [ ] **Step 1: Initialize the artifact workspace**

```powershell
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' 'C:\Users\bolub\.codex\plugins\cache\openai-primary-runtime\presentations\26.805.11740\skills\presentations\container_tools\setup_artifact_tool_workspace.mjs' --workspace 'C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final'
```

- [ ] **Step 2: Implement the minimal slide module**

Use shared helpers only for repeated title/footer/image placement. Keep each slide’s composition explicit so the layouts can vary naturally. Create one native throughput chart and one simple workflow diagram; use extracted footage for result slides.

- [ ] **Step 3: Export and structurally inspect**

Run the module with the bundled Node.js runtime, confirm the output exists and is non-empty, and inspect the presentation structure for exactly 14 slides and zero notes parts.

### Task 4: Render, inspect, and correct

**Files:**
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\rendered\slide-*.png`
- Create: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\montage.png`
- Modify: `C:\Users\bolub\.codex\visualizations\2026\08\06\019fd85f-592b-75b0-a3e6-bdd44bd8dedd\easyrtg-presentation-worktree\.codex_artifacts\easyrtg-final\build_presentation.mjs` as needed
- Replace: `EasyRTG_Final_Submission_Shivam_Chavan.pptx` after each fix

**Interfaces:**
- Consumes: the generated PowerPoint.
- Produces: the final visually verified deck.

- [ ] **Step 1: Run mechanical QA**

Run `slides_test.py` and fail the task on overflow, clipping, unresolved placeholders, or slide-count mismatch.

- [ ] **Step 2: Render every slide**

Use `render_slides.py`, then build a montage for flow review.

- [ ] **Step 3: Inspect all 14 full-size renders**

Check title wrapping, body readability, image crops, chart labels, consistent page numbers, color semantics, and evidence footers. Fix every unintended overlap or weak crop and repeat the render.

- [ ] **Step 4: Final evidence and notes audit**

Recompute displayed metrics against source files, inspect the `.pptx` package for notes, and confirm only project/user sources appear in the visible evidence labels.

- [ ] **Step 5: Deliver**

Return only the final PowerPoint with a short summary of the narrative, verified metrics, light theme, and no-notes status.

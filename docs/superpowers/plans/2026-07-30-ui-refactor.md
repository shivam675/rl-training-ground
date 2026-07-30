# UI refactor — plan and progress

Decided 2026-07-30. Goal: **a quiet, premium, light-first desktop UI** whose journey is
chat-first — you describe a behaviour, the assistant authors the environment, and the result is
always visible beside the conversation.

---

## START HERE — state as of 2026-07-30

**P1 (tokens) is landed.** P2 onward not started. Design is settled and the mockups are committed.

Baseline to hold green at the end of every phase (measured 2026-07-30, before any change):

```
cd frontend/rtg-flutter-app && flutter analyze && flutter test
# No issues found!  ·  12 tests passed
```

**The target is `docs/ui/mock.html`** — three screens (first run · reward authored · training
running) at 1440×900, self-contained apart from a Google Fonts link. `docs/ui/options.html` is the
options sheet the decisions below were picked from; keep it, it is the record of what was rejected
and why. Neither is wired into the app — they are static reference.

Render them to PNG with:

```bash
chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --user-data-dir=/tmp/cprof --force-device-scale-factor=2 --virtual-time-budget=8000 \
  --window-size=1440,2700 --screenshot=all.png <url-to-mock.html>
```

Two gotchas that cost time: `--user-data-dir` is **required** or `--screenshot` fails with
"Access is denied", and a `file://` URL renders but the in-app browser pane will not screenshot it —
serve the folder over localhost instead.

---

## The reference

A Lovable/Bolt-style builder: chat pane left, live artifact right, near-monochrome light theme,
one dark pill for the primary action, hairline-or-less chrome. That layout maps onto EasyRTG
almost exactly, because the app already *is* "talk to the agent, look at what it built" — the
current UI just buries that under a 9-destination rail, a breadcrumb header, a per-panel title bar
and a status bar, four stacked horizontal bands before any content.

---

## Decisions

### D1 — Geist + Geist Mono

Both are in `google_fonts` 8.1.0 (verified). But `pubspec.yaml` bundles `google_fonts/` as an
**asset folder** so typography works offline, and it currently holds only Inter and JetBrains Mono
`.ttf`. **Dropping the Geist files in is part of P1** — skip it and the app silently falls back to
network fetch, then to Segoe UI, and nobody notices until it ships.

### D2 — Shadow-only surfaces, no card borders

Panels separate from the canvas by elevation, not outline. This is what removes the
card → row → expression-field nesting that makes the current Rewards page read as boxes inside
boxes.

### D3 — 16px dot-grid canvas

The right pane's backdrop. Without a texture a light theme reads as unfinished white; a dot grid
reads as a workbench and, unlike the gradient wash option, does not tie to an accent colour.

### D4 — Ink-only accent, zero hue

`#15181C` for the primary action, selection and emphasis. **The semantic four
(`EasyColors` success / warning / danger / info) become the only hue in the app** — which is the
point: an audit verdict beside a reward term should be the thing your eye lands on.

Amber was rejected specifically because `EasyColors.light.warning` is `#B7791F`, the identical
hue. Adopting amber as the accent would have made every "Review: …" verdict indistinguishable from
ordinary accent text.

### D5 — Chart series use a neutral ramp, not colour

D4 removes the hue that told series apart. The replacement is a grey→ink ramp (total at full ink
and 2.4px, terms at `#69717B` / `#9AA1AA` / `#C3C8CE`, dashed for the middle two) plus **an inline
`name value` label at each line's end**. You read the series by name, not by swatch — which is
better anyway on a per-term reward breakdown where the term names are the information.

### D6 — Delete the rail and the dock

`AppPage` goes 9 → 5 canvas tabs (Robot · Environment · Reward · Training · 3D View). Settings,
Logs, Diagnostics and project New/Open/Save move into the logo menu, which already exists as
`_AppMenuButton` in the rail and just needs rehoming. The assistant dock goes because chat is
permanently the left pane — the dock was already suppressed on Home for exactly that reason
(`home.dart`: *"On Home the assistant IS the main canvas, so the right dock would be a redundant
second copy"*). The refactor makes every page Home-shaped, so the dock is redundant everywhere.

### D7 — Keep the OS title bar

A custom title bar would need `bitsdojo_window` plus a `windows/runner/main.cpp` change, and makes
snap/maximise edge cases ours. Not worth a new native dependency for 32px. The app bar sits under
the standard Windows chrome and **draws no window buttons of its own** — `docs/ui/mock.html`
reflects this.

### D8 — Light-only for now; delete the accent picker

Dark mode keeps working and inherits the new tokens, but is not polished this pass — shadow-only
surfaces barely read on dark, so a good dark theme needs borders back and is its own design job.
The 8-preset accent picker (`accentPresets`, `ThemeController.setAccent`, its Settings section and
the `theme_accent` pref) is **deleted**, because D4 makes it meaningless.

---

## Design tokens — the concrete values P1 implements

```
canvas    #F4F5F7      surface   #FFFFFF      sunken   #F1F2F4
line      #E7E9EC      line2     #DCE0E4
ink       #15181C      txt       #2B3138      txt2     #69717B      txt3  #9AA1AA
semantic  success #1F9D57   warning #B7791F   danger #D23B30        (unchanged)

shadow-sm   0 1px 2px rgba(16,20,26,.05)
shadow-md   0 1px 2px rgba(16,20,26,.04), 0 8px 24px -10px rgba(16,20,26,.14)
shadow-lg   0 1px 2px rgba(16,20,26,.04), 0 18px 44px -16px rgba(16,20,26,.18)
dot grid    radial-gradient(#D6DADF 1px, transparent 1px) @ 16px

radii       8 (inputs/chips) · 12 (tiles) · 14 (cards, composer) · 999 (pills)
type        11/600/+.09em upper (micro) · 12.5 · 13.5 (body) · 15/620 (card title) · 25/620 (stat)
```

Layout: app bar 52 · chat pane 452 fixed · canvas padding 16/24/24 · tab strip 38 · card radius 14.

---

## Invariants — a phase that breaks one is wrong

1. `flutter analyze` clean and `flutter test` green at the end of every phase.
2. **No behaviour change.** This is a presentation refactor: no API calls added or removed, no
   config semantics touched, no change to what the agent can do.
3. Every destination reachable today stays reachable — the rail's contents move, they do not
   disappear.
4. The semantic four stay reserved for meaning. Never use `success` green as decoration now that
   it is the only green in the app.
5. Deletions land *with* their replacement in the same phase, never before it.

---

# P1 — Tokens

### Steps

- Rewrite `lib/src/theme/app_theme.dart`: Geist via `GoogleFonts.geistTextTheme`, `monoStyle` to
  `GoogleFonts.geistMono`, `cardTheme` drops its `BorderSide` for elevation + shadow, `primary`
  becomes ink, radii per the table above.
- Extend `lib/src/theme/tokens.dart` with the three shadow constants so panels stop hand-rolling
  them. **The dot grid is deliberately NOT here** — CSS gets it from a `radial-gradient`, Flutter
  needs a `CustomPainter`, and nothing renders a canvas until P3. Building it now would be a
  painter with no caller. It lands in P3 with the surface that uses it (`Canvas.drawPoints` with
  `PointMode.points` — one draw call, not ~3500 `drawCircle`s).
- Add Geist + Geist Mono `.ttf` to `frontend/rtg-flutter-app/google_fonts/` (see D1).
- Delete `accentPresets` / `AccentPreset` and `ThemeController.setAccent` + the `theme_accent`
  pref (D8). `buildAppTheme`'s `seed` parameter goes with them.
- `easy_colors.dart` is **unchanged**.

### Check

Every screen restyles at once with no structural change. Confirm the Rewards panel still shows its
audit verdict in `success` green and that nothing else in the app is green.

### Status: LANDED 2026-07-30

`flutter analyze` clean, **14 tests pass** (12 + 2 new). Verified by running the real Windows app:
canvas `#F4F5F7`, Geist and Geist Mono rendering, ink logo tile, neutral rail selection, and the
only hue on screen is the green "Connected" dot — D4 working as intended.

Findings worth not re-deriving:

- **The font folder got 6× smaller, not bigger: 2.6 MB → 436 KB.** Geist's static instances are
  ~73 KB against Inter's ~415 KB. Inter and JetBrains Mono were deleted, not kept alongside.
- **`allowRuntimeFetching` was already `false`** (`main.dart:18`). Combined with the fact that a
  failed asset lookup is *caught and only `debugPrint`ed* (`google_fonts_base.dart:181`), a
  misnamed file un-brands the app with no error anywhere. That is why `widget_test.dart` now pins
  the six filenames.
- **Filename convention:** `<Family>-<Weight>.ttf`, weight from `_fontWeightToFilenameWeightParts`
  (w500→`Medium`, w600→`SemiBold`). The family for the mono is **`GeistMono`, no space** — the
  dartdoc on `toApiFilenamePrefix` claims `"Roboto-400"`, which is stale and wrong.
- **Geist ships no italic on Google Fonts.** The single italic use (`_NoticeRow`) gets a
  synthesized oblique. Fine; don't go hunting for the missing file.
- **`FontWeight.w800` had no asset.** There was exactly one use — the `EasyRTG` brand text in
  `home.dart` — dropped to w700 rather than bundling a 73 KB ExtraBold for one string. w700 is the
  ceiling now; anything heavier silently falls back.
- **`accentPresets`, `AccentPreset`, `setAccent`, `accentIndex`, `_AccentSwatch` and the
  `theme_accent` pref are gone.** `buildAppTheme` lost its `seed` parameter. A stale `theme_accent`
  in an existing install is simply ignored.
- **`Radii` was dead** before this phase (105 `Insets.` uses, zero `Radii.`), so rescaling it to
  8/12/14/999 was free.

**Screenshotting the Flutter Windows window** — worth keeping, it cost a dozen attempts. `PrintWindow`
(even `PW_RENDERFULLCONTENT`) returns the frame with a **blank client area**, because Flutter renders
to a GPU swapchain. `SetForegroundWindow` is *refused* for a background caller, so `CopyFromScreen`
silently captures whatever is actually on top. What works: `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)`
**plus a 2px resize to force a repaint**, then `CopyFromScreen`, then restore `HWND_NOTOPMOST`. Without
the resize you get a stale surface that reads as pure white — which looks exactly like a crashed app
and is not one.

### Status: NOT STARTED

---

# P2 — Shared components

### Steps

`lib/src/widgets/common.dart` (492 LOC) — 12 call sites inherit for free:

- `Panel`: lose the border and the header divider, gain `shadow-lg`.
- `SectionHeader`: grey uppercase micro-label, not `scheme.primary`.
- `StatChip`: neutral tile, drop the accent tint.
- `EmptyState`: restyle to the dashed drop-zone in the mock.
- Add the pill button styles (ink filled / outline+lift).

### Check

`flutter test` — the two RewardPanel tests exercise this path and must stay green.

### Status: NOT STARTED

---

# P3 — Shell

**The only phase with UX risk and the only one that breaks a test.**

### Steps

- `lib/src/nav.dart`: `AppPage` 9 → 5. Settings / Logs / Diagnostics leave the enum.
- `lib/src/home.dart` (603 → ~300): delete `_Header` and `_StatusBar`, add the app bar, restructure
  the body from `Row(rail, Column(header, page, status), dock)` to `Row(chat, canvas)`.
- **Delete** `lib/src/widgets/app_rail.dart` (355) — rehome `_AppMenuButton` into the app bar logo.
- **Delete** `lib/src/agent/assistant_dock.dart` (255) — see D6. `setDockExpanded` call sites in
  `dashboard_panel.dart` go with it.
- Add `WM_GETMINMAXINFO` to `windows/runner/win32_window.cpp` for a ~1100px minimum width. Not on
  the Flutter Windows docs page, ~8 lines. `main.cpp:29` opens at 1280×720, which the layout holds;
  below ~1100 the chat pane has to collapse.

Ripple is small and known: `AppPage` is **38 refs across 7 files** — `home.dart` 16, `app_rail.dart`
5 (deleted), `dashboard_panel.dart` 7, `evaluation_panel.dart` 3, `home_panel.dart` 3,
`training_panel.dart` 3, `nav.dart` 1.

### Check

**`widget_test.dart:63 'EasyRTG shell renders'` will fail** — it asserts
`find.text('Setup & Status')`, a panel title this phase deletes. Update it to assert the new shell
(app bar + chat + canvas tabs). Every other test is behavioural and should stay green untouched;
if one breaks, that is a real regression, not churn.

### Status: NOT STARTED

---

# P4 — Chat pane

### Steps

`lib/src/agent/chat_view.dart` (1007): assistant replies lose the bubble and render as plain text
(user messages keep theirs), composer becomes the floating rounded box with the ink send button,
header slims to name + status dot + overflow, tool rows adopt the shadow-chip style.

### Check

`chat_stream_test.dart` is 8 tests that find by **text and button**, so a pure restyle passes
untouched — as long as no label is renamed. Treat any failure there as a real break.

### Status: NOT STARTED

---

# P5 — Canvas content

### Steps

- `dashboard_panel.dart` (401) becomes the **Environment** tab; its setup checklist becomes the
  horizontal stepper at the top of the canvas card.
- `home_panel.dart` (371): the `_HomeSidebar` animated collapse/expand split goes away entirely —
  the canvas tabs replace it.
- `TelemetryCharts` (`training_panel.dart:768`): neutral ramp + inline end labels per D5.

### Check

Telemetry test at `widget_test.dart:45` guards the bounded-window cursor logic — must stay green.

### Status: NOT STARTED

---

## Deferred, by decision

- **Dark theme parity** (D8) — needs borders back on dark; its own pass.
- **Custom title bar** (D7) — revisit only if the OS chrome genuinely grates.
- **Resizable chat pane** — fixed 452px for now; a drag handle is easy to add later and nobody has
  asked.
- **`app_state.dart` decomposition** — the 870-LOC god object is untouched here on purpose. This
  refactor is presentational; splitting state is a separate, riskier job.

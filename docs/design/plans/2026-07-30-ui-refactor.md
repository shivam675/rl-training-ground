# UI refactor — plan and progress

Decided 2026-07-30. Goal: **a quiet, premium, light-first desktop UI** whose journey is
chat-first — you describe a behaviour, the assistant authors the environment, and the result is
always visible beside the conversation.

---

## START HERE — state as of 2026-07-30

**P1 (tokens), P2 (shared components) and P2.5 (four fixes from first real use) are landed.** P3 onward not started. Design is settled and
the mockups are committed.

**Read P2's Status section before writing any UI test here.** P2 shipped with three visible bugs
while the whole suite was green, because every test rendered Home from an empty `AppState`.

Baseline to hold green at the end of every phase (measured 2026-07-30, before any change):

```
cd frontend/rtg-flutter-app && flutter analyze && flutter test
# No issues found!  ·  16 tests passed  (was 12 before this refactor)
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

### D9 — The setup checklist becomes a persistent stepper in the app bar

Decided 2026-07-30 after seeing P2 running. The 5 steps (Robot · Observations · Actions · Reward ·
Train) compress to a thin always-visible strip in the app bar, each step clickable. This differs
from `mock.html`, which puts the stepper inside the canvas card — there it scrolls away with the
content, and "where am I in this workflow" is exactly the thing that should never scroll away. It
also costs no canvas height. Per-item detail lives in the tab each step links to.

### D10 — Reward-authoring errors stay out of this refactor

The first real run rejected a reward because the agent wrote `base_angular_velocity[2]` instead of
`yaw_rate` — the observation-name trap docs/architecture_notes.md calls the single most likely authoring mistake. It
is **not** this plan's job. Invariant 2 holds: presentation only. The rejection already surfaces in
three places, and a real fix belongs in the agent prompt or the compiler, tracked separately. Do not
let it creep in here.

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

**Do not try to screenshot this app from a script.** Four independent things defeat it, and chasing
them burned most of a session across P1 and P2:

1. `PrintWindow` (even `PW_RENDERFULLCONTENT`) returns the window frame with a **blank client area** —
   Flutter renders to a GPU swapchain the GDI path cannot read.
2. `SetForegroundWindow` is **refused** for a background caller, so `CopyFromScreen` silently grabs
   whatever is actually on top. It will happily hand you a screenshot of the user's browser.
3. This box runs at **200% display scaling** (native 3840×2160, desktop reports 1920×1080). A
   DPI-*unaware* caller reads virtualized rects, so `GetWindowRect` says `1280×720` for a window that
   is really `2560×1440`. Call `SetProcessDpiAwarenessContext(-4)` **before** any window/screen API or
   every coordinate lies. With it, the app correctly opens at 2560×1440 real px — the runner's
   `Scale(size, dpi/96)` in `win32_window.cpp:139` is fine, contrary to first appearances.
4. **`SetWindowPos` on this window minimizes it** — reproducibly, at several sizes, with
   `SWP_NOACTIVATE` set. Once minimized it will not come back via `ShowWindow(SW_RESTORE)` or
   `SW_MAXIMIZE`; the process stays alive and `Responding`, and `EnumWindows` shows the only visible
   top-level window stuck at the minimized rect. Restart the app.

A stale unpainted surface also reads as **pure white**, which looks exactly like a crashed app and is
not one. If you need to see a change, **run the app and look at it yourself** — that is cheaper and
more honest than automating this.

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

### Status: LANDED 2026-07-30

`flutter analyze` clean, **15 tests pass**.

**P2 shipped broken and the test suite said it was fine.** Running the app surfaced three faults that
14 green tests missed. This is the most important thing in this document: *green tests were not
evidence.* The suite only ever rendered Home with an **empty** `AppState` at 1600×1000, which is the
one state where none of these bugs appear.

1. **`Panel` lost its `Material` ancestor** — `Card` provided one for free; a `DecoratedBox` does not.
   Seven `ListTile background color or ink splashes may be invisible` assertions, and silently every
   `InkWell` ripple inside a panel stopped rendering. Fixed with
   `Material(type: MaterialType.transparency)` between the shadow and the clip.
2. **`RenderFlex overflowed by 250 pixels`** in the collapsed Training panel, plus a second 131px
   overflow. Root cause was **not** layout: `trainingBlockers()` spliced `configProblems` in verbatim,
   and a config problem is a full expression-compiler rejection running to several hundred
   characters. Rendered into `'Next: ${blockers.first}'` inside a fixed-height panel, it could not
   fit. `trainingBlockers()` now returns *short actionable steps only* — `fix 1 configuration
   problem` — and the full text stays where it is actionable and scrollable (Setup & Status, and
   beside the offending term in Rewards). Nothing is hidden.
3. The blocked branch of `TrainingControlCard` was a `Padding + Column`. `_HomeSidebar` gives it a
   **fixed** height, so anything that can grow must scroll: it is a `ListView` now, matching the
   unlocked branch, plus a "Review the reward" link so the problem is one click from its fix.

**The regression test that should have existed** is now `widget_test.dart` → *"Home survives a robot
loaded with a long validator problem"*: robot loaded, obs/actions/rewards enabled, one realistic
several-hundred-character `configProblems` entry, at **1280×720** (what `main.cpp` actually opens) —
and it asserts `tester.takeException()` is null, because a `RenderFlex` overflow surfaces as an
exception rather than a failed finder. Verified it genuinely catches the bug: reverting
`trainingBlockers()` to the splicing version makes it fail. **When adding UI tests, populate the
state — an empty AppState exercises almost none of the layout.**

What changed, and the one that mattered:

- **`Panel` is no longer a `Card`.** It is a `DecoratedBox` carrying `Shadows.lg`, because Material's
  `elevation:` only approximates the designed shadow. `Card` and `cardTheme` remain for
  `evaluation_panel.dart:306` and `robot_panel.dart:277`.
- **The header hairline stays**, contradicting this plan's original wording. `mock.html`'s `.chead`
  has one, and it does a different job from the outline that was removed: it separates two regions of
  the *same* surface rather than separating the surface from the canvas.
- **`SectionHeader` lost its trailing rule** and went grey — it was drawing a line across every panel
  carrying no information, in the accent colour, making the quietest element the loudest.
- **`StatusDot` is a neutral pill with a coloured dot.** Previously the fill, border and text were all
  tinted, so a routine "Connected" was as loud as a real warning. The dot alone carries the state.
- **`StatChip` lost its accent tint**; its `color` parameter now means *semantic* (one caller passes
  `colors.success` for mean reward) and unset means ink.
- `EmptyState` became the dashed drop-zone from the mock, which needed `DottedBorderBox` — Flutter has
  no dashed `BorderSide`, so it walks the `RRect` path with `computeMetrics()` and strokes 6-on/5-off.
- `CopyableValue` dropped its outline for a sunken fill.

**The one real bug this surfaced.** Raising `Panel`'s padding from `all(8)` broke
`widget_test.dart:63` with `RenderFlex overflowed by 24 pixels` — exactly the 12px-per-panel increase
across the two panels in `_HomeSidebar`. Root cause was **not** the padding: `_HomeSidebar._collapsed`
was a hardcoded `58.0` tuned to a previous header size, so the collapsed Training panel could no
longer fit its own chrome. Fixed at the root — `Panel.chromeHeight` is now a derived constant and
`_collapsed` is expressed in terms of it, so the next padding change cannot silently repeat this.
`Panel` also clamps its header to the available height, so a squeezed panel clips instead of
asserting. That guard is cheap and worth keeping: `_HomeSidebar` animates the height, so it passes
through small values every time it opens or closes.

### Status: NOT STARTED

---

# P2.5 — Six fixes from the first real use

### Status: LANDED 2026-07-30

`flutter analyze` clean, **16 tests pass**. Every one of these was found by *using* the app. None
were found by the test suite, which was green throughout.

**1 · Robot Setup was an information dump.** Loading a robot rendered 45 joint rows, the world
physics knobs and the URDF warning list, all expanded, before the user had done anything. Restructured
around progressive disclosure: what stays open is what is *actionable* or answers "did the right robot
load?" — the name/joints/movable tiles, the resolved path, the dynamics banner (it has an auto-fix
button). Everything else sits behind a `_Disclosure` (an `ExpansionTile`, no new dependency): Joints,
URDF warnings, World, Load-from-path. The no-robot state is now an `EmptyState` with **Browse files**
wired to browse-then-load as one gesture — picking a file and then hunting for a separate Load button
was the step everyone forgot.

Also killed `_JointTypeBadge`'s four hardcoded hues (blue/violet/green/grey for revolute/prismatic/
continuous/fixed). That is colour as decoration, which D4 forbids. The only distinction that matters
is *can the policy drive this joint*, and weight carries it.

**2 · Toasts never dismissed.** Not a duration bug — `duration` was already 4s. Flutter creates the
auto-dismiss timer **inside `ScaffoldMessengerState.build`** (`scaffold.dart:617`) and only when the
entry animation has completed, `route.isCurrent`, and `snackBar.persist` is false; `accessibleNavigation`
also changes the transition path so the controller may never report completed. Rather than guess which
precondition fails on this machine, `home.dart` now owns the dismissal with its own `Timer` calling
`controller.close()`. `close()` on an already-closed controller is a no-op, so it is safe alongside
the framework's timer.

**3 · The yellow banners restated the checklist.** "No observation sources enabled" sat directly under
an unticked "Observations enabled". On Home they are now gated on `basicsDone` — while the four basics
are incomplete every `configProblem` *is* the checklist in prose; once complete, a remaining problem is
substantive (a rejected expression, an unknown observation key) and earns its space. On
`observation_action_panel.dart` both banners are **deleted**: a page whose entire job is enabling things
does not need to announce that nothing is enabled.

**4 · Page transitions were imperceptible.** 220ms with a 0.035-of-a-pixel slide read as a flicker.
Now a hand-rolled shared-axis feel — 300ms in / 180ms out, fade + a 0.035 rise + a 0.985→1 scale, with
a `layoutBuilder` that stacks the outgoing page underneath instead of shoving the incoming one down.
No `animations` package needed.

**New guard:** *"Robot Setup keeps reference detail collapsed until asked"* — asserts the 45 joint
rows, the gravity field and the warning text are **absent** on first render, then present after
tapping Joints.

**5 · Robot Setup went two-column** (second pass, from a target layout the user drew). Identity on
the left — name / joints / **movable** tiles, the resolved path, and the swap button — against the
drillable detail on the right. *Movable* was added because it is the number that actually matters:
go1 reports 45 joints and only 12 of them move, and the policy can only drive those 12. Below 880px
the columns stack; that threshold is roughly where the joint table stops fitting beside a 340px
identity column. The four disclosures are grouped into one `_DisclosureGroup` surface with hairlines
between them, so they read as a list rather than four floating cards.

**6 · Outlined buttons were invisible.** They used `line2` (`#DCE0E4`) — a hairline that is correct
for separating regions and vanishes as a button edge. Added `_Neutrals.controlBorder` (`#C5CBD3`)
plus a 1dp lift and w600 label, in the **theme**, so every outlined button in the app gains it rather
than just the one that prompted the complaint.

---

# P3 — Shell

**The only phase with UX risk and the only one that breaks a test.**

### Steps

- `lib/src/nav.dart`: `AppPage` 9 → 5. Settings / Logs / Diagnostics leave the enum.
- Build the app-bar stepper (D9) from `DashboardPanel`'s existing `_SetupStep` list — the data and
  the per-step `AppPage` links already exist, only the presentation changes.
- Add the dot-grid canvas painter deferred from P1: `Canvas.drawPoints` with `PointMode.points`,
  `strokeCap: round`, one draw call. Wrap in a `RepaintBoundary` — it is a static backdrop behind a
  pane that repaints on every telemetry tick.
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

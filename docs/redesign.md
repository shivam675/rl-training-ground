# EasyRTG — UX Redesign Spec

**Status:** proposal, awaiting approval · **Author:** design pass 2026-06-14
**Direction (agreed):** restructure information architecture + remove
duplication + premium polish · assistant becomes a **persistent co-pilot dock**.

This spec is the contract for the redesign. It is written to be executed
slice-by-slice (see [§9](#9-implementation-plan-slices)). Nothing here removes a
capability — it relocates, de-duplicates and polishes what already exists.

---

## 1. Goal & principles

The app today is organized around the **engineer's 9 technical stages**, exposed
as 9 flat, equal nav tabs, with a 4-pane cockpit on top — and several panels
rendered in two places at once. The result: *nothing tells the user where they
are or what to do next.*

We re-organize around the **user's journey**:

> **Bring a robot → say what it should do → watch it learn → check it → iterate.**

Principles:

1. **One job per surface.** No panel appears in two places.
2. **The rail is the map.** Grouping (Build → Train) shows progress and gates.
3. **The assistant is always there**, not a destination you visit.
4. **Default to the next action.** Empty states are CTAs, not dead ends.
5. **One visual language.** Semantic color/spacing tokens, no stray hex.

What we are **not** doing: changing the backend, the training/eval logic, the
agent tools, or the project/.rtg format. This is a frontend IA + polish pass.

---

## 2. Information architecture — before / after

### Before (today)

```
NavigationRail (9 flat, equal tabs)            Home (index 0 = "Simulation"):
  0  Simulation   ← actually a dashboard       ┌──────────┬──────────┐
  1  Robot Setup                                │ Viewport │  Chat    │  ← Chat #1
  2  Obs / Action                               ├──────────┼──────────┤
  3  Rewards                                    │ Setup &  │ Training │  ← Training #1
  4  Training      🔒                            │ Status   │ (full!)  │
  5  Evaluation    🔒                            └──────────┴──────────┘
  6  Agents        ← Chat #2 (same as quadrant)
  7  Settings                                   Tab 4 = Training #2 (identical)
  8  Logs                                       Tab 6 = Chat #2 (identical)
```

Problems: Training rendered twice, Chat rendered twice, viewport never
full-screen, "Simulation" misnamed, no grouping, no guided start.

### After (proposed)

```
Custom AppRail (grouped journey)        Shell layout (every screen):
  ⌂  Home                               ┌────┬───────────────────┬──────────┐
  ── BUILD ───────────                  │    │                   │          │
     Robot                              │Rail│   Active page     │ Assistant│
     Obs / Action                       │    │   (one job)       │   dock   │
     Rewards                            │    │                   │(collapsi-│
  ── TRAIN ──────────  🔒 until valid    │    │                   │  ble)    │
     Training                           └────┴───────────────────┴──────────┘
     Evaluation
  ───────────────────                   • Agents tab removed → it's the dock.
     Settings                           • "Simulation" → "Home" (cockpit).
     Logs                               • Viewport gets real estate on Home.
  [theme toggle]                        • Same chat everywhere, 1 conversation.
```

**Nav targets drop 9 → 8** (Agents folds into the dock), and they're now grouped
into *Home · Build · Train · Tools*. The Train group carries the lock.

### Page identity refactor

Replace the magic-number `int` index (`navIndexProvider`, `_gatedIndices = {4,5}`,
`go(1)`, `state = 7`, …) with a typed enum so reordering is safe and readable:

```dart
enum AppPage { home, robot, obsAction, rewards, training, evaluation, settings, logs }
```

`navIndexProvider` becomes `StateProvider<AppPage>`. Gating = `{training, evaluation}`.
Every `go(4)` / `state = 7` call site becomes `AppPage.training` / `.settings`.

---

## 3. The assistant co-pilot dock (the central new pattern)

Today `AgentsPanel` is a `ConsumerStatefulWidget` that *owns* its message list,
streaming buffers and scroll — so the dashboard quadrant and the Agents tab are
**two separate instances** reconciled only through `SharedPreferences`, and
tool-call rows are lost on navigation.

**Fix: lift chat state into a controller, render it in one docked surface.**

### 3.1 `ChatController` (new) — `lib/src/agent/chat_controller.dart`

A `ChangeNotifier` provider that owns everything `_AgentsPanelState` holds today:
`messages`, streaming/think buffers + flush timer, the `<think>` parser, history
windowing, persistence, `sendChat`, `confirmTool`, `clearChat`. The widget layer
becomes pure rendering. One conversation, survives navigation, single source of
truth.

```dart
final chatControllerProvider = ChangeNotifierProvider((ref) =>
    ChatController(ref.read(appStateProvider)));
```

Migration is mechanical: move methods out of `agents_panel.dart` into the
controller; the existing message widgets (`_MessageBubble`, `_ToolActivityRow`,
`_NoticeRow`, `_ThinkingSection`, `_TypingRow`) move to
`lib/src/agent/chat_widgets.dart` unchanged.

### 3.2 `AssistantDock` (new) — `lib/src/agent/assistant_dock.dart`

Lives in the shell `Row` to the right of the page, on **every** screen:

```
body: Row[ AppRail | VDivider | Expanded(Column[Header, Page, StatusBar]) | AssistantDock ]
```

- **Expanded** (default ~360px): header (avatar · "Assistant" · connection dot ·
  copy/clear/refresh), message list, composer — i.e. today's `AgentsPanel` body.
- **Collapsed** (~52px): a vertical edge strip — avatar button + unread badge.
  Click to expand. `AnimatedContainer` width transition (220ms, easeOutCubic).
- **Auto-collapse** when window width < ~1100px (so narrow windows keep the page
  usable); user toggle always wins and persists (`dock_expanded` pref).
- Connection dot already exists (`agentConnected`); keep it in the dock header.

### 3.3 Removals
- Delete the chat quadrant from Home (`_pageFor` index 0, top-right).
- Delete the **Agents** rail destination + its `_pageFor` case.
- `agents_panel.dart` shrinks to a thin `AssistantDock` consumer of `ChatController`.

---

## 4. Screen-by-screen — before / after

### 4.1 Home (was "Simulation", index 0)

| | Before | After |
|---|---|---|
| Identity | "Simulation" but actually a 4-quadrant dashboard | **Home / cockpit**, renamed + new icon (`space_dashboard`) |
| Chat quadrant | full chat | **removed** → dock |
| Training quadrant | full 900-line `TrainingPanel` crammed in ¼ screen | **compact `TrainingControlCard`** (start/stop · progress · reward sparkline · "Open Training →") |
| Layout | 2×2 quadrants | **viewport (large, left) + status/control sidebar (right)** |
| Empty (no robot) | one of four panels shows a CTA | **hero onboarding** (see §5) |

After, configured state:

```
┌───────────────────────────────┬──────────────────────────┐
│                               │  Setup ✓ Ready            │
│       3D Viewport             │  (collapses when done)    │
│       (large, live)           │  ───────────────────────  │
│                               │  Training control         │
│   pause · step · reset        │  ▸ PPO · 10k  [Start]     │
│                               │  ▸ progress ▕▔▔▔░░░ 41%   │
│                               │  ▸ reward  ╱╲╱▔ +12.3     │
└───────────────────────────────┴──────────────────────────┘
```

### 4.2 Robot Setup (Build)
Largely unchanged — it's already focused and good (load form, fixed-base/plane,
gravity, dynamics auto-fix banner, joint table + CSV). Polish only: semantic
colors for the warning/dynamics banners; the load form's "Reset world / Apply
gravity" stay here (this is their natural home).

### 4.3 Obs / Action (Build)
Unchanged structurally — this is the **canonical** obs/action editor. It keeps
"Zero action" / "Safe random" (their real home). Polish: semantic colors on the
config-problem banner.

### 4.4 Rewards (Build)
- Keep the component editors, weights, params, custom-Python editor + Validate.
- **Fix the result hack:** replace `state.message.startsWith('Reward')`
  (`reward_panel.dart:19`) with a structured `state.lastRewardResult`
  (`{total, formula, terms}`) set by `testReward()`. The result card renders
  from that, not from sniffing the status string.

### 4.5 Training (Train, gated)
Strip the duplication; keep the real controls.

| Section | Before | After |
|---|---|---|
| Pre-flight buttons | Zero action · Safe random · Test reward · Save env (all live here too) | **removed** — they live in Obs/Action & Rewards |
| Env spaces | "Observation builder" + "Action builder" read-only lists (dupes of tab 2) | **one-line `EnvSummaryStrip`**: `obs 23 · act 8 · reward 3 terms [Edit in Build]` |
| Save | manual "Save env" button | **removed** (patches auto-persist; see §6) |
| Controls | algo · timesteps · 6 buttons | algo · timesteps · **Start/Stop**; advanced params in a collapsible |
| Keep as-is | status chips, progress, telemetry charts, advisor, tuning card | unchanged |

Net: Training becomes "configure the run + watch it", not a second setup screen.

### 4.6 Evaluation (Train)
Mostly unchanged (run list, scope toggle, evaluate, compare, export, watch-live).
Two improvements:
- **Overlaid reward curves** in the compare dialog (was a TODO in `plan.md` §3) —
  premium, high-value.
- **Gating question** — see [§10](#10-open-questions). Eval is about *existing
  artifacts*; arguably it shouldn't be locked by the *current* env's validity.

### 4.7 Settings
Unchanged in function; polish: group into cards (Appearance · Viewport · Agent ·
Diagnostics) with consistent spacing; semantic colors on the capability result.

### 4.8 Logs
Unchanged for now. (Stretch: tail the backend log file + agent-event history so
"Logs" earns its name.)

---

## 5. Onboarding / first-run

The #1 "what do I do" fix. Home adapts to state:

**No robot loaded → hero, not quadrants:**

```
        ╭────────────────────────────────────────────╮
        │            Let's train a robot.            │
        │                                            │
        │   [ ⬆ Load a URDF ]   [ ◧ Pick a sample ▾ ]│
        │                                            │
        │   …or tell the assistant your goal:        │
        │   ( Walk )( Balance )( Stand )( Reach )    │  ← chips → dock
        ╰────────────────────────────────────────────╯
            Setup progress  ◔  0 / 5
```

- **Sample gallery**: one-click load of bundled URDFs (r2d2, etc. from
  `pybullet_data`) so a new user sees a robot move in seconds.
- **Goal chips** hand the goal straight to the `ChatController` ("Make this robot
  walk") — surfacing the killer feature instead of hiding it in 11px grey text.
- **Setup progress ring** replaces the flat checklist as the hero; expands to the
  existing checklist rows (`DashboardPanel` logic, reused).
- **Unlock moment**: when the 5th item completes and Train unlocks, a brief
  celebratory transition + the rail's TRAIN group animates from locked → live.

---

## 6. The save & gating model

Today there are **three "saves"** and it confuses the flow:

1. **Project save** — `Ctrl+S` / app menu → writes `.rtg`. *(keep — the only
   explicit save)*
2. **"Save env" button** on Training → `POST /env/save_config`. *(remove)*
3. **"Setup saved" checklist item** gating `canStartTraining` via `envConfigSaved`.

Reality: toggling **any** obs/action/reward already calls `patchConfig`, which
sets `envConfigSaved = true` (`app_state.dart:272`). So the manual button mostly
implies work the user didn't need to do.

**Proposal:**
- Remove the "Save env" button.
- Add a quiet **save-state indicator** in the header near the project chip:
  `● Unsaved changes` / `✓ Saved` (project-level, drives off a dirty flag).
- Relabel checklist "Setup saved" → **"Configuration valid"**, driven by
  `configProblems.isEmpty && hasEnabled{Obs,Actions,Rewards}` rather than a
  separate save step. (`canStartTraining` keeps working; we just stop surfacing
  "save" as a user chore.)

---

## 7. Premium visual layer

### 7.1 Semantic color tokens (kills ~15 stray hex)

Today `0xff5fe089` (green), `0xffffc857` (amber), `0xffff6f64` (red),
`0xff4f9cff` (blue) are hardcoded across `dashboard_panel`, `robot_panel`,
`training_panel`, `evaluation_panel`, `agents_panel`, `simulation_panel`,
`observation_action_panel`, `notification_panel`, `common.dart`. They don't adapt
to light/dark or accent.

Add a `ThemeExtension`:

```dart
@immutable
class EasyColors extends ThemeExtension<EasyColors> {
  final Color success, warning, danger, info, positiveReward;
  // light + dark variants built in app_theme.dart; tuned per brightness
}
// usage: Theme.of(context).extension<EasyColors>()!.success
```

Replace every hardcoded status hex with a token. One source of truth, adapts to
theme, instantly more "designed".

### 7.2 Spacing & radius scale
Define `const space1=4, space2=8, space3=12, space4=16, space6=24;` and a radius
scale (`r8`, `r10`, `r12`). Apply consistently (panels currently mix 6/8/10/12/14
padding ad hoc).

### 7.3 Motion (tasteful, building on what exists)
- Dock expand/collapse slide.
- Setup-ring fill + unlock celebration.
- Rail group lock→unlock transition.
- Keep the existing page fade/slide, pulsing dot, copy-check, typing dots.

### 7.4 Rail as a premium component
A custom `AppRail` (replacing `NavigationRail` in `_buildRail`) gives us section
labels (BUILD/TRAIN), per-group lock state, and a refined selected indicator —
things `NavigationRail` can't do natively. Keeps the logo menu + theme toggle.

---

## 8. "Premium features" (capability, not just looks)

Prioritized; ★ = in redesign scope, ☆ = stretch / follow-up.

- ★ **Co-pilot dock** — omnipresent assistant (§3).
- ★ **Goal chips + sample gallery** — seconds-to-first-robot (§5).
- ★ **Reward sparkline + compact training card** on Home (§4.1).
- ★ **Overlaid reward curves** in run comparison (§4.6).
- ☆ **Command palette (`Ctrl+K`)** — jump to any page, load a sample, or fire an
  agent action from one search box. Signature "pro tool" feel.
- ☆ **"Explain this"** — tap a metric/term → the assistant explains it in the dock.
- ☆ **Logs that earn the name** — live backend log tail + agent-event history.

---

## 9. Implementation plan (slices)

Each slice is independently shippable and ends with `flutter analyze` clean (+
the existing backend suite untouched). Ordered to de-risk: polish first, then the
structural moves, then the new surfaces.

| # | Slice | Scope | Risk |
|---|---|---|---|
| 0 | **Design system** | `EasyColors` extension + spacing/radius tokens; replace all hardcoded status hex. No behavior change. | low |
| 1 | **Chat controller + dock** | Extract `ChatController`; build `AssistantDock`; remove chat quadrant + Agents tab. One live conversation. | med |
| 2 | **Nav IA** | `AppPage` enum; custom grouped `AppRail` (Home/Build/Train/Tools); rename Home; group-level lock. | med |
| 3 | **De-dup Home + Training** | Home → viewport + sidebar (`TrainingControlCard`, checklist); Training loses dupe lists + pre-flight + Save-env, gains `EnvSummaryStrip`; reward-result state fix; save-state indicator. | med |
| 4 | **Onboarding** | Hero first-run on Home (load/sample/goal chips), progress ring, unlock moment. | low |
| 5 | **Premium ☆** | Overlaid reward curves; command palette; "explain this". (pick per appetite) | low |

Suggested PR boundaries = one slice each. I'll pause for your review after Slice
1 and Slice 3 (the two with visible structural change) before continuing.

---

## 10. Open questions

1. **Evaluation gating.** Today Evaluation is locked by `canStartTraining`. But
   it operates on *existing* runs. Should it be **ungated** (you can always
   inspect/compare/evaluate past models) while Training stays gated? *I lean
   yes.* — **your call.**
2. **Light theme as a first-class citizen?** The app supports it, but the spec's
   polish examples assume dark is primary. Confirm dark-first is fine.
3. **Command palette / sample gallery** — in this pass, or as a fast follow?
4. **Narrow-window behavior** — auto-collapse the dock below ~1100px wide:
   acceptable, or do you want the dock to always stay open?

---

## 11. File-level change summary

**New**
- `lib/src/agent/chat_controller.dart` — lifted chat state.
- `lib/src/agent/chat_widgets.dart` — message bubbles (moved from `agents_panel`).
- `lib/src/agent/assistant_dock.dart` — the dock shell.
- `lib/src/widgets/app_rail.dart` — grouped custom rail.
- `lib/src/panels/home_panel.dart` — new cockpit (replaces the index-0 quadrant grid + `DashboardPanel` host).
- `lib/src/theme/easy_colors.dart` — semantic color extension + tokens.

**Changed**
- `home.dart` — shell `Row` gains the dock; `AppPage` enum; `_pageFor` slims; rail swapped.
- `nav.dart` — `navIndexProvider` → `StateProvider<AppPage>`.
- `app_state.dart` — `lastRewardResult` struct; save-dirty flag; (remove unused `showInspectorOnDashboard`).
- `training_panel.dart` — drop dupe obs/action lists + pre-flight + Save-env; add `EnvSummaryStrip`.
- `reward_panel.dart` — read `lastRewardResult` instead of string-sniffing.
- `dashboard_panel.dart` — logic reused inside the new Home sidebar/onboarding.
- `evaluation_panel.dart` — overlaid curves (Slice 5); gating per Q1.
- panels using status hex — swap to `EasyColors`.

**Removed**
- Agents rail destination + `_pageFor` Agents case.
- Home chat quadrant + Home full-Training quadrant.
- "Save env" button; dead `showInspectorOnDashboard` flag.

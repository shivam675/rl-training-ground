# EasyRTG — Pages

What each page of the application is for, why it exists, and what the user can
do there. The final section describes the ideal user flow through the app.

---

## 1. Simulation (Dashboard)

**Why this page exists:** It is the mission-control view. RL training on robots
is a watch-and-react activity — the user needs the simulation, the agent, the
training state and the robot's structure visible at the same time without tab
switching.

**Benefit to the user:** Zero-navigation overview. A new user lands here and
immediately sees the robot moving, can talk to the agent, and can kick off
training — the entire product in one screen.

**What the user can do:**
- Watch the live PyBullet stream; orbit (drag), pan (right/middle drag), zoom
  (scroll) the camera.
- Pause / step / reset the simulation from the overlay controls.
- Chat with the agent (quadrant 2), start/stop training (quadrant 3), and
  inspect or reload the robot (quadrant 4) — the same panels as the dedicated
  tabs, embedded.

## 2. Robot Setup

**Why this page exists:** Everything downstream (observations, actions,
rewards, training) is derived from the loaded URDF. Loading and sanity-checking
the robot deserves a focused page where problems are caught early.

**Benefit to the user:** Catches bad robots *before* wasting a training run —
missing joint limits, strange inertials and other URDF warnings are surfaced
with copyable messages, and the full joint table shows exactly what the
simulator parsed.

**What the user can do:**
- Pick a `.urdf` file (file browser or path), toggle fixed base / ground plane,
  and load it into the simulation.
- Set gravity and reset the world.
- Review robot stats (name, joint count, source path — all copyable) and URDF
  warnings.
- Inspect every joint: index, name, type (color-coded), limits; copy a single
  joint name or export the whole table as CSV.

## 3. Observation / Action Space

**Why this page exists:** The obs/action specification *is* the interface
between the robot and the learning algorithm. Users need to see exactly what
the policy will observe and control, because a wrong space is the most common
silent killer of training runs.

**Benefit to the user:** Transparency — vector sizes and per-source/per-joint
breakdowns mean "why is my observation 23-dimensional?" never requires reading
backend code.

**What the user can do:**
- Review every observation source (size, placeholder status) and copy its key.
- Review every actuated joint: control mode, limits, max force.
- Sanity-check actuation with one click: zero action and a safe random action.

## 4. Reward Builder

**Why this page exists:** Reward design is the core craft of RL. It gets a
dedicated page because users iterate here more than anywhere else.

**Benefit to the user:** Immediate feedback — test the reward against the
*current* simulation state and see the full formula and per-term values,
instead of discovering a broken reward 10,000 timesteps into training.

**What the user can do:**
- See all reward components with enabled state and weights.
- Run "Test reward" → get the total, the symbolic formula, and per-term
  raw/weighted values (copyable).
- Copy component keys for use in configs or agent conversations.

## 5. Training

**Why this page exists:** Starting, monitoring and stopping runs is the
central loop of the product and needs a home that combines controls with live
status.

**Benefit to the user:** One place to launch a run and know, at a glance,
whether it is alive, how far along it is, and what spaces/builders it is using.

**What the user can do:**
- Start PPO training (algorithm, hyperparams currently defaulted; tunable
  surface is planned), stop a run.
- Watch live status: state message, timestep, observation/action vector sizes
  (copyable stat chips).
- Apply zero/safe-random test actions, save the env config, test the reward —
  pre-flight checks without leaving the page.

## 6. Evaluation

**Why this page exists:** A trained policy is only worth something once it is
measured. Evaluation is split from Training because evaluating is a separate
mental mode — comparing artifacts, not tweaking a live process.

**Benefit to the user:** Will close the loop: run N deterministic episodes on a
saved model, see per-episode returns, and compare runs side by side (the
backend endpoint exists; the UI table and comparison view are the active
roadmap items in plan.md Phase 3).

**What the user can do (today):** Check backend/renderer status. (Planned:
pick any run → run episodes → score table → watch the policy in the viewport →
compare two runs.)

## 7. Agents

**Why this page exists:** The agent is the product's guide and co-pilot. A
full-page chat exists alongside the dashboard quadrant because real
conversations (debugging a reward, comparing runs) need room.

**Benefit to the user:** The agent doesn't just answer questions — it operates
the app through tools (load robots, test rewards, start/stop training, compare
runs), explains what it's doing with live tool chips, and guides the user to
the next step.

**What the user can do:**
- Chat with the helper agent with streamed replies; watch tool calls execute
  inline (spinner → ✓/✗).
- Copy any message or the whole conversation; clear the chat.
- Ask the agent to perform any workflow step ("load r2d2 and start a short PPO
  run") and get proactive next-step suggestions.

## 8. Settings

**Why this page exists:** Personalization and integration config must be
discoverable but out of the way of the working surfaces.

**Benefit to the user:** The app adapts to the user — appearance, performance
trade-offs, and which LLM powers the agent are all user-controlled and
persisted.

**What the user can do:**
- Theme: dark/light/system mode, 8 accent colors (persisted across restarts).
- Viewport: stream resolution scale (FPS vs clarity trade-off).
- Ollama agent: provider, base URL (copyable), model, bearer token
  (show/hide), timeout, temperature, top-p, max tokens; save or reset defaults.

## 9. Logs

**Why this page exists:** When something goes wrong, users (and the agent)
need the raw truth — the last status message and the full training state —
without attaching a debugger.

**Benefit to the user:** Fast self-service debugging and easy bug reporting:
everything is selectable and copyable as JSON.

**What the user can do:**
- Read/copy the latest runtime status message.
- Read/copy the full training status as pretty-printed JSON.

---

# Ideal User Flow

The app is designed around one loop: **load → inspect → configure → verify →
train → watch → evaluate → iterate.** The notification bell and the agent keep
the user oriented at every step.

```
                ┌──────────────────────────────────────────────────┐
                │                                                  ▼
Robot Setup ──► Obs/Action ──► Reward Builder ──► Training ──► Evaluation
   │                ▲                                  │            │
   ▼                │                                  ▼            │
Simulation      (agent can drive any step)      Notifications      Iterate:
(watch live)        Agents tab                  (progress/done)  reward + params
```

**1. First launch.** The user starts on the Simulation dashboard. The viewport
shows the stream (or a clear "backend offline, retrying" state). The agent
greets them in the chat quadrant; asking "what can you do" yields a guided
overview.

**2. Load a robot — Robot Setup.** Browse to a URDF (or start with the bundled
`r2d2.urdf`), toggle fixed base/plane, Load. A notification confirms the load,
lists warnings, and suggests the next step. The user reviews the joint table
and warnings.

**3. Understand the spaces — Observation / Action.** Confirm the observation
vector and actuated joints look right. Run "zero action" and "safe random" to
see the robot respond in the viewport — proof the control path works.

**4. Shape the reward — Reward Builder.** Review components and run "Test
reward" a few times while posing the robot. Per-term values reveal which
component dominates. (The reward agent can review the design on request.)

**5. Pre-flight + launch — Training.** Save the env config, then Start PPO.
A "Training started" notification arrives with watch-list advice.

**6. Watch — Simulation + notifications.** The user watches the robot learn in
the viewport; milestone notifications (25/50/75%) report progress. The user is
free to explore other tabs — the bell catches anything important, including
failure with a diagnosis hint.

**7. Completion → Evaluation.** "Training complete" arrives with the model
path and next steps. The user (or the agent, via `list_runs` /
`get_run_details`) checks episode rewards and compares against earlier runs.

**8. Iterate.** Back to Reward Builder or Training with informed tweaks —
often by simply telling the agent: "lower the action penalty and run it again
for 50k steps." The loop repeats until the policy is good enough; everything
copyable along the way (joint names, formulas, JSON, run stats) feeds bug
reports, notebooks and team chat.

**Escape hatches at every step:** Logs for raw state, Settings to repoint the
agent at a different model, and the agent itself — which can execute any step
of this flow on the user's behalf.

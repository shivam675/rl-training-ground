# EasyRTG Final Presentation Design

## Communication job

By the end, the final-submission panel should understand that EasyRTG closes a real usability gap in robot reinforcement learning, believe that the project achieved its stated goal on two robot categories, and see credible academic and commercial value in the result.

## Audience and constraints

- Audience: an elite academic panel with mixed technical and business interests.
- Duration: 12–14 minutes.
- Format: 16:9 PowerPoint, approximately 14 slides.
- Tone: clear, personal, evidence-led, and confident without sounding promotional.
- Language: plain English; technical terms appear only when they explain a decision or result.
- No speaker notes.
- No invented performance, market, or training claims.

## Narrative approach

Use a proof-led transformation story:

1. Establish the real-world problem and why current workflows exclude non-specialists.
2. Define the gap EasyRTG fills and the user experience it provides.
3. Show the original technical decision and the limits revealed by the CPU-first version.
4. Treat Linux/Brax as a serious experiment that did not produce a usable policy.
5. Present the laptop failure and Dubizzle purchase as the human turning point that enabled the Isaac Sim pivot.
6. Explain how the final application and agent support the end-to-end workflow.
7. Prove the outcome with recorded throughput, policy metrics, and video frames.
8. Close on the academic and commercial meaning of the result, then explicitly resolve the original project goal.

## Slide sequence

1. **EasyRTG** — minimal title, student and supervisor details.
2. **Training one robot still means assembling several specialist systems** — simulator, environment, reward, training, monitoring, evaluation.
3. **Powerful tools exist, but the end-to-end experience is still built for experts** — define the gap without dismissing Isaac Lab, Brax, PyBullet, or Stable-Baselines3.
4. **EasyRTG turns a robot file and a plain-language goal into a guided training workflow** — load, configure, train, evaluate, compare.
5. **We started with the most accessible stack** — Flutter, FastAPI, PyBullet, Stable-Baselines3, Windows, and an assistant.
6. **The first version worked, but CPU speed limited iteration** — 240 FPS normally and approximately 1,750 FPS with parallel environments.
7. **Linux and Brax gave us GPU access, but not a policy we could use** — speed alone was not the outcome.
8. **An unexpected hardware failure changed the project path** — laptop failure, no meeting delay, Dubizzle GPU laptop, return to Windows, move to NVIDIA Isaac Sim.
9. **The final system keeps the simple workflow and replaces the training engine** — thousands of Isaac environments running on the GPU.
10. **The agent helps across the workflow, not only in a chat box** — interprets goals, inspects the robot, proposes rewards, starts or monitors training, and supports evaluation while keeping the user in control.
11. **GPU simulation changed the iteration speed by an order of magnitude** — verified quadruped results around 25k–30k FPS and a verified Valkyrie peak of 14.6k FPS. Label robot and environment counts so the comparison is not presented as a controlled benchmark.
12. **Unitree Go1 learned a measurable galloping policy** — use the 40M-step, 6,400-environment run and its recorded video. Show 30.1k average FPS, 1.38 m/s final forward speed, 0.215 flight-phase measure, and 1,000-step mean episode length.
13. **Valkyrie extended the same workflow to a humanoid** — use the 200M-step, 5,800-environment run and the supplied video. For the strongest saved-policy evidence, show the best-reward telemetry point at 31.0M steps: reward 276.1, 204-step mean episode length, and 0.72 m/s forward velocity. Show the separately measured 14.6k peak throughput without implying that it occurred at the same checkpoint.
14. **We achieved the goal—and made robot learning easier to approach** — two robot categories, end-to-end application, trained policies, academic reuse, and faster commercial prototyping. End with the resolved claim, not a generic thank-you slide.

## Visual system

- Light theme with a warm off-white background (`#F7F5F0`).
- Deep navy text (`#17212B`), teal evidence accent (`#1F8873`), and restrained warm orange (`#D97745`) for setbacks or pivots.
- Use a modern humanist sans-serif available in the workspace; avoid decorative or futuristic display fonts.
- Minimum sizes: 50 pt deck title, 35 pt slide titles, 24 pt subheads, 16 pt body.
- Prefer one strong composition per slide. Avoid dashboard grids, repeated cards, glossy gradients, fake UI controls, and excessive iconography.
- Vary silhouettes: full-width statement, text-plus-image, timeline, one simple workflow, evidence chart, and video-frame-led result slides.
- Use real EasyRTG application imagery and extracted frames from recorded evaluation videos. Do not generate synthetic robot imagery.
- Keep copy natural and slightly conversational. Avoid repeated slogans, inflated claims, and formulaic AI phrasing.

## Evidence policy

- Primary sources are the repository, original proposal, saved run configs, telemetry logs, evaluation logs, and recorded videos.
- Show small visible evidence labels on relevant slides because the final deck must contain no speaker notes.
- Use the verified Valkyrie maximum of 14,556 FPS, rounded to 14.6k.
- Distinguish average from peak throughput and include environment counts.
- Do not claim the different robot runs form a controlled hardware benchmark.
- Do not present training telemetry as real-world physical-robot performance.

## Quality checks

- Render and inspect every slide at full size.
- Check all titles for single-line fit and all body copy for readable wrapping.
- Fix every unintended overlap, clipping, and inconsistent alignment.
- Confirm every metric against its saved run before delivery.
- Confirm the delivered file has no speaker notes.

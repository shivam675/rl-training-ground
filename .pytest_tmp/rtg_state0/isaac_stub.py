import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
p.add_argument('--total-timesteps', type=int, default=100)
p.add_argument('--num-envs', type=int, default=1)
p.add_argument('--record', action='store_true')
a, _ = p.parse_known_args()
rd = Path(a.run_dir)
with (rd / 'telemetry.jsonl').open('a', encoding='utf-8') as h:
    h.write(json.dumps({'timestep': a.total_timesteps, 'reward_mean': 1.0,
                        'fps': 1000.0, 'episode_length_mean': 100}) + '\n')
stop = rd / 'STOP'
deadline = time.monotonic() + 30.0
while not stop.exists() and time.monotonic() < deadline:
    time.sleep(0.05)

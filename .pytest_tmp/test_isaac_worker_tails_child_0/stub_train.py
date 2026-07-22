import argparse, json, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
p.add_argument('--total-timesteps', type=int)
p.add_argument('--num-envs', type=int)
a, _ = p.parse_known_args()
t = Path(a.run_dir) / 'telemetry.jsonl'
for i in (1, 2, 3):
    with t.open('a', encoding='utf-8') as h:
        h.write(json.dumps({'timestep': i * 100, 'reward_mean': float(i),
                            'fps': 1900.0, 'episode_length_mean': 300}) + '\n')
    time.sleep(0.05)

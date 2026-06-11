# V1 Usage

1. Start the backend with `scripts/start_backend.sh`.
2. Start Flutter with `scripts/start_frontend.sh`.
3. Load a URDF from disk or a PyBullet sample path.
4. Inspect joints, limits, links, actuated joints, fixed joints, and warnings.
5. Review observation and action vector sizes.
6. Use zero or safe-random action tests before training.
7. Test the default reward components and inspect warnings.
8. Save the current environment config.
9. Start a small PPO run first, then scale timesteps after the robot behaves.
10. Use the agent panel for reward, robot, and training guidance.

The viewport uses left-drag orbit, right or middle drag pan, and wheel zoom.


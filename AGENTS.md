# VLA repository rules

## Scope and Git

- Maintain this independent repository on `ros-dev`. Inspect the current branch and working tree before editing; preserve unrelated changes and existing tasks.
- Keep commits local. Do not add/change remotes, fetch, push, publish, or merge into `main` unless explicitly requested. Existing remote configuration is not authorization to use it.
- Read-only retrieval of official Python dependencies and exact upstream revisions declared in `ros_ws/dependencies.repos` is allowed. Do not modify migration source repositories, third-party checkouts, or existing environments.
- Serialize Git mutations (`add`, `commit`, checkout, reset, etc.); never run them concurrently against the same Git directory. For status polling use `git --no-optional-locks status --short` to avoid unnecessary index writes. Do not terminate another process to free a lock.
- Keep datasets, weights, tokenizer assets, replay databases, videos, logs, build outputs and third-party checkouts out of Git. Inspect staged paths before committing; do not apply a blanket `*.lock` ignore rule because dependency lockfiles are source-controlled inputs.

## Runtime and hardware boundaries

- Never start or move hardware during development/tests. Do not launch real robot, gripper, GELLO or camera processes, connect devices, or operate existing sessions. Use mocks or `--check-only`.
- Model Python and ROS Python must remain separate. Use `scripts/model_env.sh` / `scripts/test.sh` for the existing model interpreter; use `scripts/ros_env.sh` / `scripts/test_ros.sh` for ROS Python 3.10. Never source both environments into one process or install model dependencies into ROS Python.
- Do not install, upgrade or repair existing environments as a side effect of a task. Missing assets/dependencies are reported as unverified, not counted as successful tests.
- Create test datasets, checkpoints, replay, logs and build artifacts under unique temporary directories. Never reuse a live dataset or training run for verification.

## Architecture and contracts

- ROS owns the only execution queue and the 30 Hz action / 100 Hz tracking timeline; model services return candidate chunks. Web and terminal routes share the ROS collector/coordinator.
- Preserve manual recovery after queue exhaustion, storage blockage or faults. Late inference results, reconnects and successful storage recovery must not re-enable motion.
- Keep model inference and control messages independent. Generation changes revoke old results; committed prefixes and installed Actor suffixes must remain protected.
- Accept only current contracts: wire/model/statistics v2, recording schema 5, training semantics/online run v3, Replay v4. Do not add old asset/protocol conversion or fallback paths. Official base weights may initialize new training.
- Preserve execution provenance, actual timestamps, anchors and validity masks end to end. Control callbacks must not encode video or perform disk writes/fsync; storage failures preserve uncommitted records for retry.
- See `docs/EXECUTION_RTC_RLT.md` for protocol, state transitions, loss/reward semantics and acceptance limits; `docs/ROS_ACTION_CONTRACT.md` and `docs/DATA_COLLECTION_FORMAT.md` define action/data requirements.

## Lock handling

- Distinguish dependency lockfiles, Git transaction locks and persistent runtime `flock` files. A runtime file's presence or recorded PID is not evidence of an active lock; inspect the kernel lock.
- Do not unlink runtime lock files on release or crash recovery: waiters may already hold the old inode, and deleting it can allow two owners. Release the descriptor; clear diagnostic owner metadata while still holding an exclusive lease on normal exit.
- Diagnose with `python3 scripts/check_locks.py`; add `--runtime-lock PATH` for an external dataset/lease file. Default operation is read-only apart from a momentary nonblocking lock probe.
- For a confirmed abandoned Git index lock, use `python3 scripts/check_locks.py --clear-stale-index`. It refuses recent/changing locks, non-owned files, observable Git processes, open lock descriptors and incomplete process inspection. Stop initiating concurrent Git commands while recovering. Never use recursive `.lock` deletion, age alone, or PID alone as a cleanup policy.

## Verification and delivery

- Run meaningful affected tests first. Model example: `VLA_PYTHON=/path/to/existing/model/python bash scripts/test.sh tests/control -q`. ROS example: `bash scripts/test_ros.sh tests/ros -q`. Do not install dependencies merely to make a check pass.
- For packaging changes, use the model interpreter to run `scripts/validate_wheel.py`: it builds in a clean temporary source tree and checks external imports without installation. Direct builds can retain deleted modules in `build/`.
- For timing/control changes, `scripts/validate_ros_control.sh --duration 600` is the approved fixed-state mock harness, using an isolated DDS domain. Do not replace its mock plugin with hardware. Do not rerun costly checks without a relevant change or unresolved concern.
- Update relevant docs when behavior/contracts change. Report the changes, tests and skipped/unverified cases. Offline/mock results do not establish physical continuity, real GPU latency, camera throughput or task success.

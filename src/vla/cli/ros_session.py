"""Terminal front end for the same ROS session used by the Web runner."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def terminal_command(backend, confirmation, event):
    """Nonblocking confirmations stay in the terminal input adapter."""
    status = backend.sample()
    active = status.get('recording', False)
    episode = status.get('episode')
    event = confirmation.route(event, destructive=active and event in ('discard', 'quit'),
        valid=lambda: backend.sample().get('recording', False) and backend.sample().get('episode') == episode)
    if event is None:
        return False
    if event in ('quit', 'force_quit'):
        if active:
            backend.command('discard', {})
        backend.command('stop', {})
        return True
    backend.command(event, {})
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--arms', choices=['single_left', 'dual'], default='single_left')
    parser.add_argument('--source', choices=['gello', 'gamepad'], default='gamepad')
    parser.add_argument('--action-space', choices=['joint', 'tcp'], default='tcp')
    parser.add_argument('--workflow', choices=['collect', 'deploy'], default='collect')
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--task')
    parser.add_argument('--tasks', nargs='+', help='Task prompts selectable between episodes')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--side-camera-serial', default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--online', action='store_true', help='Require an existing online RLT server contract')
    parser.add_argument('--hardware-config', type=Path)
    parser.add_argument('--policy-path', default='')
    parser.add_argument('--policy-kind', choices=['sft', 'rlt'], default='sft')
    parser.add_argument('--tokenizer-path', default='')
    parser.add_argument('--model-python', default=os.environ.get('VLA_PYTHON', sys.executable))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    from vla.web.config import SessionConfig
    from vla.paths import workspace
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    root = workspace()
    config = SessionConfig(workflow=args.workflow, arms=args.arms, source=args.source,
        action_space=args.action_space, config_path=str(args.config or ''),
        dataset_root=str(args.dataset_root or ''), task=args.task or "", resume=args.resume, side_camera_serial=args.side_camera_serial,
        policy_path=args.policy_path, policy_kind=args.policy_kind, tokenizer_path=args.tokenizer_path,
        model_python=args.model_python, ros_workspace=str(root/'ros_ws'), ros_config=str(args.hardware_config or ''))
    rollout = load_rollout_config(config.default_config())
    arms = 'dual' if len(rollout.arms) == 2 else 'single_left'
    tasks = tuple(dict.fromkeys(args.tasks or rollout.tasks))
    task = args.task or (tasks[0] if args.tasks else rollout.task)
    tasks = tuple(dict.fromkeys((task, *tasks)))
    config = replace(config, arms=arms, task=task, tasks=tasks)
    if args.resume and args.dataset_root is None:
        parser.error('--resume requires --dataset-root')
    if args.dataset_root is not None and not args.online:
        from vla.datacollection.franka_robotiq_dual.gamepad_record import validate_dataset_root
        validate_dataset_root(args.dataset_root, resume=args.resume)
    if not args.execute:
        print(f'ROS2 configuration valid: {arms}, {args.source}, {args.action_space}, '
              f'{50*len(rollout.arms)}D measured state, absolute storage; hardware not started')
        return
    if not sys.stdin.isatty():
        parser.error('--execute requires an interactive terminal')
    if sys.version_info[:2] != (3, 10):
        from vla.web.processes import interpreter_command, clean_environment
        command = interpreter_command('/usr/bin/python3', ['-m', 'vla.cli.ros_session',
            *(sys.argv[1:] if argv is None else argv), '--model-python', args.model_python],
            ros=True, root=root, workspace=root/'ros_ws')
        raise SystemExit(subprocess.call(command, env=clean_environment(root, '/usr/bin/python3', ros=True)))
    directory = root/'runtime/terminal'/f'ros-{time.time_ns()}'
    directory.mkdir(parents=True)
    if not config.ros_config:
        from vla.web.hardware import preset
        hardware = preset(arms)
        hardware.update({k: os.environ[k] for k in hardware if k in os.environ})
        path = directory/'hardware.json'
        path.write_text(json.dumps(hardware, indent=2))
        config = replace(config, ros_config=str(path))
    config = replace(config, dataset_root=str(config.dataset_path(root)))
    from vla.web.backends import make_backend
    from vla.datacollection.franka_robotiq_single_left.auto_record import KeyboardEvents
    from vla.terminal import console_status
    backend = make_backend(config, directory, root)
    from vla.terminal import TerminalConfirmation
    confirmation = TerminalConfirmation()
    if args.online and not backend.collector.require_online:
        backend.close()
        raise ValueError('--online requires an online-enabled policy server')
    keyboard = KeyboardEvents(keymap={b'e':'enable', b'p':'pause', b'r':'record', b's':'save',
        b'd':'discard', b'c':'recover', b' ':'intervention', b'q':'quit', b'k':'critical', b'\t':'next_subtask', b't':'next_task'}, confirmation=confirmation)
    print('e=enable p=pause r=record s=save d=discard c=recover Space=takeover k=critical Tab=stage t=task q=quit')
    from types import SimpleNamespace
    from vla.datacollection.start_teleop.gamepad_preview import CameraPreview, configure_roles
    configure_roles(config.cameras)
    def snapshot():
        frames=backend.images()
        if set(frames)!=set(config.cameras):
            raise RuntimeError('waiting for all camera frames')
        return SimpleNamespace(images={k:v[1] for k,v in frames.items()}, timestamps={k:v[0] for k,v in frames.items()})
    preview=CameraPreview(directory,snapshot,enabled=not args.no_preview).start()
    next_status = 0.
    try:
        while True:
            backend.tick()
            confirmation.expire()
            for event in confirmation.prioritize(keyboard.consume()):
                try:
                    if terminal_command(backend, confirmation, event):
                        return
                except (RuntimeError, ValueError) as error:
                    print(f'Command rejected: {error}', flush=True)
            if time.monotonic() > next_status:
                status = backend.sample()
                preview.update_status({**status, 'mode':status['state'], 'critical':'critical' if status.get('critical') else 'normal'})
                console_status(f"{status['state']} episode={status['episode']} frames={status['frames']} "
                               f"controller={status['controller_state']}")
                next_status = time.monotonic()+1
            time.sleep(.005)
    finally:
        preview.close()
        keyboard.close()
        backend.close()


if __name__ == '__main__':
    main()

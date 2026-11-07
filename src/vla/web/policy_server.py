"""Managed, inference-only model service; loaded before any hardware starts."""
from __future__ import annotations
import argparse
import socket
import signal
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=('sft','rlt'),required=True)
    p.add_argument('--policy-path',required=True)
    p.add_argument('--tokenizer-path',required=True)
    p.add_argument('--handover-path',default='')
    p.add_argument('--socket-path',required=True,type=Path)
    p.add_argument('--task',required=True)
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--arms',choices=('single_left','dual'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--rtc-deployment-config',type=Path)
    args=p.parse_args()
    if args.kind == 'rlt':
        from vla.cli.serve_franka_rollout import parse_args, FrankaRolloutPolicyServer
        # Use original CLI defaults, including all RTC parameters. No online flags.
        sys.argv=[sys.argv[0],'--policy-path',args.policy_path,'--socket-path',str(args.socket_path),
                  '--task',args.task,'--device',args.device,'--tokenizer-path',args.tokenizer_path,
                  '--robot-type','franka_robotiq_dual' if args.arms=='dual' else 'franka_robotiq']
        if args.tasks:
            sys.argv += ['--tasks', *args.tasks]
        if args.rtc_deployment_config:
            sys.argv += ['--rtc-deployment-config',str(args.rtc_deployment_config)]
        defaults=parse_args()
        defaults.handover_path=args.handover_path
        server=FrankaRolloutPolicyServer(defaults)
    else:
        from vla.cli.serve_franka_sft import Server
        args.checkpoint=Path(args.policy_path)
        server=Server(args)
    serve(server,args.socket_path)


def serve(server, path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import BoundedSemaphore
    from vla_rollout_protocol import recv_packet, send_packet
    from vla.cli.serve_franka_rollout import _TrainingProgressSender
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = False
    slots = BoundedSemaphore(8)

    def interrupt(*_):
        nonlocal stop
        stop = True

    def handle_connection(connection):
        header = {}
        try:
            with connection:
                connection.settimeout(30)
                try:
                    header, arrays = recv_packet(connection)
                    if header.get('kind') in {'episode_saved', 'recover_online', 'begin_episode'}:
                        if not server.plan_gate.worker.acquire(blocking=False):
                            raise RuntimeError('model worker is busy')
                        try:
                            callback = _TrainingProgressSender(connection, header.get('request_id', -1))
                            response, values = server.handle(header, arrays, progress_callback=callback)
                        finally:
                            server.plan_gate.worker.release()
                    else:
                        response, values = server.handle(header, arrays)
                    send_packet(connection, response, values)
                except Exception as error:
                    try:
                        send_packet(connection, {'kind': 'error', 'request_id': header.get('request_id', -1),
                                                 'error_type': type(error).__name__, 'error': str(error)})
                    except OSError:
                        pass
        finally:
            slots.release()

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix='plan-service') as workers:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            path.chmod(0o600)
            listener.listen(8)
            listener.settimeout(.5)
            try:
                while not stop:
                    try:
                        connection, _ = listener.accept()
                    except socket.timeout:
                        continue
                    if slots.acquire(blocking=False):
                        workers.submit(handle_connection, connection)
                    else:
                        connection.close()
            finally:
                path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

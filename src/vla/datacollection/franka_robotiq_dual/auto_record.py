"""Dual-arm GELLO recording through the shared schema-5 ROS session."""
import sys


def main() -> None:
    from vla.cli.ros_session import main as shared_main
    shared_main(['--arms', 'dual', '--source', 'gello', *sys.argv[1:]])


if __name__ == '__main__':
    main()

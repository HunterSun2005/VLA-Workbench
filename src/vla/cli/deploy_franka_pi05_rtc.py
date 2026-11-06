"""Launch a ROS policy session using the checkpoint action contract."""
import sys
from vla.cli.ros_session import main as session_main


def main():
    session_main(['--workflow', 'deploy', *sys.argv[1:]])


if __name__ == '__main__':
    main()

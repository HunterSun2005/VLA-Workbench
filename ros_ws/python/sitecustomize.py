"""Only loaded by VLA ROS launchers; exclude unrelated editable workspaces."""
from pathlib import Path
import os
import sys

_root = Path(__file__).resolve().parents[2]
_allowed = (_root, Path("/usr/lib"), Path("/usr/local/lib"), Path("/usr/bin"), Path("/opt/ros/humble"), Path("/opt/openrobots"))
sys.path[:] = [value for value in sys.path if any(Path(value or os.getcwd()).resolve().is_relative_to(root) for root in _allowed)]
# Global site-packages may register editable finders independently of sys.path.
sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, "__module__", "").startswith("__editable__")]

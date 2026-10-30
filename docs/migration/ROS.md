# ROS source decisions

Custom packages are snapshots of /home/amax/workspace/ros2/src at workspace HEAD fa0407bb15be5c220f0696823dbc3803a657c9c7 plus local changes and required untracked files. See ros-files.json for individual hashes. franka_gello_sensor_reader and my_vla_package are not dependencies of retained routes and are excluded.

Upstream Franka revisions are pinned in ros_ws/dependencies.repos. The old RViz preferences are archived but not applied. The libfranka-common local change raised minimum CMake to 4.1.3; it is archived but not applied because the clean Humble build uses system CMake 3.22 and upstream supports it. Neither changes robot control behavior. serial sources had been deleted by the old installer; the upstream ros2 branch is now frozen to d8d160678aa0b31cdf467c052b954fa287cc6cdf, so binary equivalence to the old unpinned serial build is unverified.

Robotiq behavior patches are applied separately to the pinned upstream. robotiq_description and robotiq_controllers use installed Humble packages. System /opt/openrobots is an explicit native dependency (pinocchio); no old workspace overlay is sourced.

## Python isolation findings

The machine's Dynamixel SDK metadata pointed to an editable factr_ws checkout. Official SDK 3.7.51 (commit ad2d6136831dc3400aa43db21323c9cbfc182a34) has byte-identical contents for all 10 Python SDK source files; VLA installs that source into its own ignored python_deps directory. Its revision is in the dependency manifest, and comparison hashes are in dynamixel-source.json.

System NumPy 2.2.6 conflicted with the installed OpenCV binary. The ROS overlay now pins NumPy 1.26.4 with compatible OpenCV/scientific/camera wheels, independently from model NumPy 2.2.6. The VLA-only sitecustomize excludes unrelated editable Python paths. No global packages were uninstalled or altered. Runtime import locations are recorded in system-python.json.

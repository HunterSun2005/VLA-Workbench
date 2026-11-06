# VLA repository rules

Maintain this independent repository on ros-dev. No remotes or publishing until explicitly requested. Read-only retrieval of official Python dependencies and exact upstream revisions declared in ros_ws/dependencies.repos is allowed. Do not modify migration source repositories or existing environments. Never start or move hardware during tests. Keep datasets, weights, tokenizer assets, replay databases, logs and third-party checkouts out of Git. Model Python and ROS Python must remain separate.

# LeRobot 0.6.1 adaptation

Official PyPI `lerobot[pi,training]==0.6.1`; no vendored code or fork. Baseline commit 13d755c retains 0.5.1.

- Public config registration and convention-based factories: RLTokenConfig/RLTokenPolicy, ChunkACConfig/ChunkACPolicy, RLTPretrainedConfig/RLTPretrainedPolicy. Upstream policy factories are not monkeypatched.
- Attention masks use the shared public utility. SFT observation preparation uses lerobot.policies.utils; obsolete control_utils imports are removed.
- AC training guards the trainer's make_train_eval_datasets boundary and returns pre-split transition caches. Its metadata implements episode boundaries, index mapping, camera keys and language-column detection. Resume processors retain cached-batch bypass without changing saved SFT steps.
- Training uses env_eval_freq; VLA SFT and cache paths use eval_split=0. Upstream handles uint8 image conversion before preprocessing. Existing official-loader/video fixtures exercise the data path.
- meta/vla_metadata.json preserves custom metadata when typed DatasetInfo drops unknown fields. Conflicting sidecar/raw metadata fails explicitly. The original evo_rlt dataset namespace remains unchanged.
- Actual upstream update_policy is tested for optimizer updates and metric counts. Spawn worker tests cover public registration and sampler resume. No site-packages patches.

Torch 2.10.0, TorchVision 0.25.0 and TorchCodec 0.10.0 are pinned as a compatible group. Full installed versions are in requirements.lock.

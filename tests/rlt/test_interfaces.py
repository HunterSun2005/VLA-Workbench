from __future__ import annotations

from vla.core.config import RLTConfig


def test_config_from_yaml(tmp_path):
    yaml_content = """
seed: 42
action_dim: 7
rl_token:
  token_dim: 512
  nhead: 4
  enc_layers: 2
  dec_layers: 2
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_content)
    cfg = RLTConfig.from_yaml(path)

    assert cfg.seed == 42
    assert cfg.action_dim == 7
    assert cfg.rl_token.token_dim == 512
    assert cfg.rl_token.nhead == 4
    # ff_dim should auto-compute
    assert cfg.rl_token.ff_dim == 4 * 512


def test_config_default_yaml():
    cfg = RLTConfig.default()
    assert cfg.action_dim == 14
    assert cfg.training.utd_ratio == 5

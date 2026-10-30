from __future__ import annotations


def test_train_actor_critic_delegates_to_registered_lerobot(monkeypatch) -> None:
    import vla.adapters.lerobot as adapter
    import lerobot.scripts.lerobot_train as lerobot_train
    from vla.cli import train_actor_critic_native

    calls = []
    monkeypatch.setattr(adapter, "register", lambda: calls.append("register"))
    monkeypatch.setattr(lerobot_train, "main", lambda: calls.append("train"))

    for name in ("update_policy", "make_train_eval_datasets", "make_pre_post_processors"):
        monkeypatch.setattr(lerobot_train, name, getattr(lerobot_train, name))
    train_actor_critic_native.main()

    assert calls == ["register", "train"]

"""Released-channel state transitions use actual samples, not command velocities."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.datacollection.franka_robotiq_rollout.gamepad import GamepadRelease


def sample(release, t, *, active=(), released=(), p=None, r=None, now=None):
    a, b = np.zeros(6, bool), np.zeros(6, bool)
    a[list(active)], b[list(released)] = True, True
    return np.r_[release.update(a, b, np.zeros(3) if p is None else np.asarray(p),
                                np.eye(3) if r is None else r, np.eye(3), t,
                                t if now is None else now)]


def test_partial_release_follows_until_stopped_then_latches_and_reengages():
    release = GamepadRelease()
    sample(release, 0., active=(0, 1))
    assert sample(release, .03, active=(1,), released=(0,)).tolist() == [True, False, False, False, False, False]
    # Genuine continued motion resets the stability counter.
    for t in np.arange(.06, .42, .03):
        assert sample(release, t, active=(1,), p=[t*.01, 0, 0])[0]
        assert release.braking[0]
    for t in np.arange(.42, .57, .03):
        sample(release, t, active=(1,), p=[.0039, 0, 0])
    assert not release.braking.any()
    assert not sample(release, .6, active=(1,), p=[.1, 0, 0]).any()
    sample(release, .63, released=(0,))
    assert release.braking[0]
    sample(release, .66, active=(0,))
    assert not release.braking.any()
    release.check_deadline(10.)


def test_minimum_duration_duplicate_samples_and_large_gaps_do_not_fake_stop():
    release = GamepadRelease()
    sample(release, 0., active=(0,))
    sample(release, .03, released=(0,))
    for i in range(4):
        sample(release, .06+i*.03)
    assert release.braking[0]  # Stable, but less than 0.2 seconds since release.
    sample(release, .15, now=.4)  # Cached sample: never stop on host clock alone.
    assert release.braking[0] and release.stable[0] == 0
    sample(release, .5)  # A sample gap also provides no stability evidence.
    assert release.braking[0] and release.stable[0] == 0
    for t in (.53, .56, .59, .62):
        sample(release, t)
    assert not release.braking.any()


def test_rotation_velocity_uses_so3_and_released_channel_only():
    release = GamepadRelease()
    sample(release, 0., active=(3, 4))
    sample(release, .03, active=(4,), released=(3,))
    for t in np.arange(.06, .36, .03):
        r = Rotation.from_rotvec([t*.1, 0, 0]).as_matrix()
        sample(release, t, active=(4,), r=r)
        assert release.braking[3] and not release.braking[4]


def test_timeout_clear_and_bad_sample_clock():
    release = GamepadRelease()
    sample(release, 0., released=(0, 5))
    with pytest.raises(RuntimeError, match='settle'):
        release.check_deadline(2.)
    release.clear()
    release.check_deadline(5.)
    assert release.sample is None
    sample(release, 1.)
    with pytest.raises(RuntimeError, match='regressing'):
        sample(release, .9)
    for invalid in (None, True, float('nan'), float('inf'), -1.):
        with pytest.raises(RuntimeError, match='sample time'):
            sample(release, invalid, now=1.)

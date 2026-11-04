"""GELLO's existing collision profile for explicit opt-in on Franky routes.

Values match start_teleop/lib/controllers.sh::apply_collision_behavior.
Tuple order is the full eight-argument Franky/libfranka overload: lower/upper
joint torque (Nm), acceleration then nominal, followed by lower/upper Cartesian
wrench (N for XYZ force, Nm for XYZ torque), acceleration then nominal.
These are contact/reflex thresholds, not payload compensation or rated payload.
"""

GELLO_COLLISION_THRESHOLDS = (
    (50., 50., 50., 40., 40., 40., 40.),
    (100., 100., 100., 80., 80., 40., 40.),
    (50., 50., 50., 30., 40., 40., 40.),
    (100., 100., 100., 80., 80., 40., 40.),
    (50., 50., 50., 30., 30., 30.),
    (100., 100., 100., 30., 30., 30.),
    (50., 50., 50., 30., 30., 30.),
    (100., 100., 100., 30., 30., 30.),
)

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from .configuration_vla_pi05 import VLAPI05Config


class VLAPI05Policy(PI05Policy):
    config_class = VLAPI05Config
    name = 'vla_pi05'

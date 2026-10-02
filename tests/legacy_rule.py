"""Config copy with the DEV-007 clauses switched off: the legacy first-crossing state rule, no parameter clause."""
import copy


def legacy(cfg):
    c = copy.deepcopy(cfg)
    c["ukf"]["divergence"]["state_dwell_s"] = 0
    c["ukf"]["divergence"]["parameter_sd_multiple"] = None
    return c


def new_rule(cfg):
    """The dormant DEV-007 settings as tested in IMP-091 (not adopted): 0.5 s consecutive, 8 prior SD."""
    c = copy.deepcopy(cfg)
    c["ukf"]["divergence"]["state_dwell_s"] = 0.5
    c["ukf"]["divergence"]["parameter_sd_multiple"] = 8
    c["ukf"]["divergence"]["parameter_dwell_s"] = 0.5
    return c

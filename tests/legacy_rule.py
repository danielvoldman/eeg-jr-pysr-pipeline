"""Config copy with the DEV-007 clauses switched off: the legacy first-crossing state rule, no parameter clause."""
import copy


def legacy(cfg):
    c = copy.deepcopy(cfg)
    c["ukf"]["divergence"]["state_dwell_s"] = 0
    c["ukf"]["divergence"]["parameter_sd_multiple"] = None
    return c

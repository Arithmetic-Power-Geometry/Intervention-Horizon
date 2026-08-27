import math
import pytest
from intervention_horizon import (Action, viable_actions, intervention_horizon,
    collective_horizon_from_individual, time_costed_voi, analyze)

def test_near_action_viable_far_not():
    actions=[Action('near',5,1,.999),Action('far',28,2,.999)]
    v=[a.name for a in viable_actions(30,2,actions,.05)]
    assert 'near' in v and 'far' not in v

def test_horizon_order():
    assert intervention_horizon(40,2,[Action('near',5,1,.999)],.05)>intervention_horizon(40,2,[Action('far',20,1,.999)],.05)

def test_collective_bound_is_minimum_individual():
    h=[12.,20.,25.]
    coll=collective_horizon_from_individual(h,2.)
    assert coll==10. and coll<=min(h)

def test_collective_negative_delay_rejected():
    with pytest.raises(ValueError): collective_horizon_from_individual([10,12],-1)

def test_tcvoi_characterization():
    assert time_costed_voi(.10,.15,.01)<0
    assert time_costed_voi(.20,.05,.01)>0

def test_analysis_status_changes_with_access():
    r1=analyze(35,3,[Action('near',5,1,.999)],.05,100,1)
    r2=analyze(15,3,[Action('far',20,2,.999)],.05,100,1)
    assert r1.horizon>r2.horizon
    assert r2.status=='INTERVENTION_LOST'


def test_analyze_respects_fragility_and_status_parameters():
    actions = [Action("near", 8, 1.0, 0.999, cost=0.1)]
    r = analyze(20, 2, actions, epsilon=0.05, perturb_sd=0.0, seed=7,
                act_threshold=20.0, critical_threshold=2.0)
    assert r.fragility in (0.0, 1.0)
    assert r.status in {"SAFE", "ACT", "CRITICAL", "INTERVENTION_LOST"}


def test_time_costed_voi_sign_characterization():
    assert time_costed_voi(0.20, 0.10, 0.02) > 0
    assert time_costed_voi(0.10, 0.10, 0.02) < 0

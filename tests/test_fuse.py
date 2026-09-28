from sie.models import Hit
from sie.index.fuse import reciprocal_rank_fusion


def test_rrf_prefers_consensus():
    a = [Hit("alpha", 0.9), Hit("beta", 0.8)]
    b = [Hit("alpha", 0.7), Hit("gamma", 0.6)]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].skill_slug == "alpha"   # appears high in both

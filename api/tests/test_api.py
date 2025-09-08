from __future__ import annotations

from api.api.schemas import AnalyzeRequest
from api.api.service import analyze


def test_analyze_stub_golden():
    req = AnalyzeRequest(
        context="EU, Digital Markets",
        options=["Option A", "Option B"],
        stakeholders=["Consumers", "SMEs"],
        riskToggles={"privacy": True, "competition": False},
    )
    resp = analyze(req)
    assert "Options considered: Option A, Option B" in resp.scope_summary
    assert len(resp.impact_matrix) == 2
    assert any("Data minimization" in m for m in resp.mitigation_checklist)
    # In CI, we expect deterministic stub (no model used)
    assert resp.used_model in (None, "gpt-4o-mini")


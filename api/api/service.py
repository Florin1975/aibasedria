from __future__ import annotations

import os
from typing import List

from .schemas import AnalyzeRequest, AnalyzeResponse, ImpactEntry


def _stub_scope_summary(req: AnalyzeRequest) -> str:
    options = ", ".join(req.options) if req.options else "(no options provided)"
    return (
        f"Jurisdiction/policy context provided. Options considered: {options}. "
        f"Stakeholders: {', '.join(req.stakeholders) if req.stakeholders else '(none)'}; "
        f"Risk toggles: {', '.join([k for k,v in req.riskToggles.items() if v]) or 'none'}."
    )


def _stub_impact_matrix(req: AnalyzeRequest) -> List[ImpactEntry]:
    if not req.options:
        opts = ["Option A", "Option B"]
    else:
        opts = req.options
    rows: List[ImpactEntry] = []
    for o in opts:
        rows.append(
            ImpactEntry(
                option=o,
                economic=f"Economic effects for {o}: costs vs benefits summarized.",
                social=f"Social effects for {o}: distributional and equity notes.",
                legal=f"Legal effects for {o}: compliance, authority, and risk.",
            )
        )
    return rows


def _stub_mitigations(req: AnalyzeRequest) -> List[str]:
    base = [
        "Pilot phase with limited scope",
        "Targeted exemptions for small entities",
        "Sunset review after 12 months",
        "Guidance + outreach for stakeholders",
    ]
    if req.riskToggles.get("privacy", False):
        base.append("Data minimization + DPIA checklist")
    if req.riskToggles.get("competition", False):
        base.append("Market monitoring + anti-trust safe harbor review")
    return base


def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # Local-only AI: if key available, you may opt-in to model calls.
    openai_key = os.getenv("OPENAI_API_KEY")
    used_model = None

    # For CI and default behavior, return deterministic stub.
    scope = _stub_scope_summary(req)
    matrix = _stub_impact_matrix(req)
    mitigations = _stub_mitigations(req)

    # Optional: if OPENAI_API_KEY present, attempt to enhance text blocks.
    if openai_key:
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=openai_key)
            prompt = (
                "Summarize the RIA scope in 3 concise sentences given: "
                f"context='{req.context}', options={req.options}, stakeholders={req.stakeholders}."
            )
            # Keep this lightweight; errors fall back to stub
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=160,
            )
            used_model = resp.model or "gpt-4o-mini"
            if resp.choices and resp.choices[0].message.content:
                scope = resp.choices[0].message.content.strip()
        except Exception:
            # Never fail; keep stubbed content
            used_model = None

    return AnalyzeResponse(
        scope_summary=scope,
        impact_matrix=matrix,
        mitigation_checklist=mitigations,
        used_model=used_model,
    )


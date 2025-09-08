from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    context: str = Field(..., description="Problem context: jurisdiction, policy area, scope")
    options: List[str] = Field(default_factory=list, description="Policy options A/B/C")
    stakeholders: List[str] = Field(default_factory=list, description="Stakeholder groups")
    riskToggles: Dict[str, bool] = Field(default_factory=dict, description="Risk flags")


class ImpactEntry(BaseModel):
    option: str
    economic: str
    social: str
    legal: str


class AnalyzeResponse(BaseModel):
    scope_summary: str
    impact_matrix: List[ImpactEntry]
    mitigation_checklist: List[str]
    used_model: Optional[str] = None


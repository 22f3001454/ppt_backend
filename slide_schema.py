from typing import List
from pydantic import BaseModel, Field

class SlideItem(BaseModel):
    title: str = Field(..., description="Slide title")
    bullets: List[str] = Field(default_factory=list, description="Bullet points")
    notes: str = Field(default="", description="Optional speaker notes")

class SlidePlan(BaseModel):
    slides: List[SlideItem] = Field(default_factory=list)

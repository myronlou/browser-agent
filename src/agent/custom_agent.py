from pydantic import BaseModel, Field
from typing import Optional

class CurrentState(BaseModel):
    task_progress: Optional[str] = "In progress"
    future_plans: Optional[str] = "Planning next steps"
    thought: Optional[str] = "Analyzing page"
    summary: Optional[str] = "Task ongoing"

class CustomAgentOutput(BaseModel):
    current_state: CurrentState
    action: str = Field(..., description="Next browser action")
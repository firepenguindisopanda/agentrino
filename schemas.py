from datetime import datetime

from pydantic import BaseModel, Field


class AgentBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)
    description: str | None = Field(None, max_length=300)
    system_prompt: str = Field(..., min_length=5)
    color: str | None = Field(None, max_length=20)
    icon: str | None = Field(None, max_length=40)


class AgentOut(AgentBase):
    id: str
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    agent_id: str = Field(..., min_length=5)
    session_id: str = Field(..., min_length=5)
    title: str | None = Field(None, max_length=120)


class ConversationGetOrCreate(BaseModel):
    agent_id: str = Field(..., min_length=5)
    session_id: str = Field(..., min_length=5)


class ConversationOut(BaseModel):
    id: str
    agent_id: str
    session_id: str
    title: str | None = None
    is_archived: bool = False
    last_activity_at: datetime
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict


class ChatStreamRequest(BaseModel):
    content: str = Field(..., min_length=1)


class DocumentAdd(BaseModel):
    content: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)


class DocumentAddResponse(BaseModel):
    ids: list[str]


class DocumentDelete(BaseModel):
    ids: list[str] = Field(..., min_length=1)


class DocumentDeleteResponse(BaseModel):
    deleted: bool


class AdminLogin(BaseModel):
    password: str


class OracleAnalysisOption(BaseModel):
    title: str = Field(description="The architectural choice or solution name")
    description: str = Field(description="A concise summary of the approach")
    pros: list[str] = Field(description="List of positive trade-offs (max 3)")
    cons: list[str] = Field(description="List of negative trade-offs or risks (max 3)")
    effort: str = Field(description="Estimated effort: Quick(<1h), Short(1-4h), Medium(1-2d), or Large(3d+)")
    recommended: bool = Field(default=False, description="Is this the single primary recommendation?")


class OracleAnalysisResponse(BaseModel):
    bottom_line: str = Field(description="2-3 sentences capturing the primary recommendation. No preamble.")
    options: list[OracleAnalysisOption] = Field(
        description="Exactly 4 comparative architectural options.",
        min_length=4,
        max_length=4
    )
    action_plan: list[str] = Field(description="An actionable, step-by-step implementation plan for the recommended choice.")
    watch_out_for: list[str] = Field(default_factory=list, description="Optional lists of risks, edge cases, or mitigations.")

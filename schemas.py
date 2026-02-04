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
    title: str | None = Field(None, max_length=120)


class ConversationOut(BaseModel):
    id: str
    agent_id: str
    title: str | None = None
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

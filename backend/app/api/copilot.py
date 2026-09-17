"""Quality Copilot API (9N).

POST /api/v1/copilot/query            {conversation_id, message}
GET  /api/v1/copilot/conversations/{id}
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..copilot.conversation import conversation_store
from ..copilot.service import CopilotService
from ..security.auth import require_any_authenticated

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


class QueryIn(BaseModel):
    conversation_id: str | None = None
    message: str


@router.post("/query", dependencies=[Depends(require_any_authenticated())])
async def copilot_query(
    body: QueryIn,
    session: AsyncSession = Depends(get_session),
) -> dict:
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=422, detail={"error": {"code": "empty_message", "message": "message is required"}})
    return await CopilotService().query(
        session, conversation_id=body.conversation_id, message=body.message.strip()
    )


@router.get("/conversations/{conversation_id}", dependencies=[Depends(require_any_authenticated())])
async def copilot_conversation(conversation_id: str) -> dict:
    conv = await conversation_store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail={"error": {"code": "not_found", "message": "conversation not found"}})
    return conv.to_dict()

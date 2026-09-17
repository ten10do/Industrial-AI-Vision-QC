from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database import get_session
from ..models import Inspection, Product
from ..schemas import InspectionOut, ProductOut
from ..security.auth import require_any_authenticated
from .serializers import to_inspection_out

router = APIRouter(prefix="/api/v1", tags=["products"])


@router.get("/products/{product_id}", response_model=ProductOut, dependencies=[Depends(require_any_authenticated())])
async def get_product(product_id: str, session: AsyncSession = Depends(get_session)) -> Product:
    result = await session.execute(select(Product).where(Product.product_id == product_id))
    product = result.scalar_one_or_none()
    if product is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "product not found"))
    return product


@router.get("/products/{product_id}/inspections", response_model=list[InspectionOut], dependencies=[Depends(require_any_authenticated())])
async def get_product_inspections(product_id: str, session: AsyncSession = Depends(get_session)) -> list[InspectionOut]:
    product = (
        await session.execute(select(Product).where(Product.product_id == product_id))
    ).scalar_one_or_none()
    if product is None:
        raise HTTPException(status_code=404, detail=_err("not_found", "product not found"))
    result = await session.execute(
        select(Inspection)
        .where(Inspection.product_id == product.id)
        .options(selectinload(Inspection.defects), selectinload(Inspection.product))
        .order_by(Inspection.created_at)
    )
    return [to_inspection_out(i) for i in result.scalars()]


def _err(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": uuid.uuid4().hex[:12]}}

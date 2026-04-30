"""
Admin aggregate: multi-customer pipeline visibility + embedding-alignment scores.

GET /api/v2/admin/customers-rag-dashboard
    Auth: admin | superadmin
    Tenant list: stems from *.json config files under app/core/configs and configs/.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.guards import require_role
from app.core.pipeline_factory import pipeline_factory
from app.api.v2.embedding_alignment_api import compute_embedding_alignment_report
from app.services.admin.config_client_registry import (
    get_config_search_paths,
    list_config_client_stems,
)
from app.utils.tenant_validator import TenantValidationError, validate_tenant_id
from app.utils.tenant_storage_uuid import storage_business_uuid_for_tenant
from app.db.models.ingested_file_v2 import IngestedFileV2
from app.db.session_v2 import get_db
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/admin",
    tags=["Admin — Multi-Customer RAG"],
)


class PipelineBlock(BaseModel):
    cached: bool = False
    health: Optional[Dict[str, Any]] = None
    health_error: Optional[str] = None


class AlignmentSummary(BaseModel):
    overall_score: int
    ingestion_ready: bool
    readiness_label: str
    overall_score_updated_at: str
    embedder_model_id: Optional[str] = None
    catalog_available: bool = False


class RagEvalMeta(BaseModel):
    """Static capability info (POST evals need golden datasets per call)."""

    retrieval_eval_endpoint: str = "/api/v2/rag-eval/retrieval"
    faithfulness_eval_endpoint: str = "/api/v2/rag-eval/faithfulness"
    calibration_endpoint: str = "/api/v2/rag-eval/calibrate"
    evaluation_matrix_endpoint: str = "/api/v2/rag-eval/evaluation-matrix"


class CustomerRow(BaseModel):
    client_id: str
    validation_error: Optional[str] = None
    pipeline: PipelineBlock = Field(default_factory=PipelineBlock)
    alignment_summary: Optional[AlignmentSummary] = None
    alignment_error: Optional[str] = None
    rag_eval: RagEvalMeta = Field(default_factory=RagEvalMeta)


class CustomersRagDashboardResponse(BaseModel):
    generated_at: str
    status: str  # ok | partial
    config_roots_searched: List[str]
    customer_ids_from_config_files: List[str]
    cached_pipeline_ids: List[str]
    cached_pipeline_count: int
    evaluation_templates_count: int
    customers: List[CustomerRow]


class TenantOverviewResponse(BaseModel):
    """Single-tenant admin snapshot (files + config + RAG health) without cross-tenant data."""
    client_id: str
    storage_business_id: str
    generated_at: str
    ingestion_files_total: int
    ingestion_files_by_status: Dict[str, int]
    recent_files: List[Dict[str, Any]]
    pipeline: PipelineBlock
    alignment_summary: Optional[AlignmentSummary] = None
    alignment_error: Optional[str] = None
    rag_config_highlight: Dict[str, Any] = Field(default_factory=dict)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summarize_alignment(full: Dict[str, Any], ts: str) -> AlignmentSummary:
    return AlignmentSummary(
        overall_score=int(full.get("overall_score") or 0),
        ingestion_ready=bool(full.get("ingestion_ready")),
        readiness_label=str(full.get("readiness_label") or "Unknown"),
        overall_score_updated_at=ts,
        embedder_model_id=full.get("embedder_model_id"),
        catalog_available=bool(full.get("catalog_available")),
    )


@router.get(
    "/customers-rag-dashboard",
    response_model=CustomersRagDashboardResponse,
)
async def get_customers_rag_dashboard(
    _user: Any = Depends(require_role("admin", "superadmin")),
) -> CustomersRagDashboardResponse:
    paths = get_config_search_paths()
    stems = list_config_client_stems()
    cached_ids = sorted(pipeline_factory.list_cached())
    ts = _iso_now()

    eval_count = 0
    try:
        from app.ai.evaluation.rag_evaluator import RAGEvaluator

        eval_count = len(RAGEvaluator.gemini_pipeline_configs())
    except Exception as exc:
        logger.warning("[customers-rag-dashboard] evaluation matrix unavailable: %s", exc)

    rows: List[CustomerRow] = []
    partial = False

    for stem in stems:
        try:
            vctx = validate_tenant_id(
                stem,
                source="dashboard",
                endpoint="customers_rag_dashboard",
                allow_default=True,
            )
            cid = vctx.tenant_id
        except TenantValidationError as e:
            partial = True
            rows.append(
                CustomerRow(
                    client_id=stem,
                    validation_error=str(e),
                    rag_eval=RagEvalMeta(),
                )
            )
            continue

        pl = PipelineBlock(cached=cid in cached_ids)
        if pl.cached:
            try:
                pipe = pipeline_factory.get_cached(cid)
                if pipe is not None:
                    pl.health = pipe.health_check()
            except Exception as ex:
                pl.health_error = str(ex)[:500]
                partial = True
        align_err: Optional[str] = None
        align_summary: Optional[AlignmentSummary] = None
        try:
            report = await compute_embedding_alignment_report(
                cid,
                allow_default=True,
                catch_tenant_error=True,
            )
            reason = report.get("reason")
            comps = report.get("component_checks") or []
            if reason and "Invalid tenant" in str(reason):
                align_err = str(reason)
                partial = True
            elif comps and any(
                isinstance(c, dict) and c.get("component") == "config"
                for c in comps
            ):
                align_err = str(reason or "config resolution failed")
                partial = True
            else:
                align_summary = _summarize_alignment(report, ts)
        except Exception as ex:
            align_err = str(ex)[:500]
            partial = True

        rows.append(
            CustomerRow(
                client_id=cid,
                pipeline=pl,
                alignment_summary=align_summary,
                alignment_error=align_err,
                rag_eval=RagEvalMeta(),
            )
        )

    return CustomersRagDashboardResponse(
        generated_at=ts,
        status="partial" if partial else "ok",
        config_roots_searched=paths if paths else [],
        customer_ids_from_config_files=stems,
        cached_pipeline_ids=cached_ids,
        cached_pipeline_count=len(cached_ids),
        evaluation_templates_count=eval_count,
        customers=sorted(rows, key=lambda r: r.client_id),
    )


@router.get(
    "/tenants/{client_id}/overview",
    response_model=TenantOverviewResponse,
)
async def get_tenant_overview(
    client_id: str,
    db: AsyncSession = Depends(get_db),
    _user: Any = Depends(require_role("admin", "superadmin")),
) -> TenantOverviewResponse:
    """
    Aggregated, tenant-scoped view for one client: ingestion rows for that tenant only,
    plus pipeline health, alignment summary, and safe config highlights.
    """
    ts = _iso_now()
    try:
        vctx = validate_tenant_id(
            client_id,
            source="path",
            endpoint="tenant_overview",
            allow_default=True,
        )
        cid = vctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    suid = storage_business_uuid_for_tenant(cid, None)

    if cid == "default":
        stmt = (
            select(IngestedFileV2)
            .where(
                or_(
                    IngestedFileV2.business_id == suid,
                    IngestedFileV2.business_id.is_(None),
                )
            )
            .order_by(IngestedFileV2.created_at.desc())
        )
    else:
        stmt = (
            select(IngestedFileV2)
            .where(IngestedFileV2.business_id == suid)
            .order_by(IngestedFileV2.created_at.desc())
        )

    res = await db.execute(stmt)
    file_rows = list(res.scalars().all())
    by_status = Counter((r.status or "unknown") for r in file_rows)
    recent_files: List[Dict[str, Any]] = []
    for f in file_rows[:50]:
        recent_files.append(
            {
                "id": str(f.id),
                "file_name": f.file_name,
                "file_type": f.file_type,
                "status": f.status,
                "total_chunks": f.total_chunks,
                "created_at": (
                    f.created_at.isoformat() if getattr(f, "created_at", None) else None
                ),
            }
        )

    cached_ids = sorted(pipeline_factory.list_cached())
    pl = PipelineBlock(cached=cid in cached_ids)
    if pl.cached:
        try:
            pipe = pipeline_factory.get_cached(cid)
            if pipe is not None:
                pl.health = pipe.health_check()
        except Exception as ex:
            pl.health_error = str(ex)[:500]

    align_err: Optional[str] = None
    align_sum: Optional[AlignmentSummary] = None
    try:
        report = await compute_embedding_alignment_report(
            cid,
            allow_default=True,
            catch_tenant_error=True,
        )
        reason = report.get("reason")
        comps = report.get("component_checks") or []
        if comps and any(
            isinstance(c, dict) and c.get("component") == "config"
            for c in comps
        ):
            align_err = str(reason or "config resolution failed")
        else:
            align_sum = _summarize_alignment(report, ts)
    except Exception as ex:
        align_err = str(ex)[:500]

    cfg_h: Dict[str, Any] = {}
    try:
        from app.core.config.client_config_resolver import get_client_config

        cfg = get_client_config(cid)
        cfg_h = {
            "embedder_type": cfg.embedder.type.value,
            "vectordb_type": cfg.vectordb.type.value,
        }
        if cfg.llm is not None:
            cfg_h["llm_type"] = cfg.llm.type.value
    except Exception as ex:
        cfg_h = {"resolution_error": str(ex)[:400]}

    return TenantOverviewResponse(
        client_id=cid,
        storage_business_id=str(suid),
        generated_at=ts,
        ingestion_files_total=len(file_rows),
        ingestion_files_by_status=dict(by_status),
        recent_files=recent_files,
        pipeline=pl,
        alignment_summary=align_sum,
        alignment_error=align_err,
        rag_config_highlight=cfg_h,
    )


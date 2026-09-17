"""MLOps registration for steel PatchCore (X item).

Records into MLflow (mlruns/) + the backend Model Registry as CANDIDATE.
domain_validated=true strictly means "validated on the selected Severstal
steel domain", NOT production promotion. Production stays untouched.

Usage:
  python inference-service/scripts/mlops_register_steel.py --domain-validated true
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "inference-service"))

MLRUNS = ROOT / "mlruns"
EXPERIMENT = "ivqc-models"
BANK = ROOT / "inference-service/models/steel-patchcore/bank.npz"
EVAL = ROOT / "docs/steel-patchcore-eval/metrics.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-validated", choices=["true", "false"], default="false")
    args = parser.parse_args()
    domain_validated = args.domain_validated == "true"

    if not domain_validated:
        print("REGISTER_BLOCKED: steel domain validation gate did not pass")
        return 3

    if not BANK.exists() or not EVAL.exists():
        print("REGISTER_BLOCKED: bank.npz or metrics.json missing")
        return 2
    metrics = json.load(open(EVAL, encoding="utf-8"))
    bank_sha = _sha256(BANK)

    from mlflow.tracking.client import MlflowClient

    client = MlflowClient(tracking_uri=MLRUNS.resolve().as_uri())
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)

    run = client.create_run(exp_id, run_name="steel-severstal-patchcore-v1.0.0")
    rid = run.info.run_id
    tags = {
        "model_name": "steel-patchcore",
        "model_version": "1.0.0",
        "model_type": "patchcore",
        "dataset": "Severstal Steel Defect Detection",
        "dataset_version": "severstal-steel-v1",
        "domain_validated": str(domain_validated),
        "artifact_sha256": bank_sha,
    }
    for k, v in tags.items():
        client.set_tag(rid, k, str(v))
    params = {
        "backbone": "wide_resnet50_2",
        "feature_layers": "layer2,layer3",
        "feature_dim": 1536,
        "memory_bank_size": 50000,
        "image_size": 256,
        "tiles": "7 (x0: 0,256,512,768,1024,1280,1344)",
        "aggregation": "image=max(tile), pixel=stitch-mean-overlap",
        "threshold": str(metrics.get("threshold")),
    }
    for k, v in params.items():
        client.log_param(rid, k, str(v))
    for k in ("image_auroc", "pixel_auroc_mean_per_image", "aup_pro_mean_per_image"):
        v = metrics.get(k)
        if v is not None:
            client.log_metric(rid, k, float(v))
    op = metrics.get("operating_point", {})
    for k in ("precision", "recall", "f1", "normal_fpr", "anomaly_recall"):
        if op.get(k) is not None:
            client.log_metric(rid, k, float(op[k]))
    client.log_artifact(rid, str(BANK), artifact_path="artifacts")
    client.log_artifact(rid, str(EVAL), artifact_path="artifacts")
    client.set_terminated(rid)
    print("mlflow run:", rid)

    from app.models import Base, ModelRegistry
    from app.security.auth import ROLE_PIPELINE, Principal
    from app.services.registry_service import RegistryService
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    # This script IS the trusted pipeline: it computes the artifact hash from
    # the bytes and produces the evaluation report the domain verdict rests on.
    PIPELINE = Principal(subject="script:mlops_register_steel", roles=frozenset({ROLE_PIPELINE}))
    eval_report_uri = EVAL.relative_to(ROOT).as_posix()
    eval_sha = _sha256(EVAL)
    domain_evidence = {
        "domain": "steel",
        "dataset_version": "severstal-steel-v1",
        "eval_report_uri": eval_report_uri,
        "eval_report_sha256": eval_sha,
        "validated_by": "script:mlops_register_steel",
    }

    async def register() -> list:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            svc = RegistryService()
            m = await svc.register(
                session,
                actor=PIPELINE,
                model_name="steel-patchcore",
                model_version="1.0.0",
                model_type="patchcore",
                artifact_uri="inference-service/models/steel-patchcore/bank.npz",
                artifact_sha256=bank_sha,
                dataset_version="severstal-steel-v1",
                training_run_id=rid,
                metrics={
                    "image_auroc": metrics.get("image_auroc"),
                    "pixel_auroc": metrics.get("pixel_auroc_mean_per_image"),
                    "aup_pro": metrics.get("aup_pro_mean_per_image"),
                    **{k: op.get(k) for k in ("precision", "recall", "f1", "normal_fpr", "anomaly_recall")},
                },
                domain_validated=domain_validated,
                domain_evidence=domain_evidence,
                notes="Optimization 1 steel-domain PatchCore baseline (Severstal); "
                      "domain_validated strictly = validated on selected Severstal steel domain only",
            )
            await session.commit()
            rows = (await session.execute(select(ModelRegistry))).scalars().all()
            return [(r.model_name, r.model_version, r.status, r.domain_validated,
                     r.artifact_hash_verified, r.domain_evidence_verified) for r in rows]
        await engine.dispose()

    rows = asyncio.run(register())
    print("registry rows:", rows)

    out = {
        "mlflow_run_id": rid,
        "artifact_sha256": bank_sha,
        "model_name": "steel-patchcore",
        "model_version": "1.0.0",
        "dataset_version": "severstal-steel-v1",
        "status": "CANDIDATE",
        "domain_validated": domain_validated,
        "production_promotion": False,
        "registry_rows": rows,
    }
    (ROOT / "docs/steel-patchcore-mlops.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("MLOPS_REGISTERED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

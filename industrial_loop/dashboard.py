"""Phase 6 — production dashboard (FastAPI + embedded zero-dependency SPA).

Framework note: React/Streamlit were evaluated; a self-contained vanilla-JS
single page served by FastAPI was chosen so the dashboard runs fully offline
with no build step and no CDN, while the REST API below stays
framework-agnostic (a React client can be attached to the same endpoints).

Endpoints:
    GET /                     embedded dashboard page
    GET /api/summary          totals + subsystem status
    GET /api/events           recent inspection events (filterable)
    GET /api/anomalies/recent recent REJECT/HOLD with heatmap previews
    GET /api/trend            decision trend buckets
    GET /api/work-orders      MES work orders
    GET /api/reviews          human-review records + queue depth
    GET /api/plc/state        PLC state + action counters
"""
from __future__ import annotations

import threading
from collections import deque

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from .events import Decision, InspectionEvent, utc_now_iso


class LoopStore:
    """Thread-safe registry shared by the loop services and the dashboard."""

    TREND_BUCKETS = 20

    def __init__(self, *, plc=None, mes=None, review=None, maxlen: int = 5000) -> None:  # noqa: ANN001
        self.plc = plc
        self.mes = mes
        self.review = review
        self._events: deque[InspectionEvent] = deque(maxlen=maxlen)
        self._previews: dict[str, list] = {}
        self._lock = threading.Lock()

    def add_event(self, event: InspectionEvent, heatmap_preview: list | None = None) -> None:
        with self._lock:
            self._events.append(event)
            if heatmap_preview is not None:
                self._previews[event.id] = heatmap_preview

    def update_event(self, event: InspectionEvent) -> None:
        with self._lock:
            for index, existing in enumerate(self._events):
                if existing.id == event.id:
                    self._events[index] = event
                    return
            self._events.append(event)

    # -- queries --------------------------------------------------------------

    def events(self, limit: int = 100, decision: str | None = None) -> list[dict]:
        with self._lock:
            rows = [e.short() for e in self._events]
        if decision:
            rows = [r for r in rows if r["decision"] == decision]
        return list(reversed(rows))[:limit]

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)
        counts = {d.value: 0 for d in Decision}
        for event in events:
            counts[event.decision.value] += 1
        return {
            "updated_at": utc_now_iso(),
            "total": len(events),
            "pass": counts["PASS"],
            "reject": counts["REJECT"],
            "hold": counts["HOLD"],
            "plc": self._plc_snapshot(),
            "mes": self.mes.counts() if self.mes is not None else {},
            "reviews": self.review.counts() if self.review is not None else {},
        }

    def _plc_snapshot(self) -> dict:
        if self.plc is None:
            return {}
        return {
            "state": self.plc.state.value,
            "counters": dict(self.plc.counters),
            "commands_executed": len(self.plc.executed),
        }

    def recent_anomalies(self, limit: int = 12) -> list[dict]:
        with self._lock:
            rows = []
            for event in reversed(self._events):
                if event.decision in (Decision.REJECT, Decision.HOLD):
                    rows.append(
                        {
                            **event.short(),
                            "heatmap_reference": event.heatmap_reference,
                            "heatmap_preview": self._previews.get(event.id),
                        }
                    )
                    if len(rows) >= limit:
                        break
            return rows

    def trend(self) -> list[dict]:
        with self._lock:
            events = list(self._events)
        total = len(events)
        buckets = min(self.TREND_BUCKETS, max(1, total))
        size = max(1, -(-total // buckets))
        out = []
        for start in range(0, total, size):
            chunk = events[start : start + size]
            out.append(
                {
                    "from_index": start,
                    "to_index": start + len(chunk) - 1,
                    "pass": sum(1 for e in chunk if e.decision is Decision.PASS),
                    "reject": sum(1 for e in chunk if e.decision is Decision.REJECT),
                    "hold": sum(1 for e in chunk if e.decision is Decision.HOLD),
                }
            )
        return out


def create_app(
    store: LoopStore,
    runtime_manager=None,  # noqa: ANN001 - industrial_runtime.EdgeRuntimeManager, optional
    drift_detector=None,  # noqa: ANN001 - monitoring.drift.DriftDetector, optional
    lifecycle_manager=None,  # noqa: ANN001 - model_governance.ModelLifecycleManager, optional
) -> FastAPI:
    app = FastAPI(title="IndustrialVision-QC Closed-Loop Dashboard", version="1.0.0")

    @app.get("/api/summary")
    async def api_summary() -> dict:
        return store.summary()

    @app.get("/api/events")
    async def api_events(limit: int = Query(default=100, le=1000), decision: str | None = None) -> list[dict]:
        return store.events(limit=limit, decision=decision)

    @app.get("/api/anomalies/recent")
    async def api_anomalies(limit: int = Query(default=12, le=50)) -> list[dict]:
        return store.recent_anomalies(limit=limit)

    @app.get("/api/trend")
    async def api_trend() -> list[dict]:
        return store.trend()

    @app.get("/api/work-orders")
    async def api_work_orders() -> list[dict]:
        if store.mes is None:
            return []
        return [o.model_dump() for o in store.mes.list()][::-1][:200]

    @app.get("/api/reviews")
    async def api_reviews() -> dict:
        if store.review is None:
            return {"records": [], "counts": {}}
        return {
            "records": [r.model_dump() for r in store.review.records()][::-1][:200],
            "counts": store.review.counts(),
        }

    @app.get("/api/plc/state")
    async def api_plc_state() -> dict:
        return store._plc_snapshot()

    # --- edge runtime + drift monitoring extension (Phase 3) ---

    @app.get("/api/runtime/status")
    async def api_runtime_status() -> dict:
        if runtime_manager is None:
            return {"available": False}
        return {
            "available": True,
            "runtime": runtime_manager.get_status(),
            "health": runtime_manager.health_check(),
        }

    @app.get("/api/runtime/history")
    async def api_runtime_history() -> list[dict]:
        if runtime_manager is None:
            return []
        return [m.as_dict() for m in runtime_manager.monitor.history()]

    @app.get("/api/drift/status")
    async def api_drift_status() -> dict:
        if drift_detector is None:
            return {"available": False}
        latest = drift_detector.latest()
        return {
            "available": True,
            "state": latest.state.value if latest else None,
            "thresholds": drift_detector.thresholds.as_dict(),
            "latest": latest.as_dict() if latest else None,
        }

    @app.get("/api/drift/history")
    async def api_drift_history() -> list[dict]:
        if drift_detector is None:
            return []
        return [r.as_dict() for r in drift_detector.history()]

    @app.get("/api/operations")
    async def api_operations() -> dict:
        if lifecycle_manager is None:
            return {
                "available": False,
                "current_model_version": None,
                "lifecycle_state": None,
                "artifact_hash": None,
                "rollback_status": None,
            }
        return lifecycle_manager.operations_snapshot()

    @app.get("/api/model")
    async def api_model() -> dict:
        if lifecycle_manager is None:
            return {"available": False, "versions": [], "history": []}
        return lifecycle_manager.model_snapshot()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _PAGE

    @app.get("/operations", response_class=HTMLResponse)
    async def operations_page() -> str:
        return _PAGE

    @app.get("/model", response_class=HTMLResponse)
    async def model_page() -> str:
        return _PAGE

    return app


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>工业 AI 视觉质检与异常闭环处理系统</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family:'Segoe UI',system-ui,sans-serif; background:#0f1419; color:#d8dee6; }
  header { padding:14px 22px; background:#161d26; border-bottom:1px solid #232c38;
           display:flex; align-items:center; gap:14px; }
  header h1 { font-size:17px; margin:0; letter-spacing:.4px; }
  .pill { font-size:11px; padding:3px 10px; border-radius:10px; background:#20303f; }
  main { padding:18px 22px; display:grid; gap:16px; }
  .cards { display:flex; gap:14px; flex-wrap:wrap; }
  .card { flex:1; min-width:150px; background:#161d26; border:1px solid #232c38;
          border-radius:10px; padding:14px 16px; }
  .card h2 { margin:0; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#7d8b9d; }
  .card .v { font-size:30px; font-weight:600; margin-top:6px; }
  .pass .v{color:#39c26d} .reject .v{color:#e5484d} .hold .v{color:#f5a524} .total .v{color:#4cc2ff}
  section { background:#161d26; border:1px solid #232c38; border-radius:10px; padding:14px 16px; }
  section h2 { margin:0 0 10px; font-size:13px; color:#9fb0c3; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th { text-align:left; color:#7d8b9d; font-weight:500; padding:4px 8px; border-bottom:1px solid #232c38;}
  td { padding:4px 8px; border-bottom:1px solid #1b2430; }
  .tag { padding:2px 8px; border-radius:8px; font-size:11px; }
  .t-PASS{background:#12351f;color:#39c26d}.t-REJECT{background:#3a1518;color:#ff7b81}
  .t-HOLD{background:#3a2c10;color:#ffc86b}
  canvas { image-rendering: pixelated; width:64px; height:64px; border-radius:4px; }
  .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
  @media (max-width:900px){ .grid2{grid-template-columns:1fr;} }
  .mono { font-family:ui-monospace,Consolas,monospace; font-size:11.5px; color:#8fa3b8; }
  .nav a { color:#9fb0c3; text-decoration:none; font-size:13px; padding:4px 10px; border-radius:8px; }
  .nav a:hover { background:#20303f; color:#fff; }
</style>
</head>
<body>
<header>
  <h1>工业 AI 视觉质检与异常闭环处理系统</h1>
  <nav class="nav">
    <a href="#/" data-page="live">Live</a>
    <a href="#/runtime" data-page="runtime">Runtime</a>
    <a href="#/drift" data-page="drift">Drift</a>
    <a href="#/operations" data-page="operations">Operations</a>
    <a href="#/model" data-page="model">Model</a>
  </nav>
  <span class="pill" id="plc-state">PLC ?</span>
  <span class="pill" id="updated">—</span>
</header>
<main>
  <div id="page-live">
  <div class="cards">
    <div class="card total"><h2>Total inspected</h2><div class="v" id="c-total">0</div></div>
    <div class="card pass"><h2>PASS</h2><div class="v" id="c-pass">0</div></div>
    <div class="card reject"><h2>REJECT</h2><div class="v" id="c-reject">0</div></div>
    <div class="card hold"><h2>HOLD</h2><div class="v" id="c-hold">0</div></div>
    <div class="card"><h2>MES open orders</h2><div class="v" id="c-mes">0</div></div>
    <div class="card"><h2>Reviews pending</h2><div class="v" id="c-rev">0</div></div>
  </div>
  <section><h2>Decision trend</h2><svg id="trend" width="100%" height="120" viewBox="0 0 800 120" preserveAspectRatio="none"></svg></section>
  <div class="grid2">
    <section><h2>Recent anomalies (heatmap)</h2>
      <table id="anomalies"><thead><tr><th>preview</th><th>product</th><th>decision</th><th>image score</th><th>pixel score</th><th>ref</th></tr></thead><tbody></tbody></table>
    </section>
    <section><h2>Latest events</h2>
      <table id="events"><thead><tr><th>time</th><th>product</th><th>camera</th><th>decision</th><th>reason</th><th>PLC</th><th>MES</th><th>operator</th></tr></thead><tbody></tbody></table>
    </section>
  </div>
  </div><!-- /page-live -->

  <div id="page-runtime" style="display:none">
    <div class="cards">
      <div class="card total"><h2>Runtime state</h2><div class="v" id="rt-state">—</div></div>
      <div class="card"><h2>CPU %</h2><div class="v" id="rt-cpu">—</div></div>
      <div class="card"><h2>Memory MB</h2><div class="v" id="rt-mem">—</div></div>
      <div class="card"><h2>GPU MB</h2><div class="v" id="rt-gpu">—</div></div>
      <div class="card"><h2>Latency ms</h2><div class="v" id="rt-lat">—</div></div>
      <div class="card"><h2>Throughput r/s</h2><div class="v" id="rt-tps">—</div></div>
    </div>
    <section><h2>Services</h2>
      <table id="rt-services"><thead><tr><th>service</th><th>status</th><th>last error</th></tr></thead><tbody></tbody></table>
    </section>
    <section><h2>Metrics history</h2>
      <table id="rt-history"><thead><tr><th>time</th><th>cpu %</th><th>mem MB</th><th>gpu MB</th><th>latency ms</th><th>requests</th><th>errors</th><th>r/s</th></tr></thead><tbody></tbody></table>
    </section>
  </div>

  <div id="page-drift" style="display:none">
    <div class="cards">
      <div class="card hold"><h2>Drift state</h2><div class="v" id="dr-state">—</div></div>
      <div class="card total"><h2>PSI (mean)</h2><div class="v" id="dr-psi">—</div></div>
      <div class="card"><h2>PSI (max dim)</h2><div class="v" id="dr-psimax">—</div></div>
      <div class="card"><h2>Cosine shift</h2><div class="v" id="dr-cos">—</div></div>
      <div class="card"><h2>Embedding dist</h2><div class="v" id="dr-dist">—</div></div>
      <div class="card"><h2>Evaluations</h2><div class="v" id="dr-evals">—</div></div>
    </div>
    <section><h2>Thresholds (config-driven)</h2><div class="mono" id="dr-thresholds">—</div></section>
    <section><h2>Evaluation history</h2>
      <table id="dr-history"><thead><tr><th>time</th><th>state</th><th>psi mean</th><th>psi max</th><th>cosine</th><th>dist</th><th>alerts</th></tr></thead><tbody></tbody></table>
    </section>
  </div>

  <div id="page-operations" style="display:none">
    <div class="cards">
      <div class="card total"><h2>Current model</h2><div class="v" id="op-version">—</div></div>
      <div class="card"><h2>Lifecycle state</h2><div class="v" id="op-state">—</div></div>
      <div class="card"><h2>Rollback status</h2><div class="v" id="op-rollback">—</div></div>
    </div>
    <section><h2>Immutable artifact hash</h2><div class="mono" id="op-hash">—</div></section>
  </div>

  <div id="page-model" style="display:none">
    <section><h2>Version history and approvals</h2>
      <table id="model-versions"><thead><tr><th>version</th><th>state</th><th>approval</th><th>metrics</th><th>artifact hash</th><th>timestamp</th></tr></thead><tbody></tbody></table>
    </section>
  </div>
</main>
<script>
const $ = (id) => document.getElementById(id);
function tag(d){ return `<span class="tag t-${d}">${d}</span>`; }
function drawTrend(rows){
  const svg = $("trend"); const W=800,H=120;
  if(!rows.length){ svg.innerHTML=""; return; }
  const max = Math.max(1,...rows.map(r=>r.pass+r.reject+r.hold));
  const bw = W/rows.length;
  let out="";
  rows.forEach((r,i)=>{
    const h = v => v/max*(H-16);
    const x = i*bw;
    const hp=h(r.pass), hr=h(r.reject), hh=h(r.hold);
    let y=H-2;
    [["#39c26d",hp],["#e5484d",hr],["#f5a524",hh]].forEach(([c,vh])=>{
      y-=vh; out+=`<rect x="${x+1}" y="${y}" width="${Math.max(bw-2,1)}" height="${vh}" fill="${c}"/>`;
    });
  });
  svg.innerHTML=out;
}
async function refresh(){
  try{
    const s = await (await fetch("/api/summary")).json();    $("c-total").textContent=s.total; $("c-pass").textContent=s.pass;
    $("c-reject").textContent=s.reject; $("c-hold").textContent=s.hold;
    $("c-mes").textContent=(s.mes&&s.mes.OPEN)||0; $("c-rev").textContent=(s.reviews&&s.reviews.pending)||0;
    $("plc-state").textContent="PLC "+((s.plc&&s.plc.state)||"?");
    $("updated").textContent=s.updated_at;
    drawTrend(await (await fetch("/api/trend")).json());
    const anomalies = await (await fetch("/api/anomalies/recent")).json();
    $("anomalies").tBodies[0].innerHTML = anomalies.map(a=>{
      let cell="<span class='mono'>n/a</span>";
      if(a.heatmap_preview){
        const g=a.heatmap_preview, n=g.length, cellN=g[0].length;
        cell=`<canvas width="${cellN}" height="${n}" data-g='${JSON.stringify(g)}'></canvas>`;
      }
      return `<tr>${cell.startsWith("<canvas")?`<td>${cell}</td>`:`<td>${cell}</td>`}
        <td class="mono">${a.product_id}</td><td>${tag(a.decision)}</td>
        <td class="mono">${a.image_score??"—"}</td><td class="mono">${a.pixel_score??"—"}</td>
        <td class="mono">${(a.heatmap_reference||"").slice(0,28)}</td></tr>`;
    }).join("");
    document.querySelectorAll("canvas[data-g]").forEach(cv=>{
      const ctx=cv.getContext("2d"), g=JSON.parse(cv.dataset.g);
      g.forEach((row,y)=>row.forEach((v,x)=>{
        const c=Math.min(255,Math.round(v*255));
        ctx.fillStyle=`rgb(${c},${Math.round(40+c*0.25)},${Math.round(60-c*0.15)})`;
        ctx.fillRect(x,y,1,1);
      }));
    });
    const events = await (await fetch("/api/events?limit=15")).json();
    $("events").tBodies[0].innerHTML = events.map(e=>`<tr>
      <td class="mono">${(e.timestamp||"").slice(11,19)}</td><td class="mono">${e.product_id}</td>
      <td class="mono">${e.camera_id}</td><td>${tag(e.decision)}</td>
      <td class="mono">${e.reason_code}</td><td class="mono">${e.plc_status}</td>
      <td class="mono">${e.mes_status}</td><td class="mono">${e.operator_status}</td></tr>`).join("");
  }catch(err){ console.error(err); }
}
async function refreshRuntime(){
  try{
    const st = await (await fetch("/api/runtime/status")).json();
    if(!st.available){ $("rt-state").textContent="n/a"; return; }
    $("rt-state").textContent = st.runtime.state;
    $("rt-services").tBodies[0].innerHTML = Object.entries(st.runtime.services)
      .map(([name, info]) => {
        const svc = (st.health.services||{})[name] || "unknown";
        return `<tr><td class="mono">${name}</td><td class="mono">${svc}</td>
                <td class="mono">${(info.last_error||"—").slice(0,60)}</td></tr>`;
      }).join("");
    const m = st.health.metrics || {};
    $("rt-cpu").textContent = m.cpu_percent ?? "—";
    $("rt-mem").textContent = m.memory_mb ?? "—";
    $("rt-gpu").textContent = m.gpu_memory_mb ?? "n/a";
    $("rt-lat").textContent = m.latency_ms ?? "—";
    $("rt-tps").textContent = m.requests_per_second ?? "—";
    const hist = await (await fetch("/api/runtime/history")).json();
    $("rt-history").tBodies[0].innerHTML = hist.slice(-12).reverse().map(r=>`<tr>
      <td class="mono">${(r.timestamp||"").slice(11,19)}</td>
      <td class="mono">${r.cpu_percent}</td><td class="mono">${r.memory_mb}</td>
      <td class="mono">${r.gpu_memory_mb ?? "n/a"}</td><td class="mono">${r.latency_ms ?? "—"}</td>
      <td class="mono">${r.request_count}</td><td class="mono">${r.error_count}</td>
      <td class="mono">${r.requests_per_second}</td></tr>`).join("");
  }catch(err){ console.error(err); }
}
async function refreshDrift(){
  try{
    const st = await (await fetch("/api/drift/status")).json();
    if(!st.available){ $("dr-state").textContent="n/a"; return; }
    $("dr-state").textContent = st.state || "NO DATA";
    $("dr-evals").textContent = st.latest ? "see history" : "0";
    $("dr-thresholds").textContent = JSON.stringify(st.thresholds);
    if(st.latest){
      $("dr-psi").textContent = st.latest.psi_mean;
      $("dr-psimax").textContent = st.latest.psi_max;
      $("dr-cos").textContent = st.latest.cosine_shift;
      $("dr-dist").textContent = st.latest.mean_distance;
    }
    const hist = await (await fetch("/api/drift/history")).json();
    $("dr-history").tBodies[0].innerHTML = hist.slice().reverse().map(r=>`<tr>
      <td class="mono">${(r.timestamp||"").slice(11,19)}</td>
      <td>${tag(r.state==="CRITICAL"?"REJECT":r.state==="WARNING"?"HOLD":"PASS")}</td>
      <td class="mono">${r.psi_mean}</td><td class="mono">${r.psi_max}</td>
      <td class="mono">${r.cosine_shift}</td><td class="mono">${r.mean_distance}</td>
      <td class="mono">${(r.alerts||[]).join("; ").slice(0,50)||"—"}</td></tr>`).join("");
  }catch(err){ console.error(err); }
}
async function refreshOperations(){
  try{
    const op = await (await fetch("/api/operations")).json();
    $("op-version").textContent = op.current_model_version || "n/a";
    $("op-state").textContent = op.lifecycle_state || "n/a";
    $("op-hash").textContent = op.artifact_hash || "n/a";
    $("op-rollback").textContent = op.rollback_status ? op.rollback_status.status : "NOT EXECUTED";
  }catch(err){ console.error(err); }
}
async function refreshModel(){
  try{
    const data = await (await fetch("/api/model")).json();
    $("model-versions").tBodies[0].innerHTML = (data.versions || []).map(row=>`<tr>
      <td class="mono">${row.model_version}</td><td>${row.state}</td>
      <td>${row.approval_status}</td><td class="mono">${JSON.stringify(row.metrics)}</td>
      <td class="mono">${row.artifact_hash}</td><td class="mono">${row.timestamp}</td></tr>`).join("");
  }catch(err){ console.error(err); }
}
function route(){
  let page = (location.hash || "").replace("#/","");
  if(!page) page = location.pathname === "/operations" ? "operations" : location.pathname === "/model" ? "model" : "live";
  for(const id of ["live","runtime","drift","operations","model"]){
    document.getElementById("page-"+id).style.display = id===page ? "" : "none";
  }
  if(page==="runtime") refreshRuntime();
  if(page==="drift") refreshDrift();
  if(page==="operations") refreshOperations();
  if(page==="model") refreshModel();
}
window.addEventListener("hashchange", route);
route();
refresh(); setInterval(refresh, 2000);
setInterval(()=>{ if((location.hash||"#/")==="#/runtime") refreshRuntime();
                  if(location.hash==="#/drift") refreshDrift();
                  if(location.hash==="#/operations" || location.pathname==="/operations") refreshOperations();
                  if(location.hash==="#/model" || location.pathname==="/model") refreshModel(); }, 3000);
</script>
</body>
</html>"""

import { lazy, Suspense, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const OverviewPage = lazy(() =>
  import("./features/overview/OverviewPage").then((module) => ({ default: module.OverviewPage })),
);
const LivePage = lazy(() =>
  import("./features/live/LivePage").then((module) => ({ default: module.LivePage })),
);
const TracePage = lazy(() =>
  import("./features/trace/TracePage").then((module) => ({ default: module.TracePage })),
);
const ReviewQueuePage = lazy(() =>
  import("./features/review/ReviewQueuePage").then((module) => ({ default: module.ReviewQueuePage })),
);
const ModelOpsPage = lazy(() =>
  import("./features/modelops/ModelOpsPage").then((module) => ({ default: module.ModelOpsPage })),
);
const CopilotPage = lazy(() =>
  import("./features/copilot/CopilotPage").then((module) => ({ default: module.CopilotPage })),
);

type Tab = "overview" | "live" | "review" | "trace" | "modelops" | "copilot";

const TABS: Array<{ key: Tab; label: string }> = [
  { key: "overview", label: "Production Overview" },
  { key: "live", label: "Live Inspection" },
  { key: "review", label: "Review Queue" },
  { key: "trace", label: "Quality Traceability" },
  { key: "modelops", label: "Model Operations" },
  { key: "copilot", label: "Quality Copilot" },
];

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 1500, refetchOnWindowFocus: false } },
});

function Dashboard() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">IVQC</span>
          <span className="brand-title">工业 AI 视觉质检与异常闭环处理系统</span>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={`tab ${tab === t.key ? "active" : ""}`} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">
        <Suspense fallback={<div className="panel">Loading…</div>}>
          {tab === "overview" ? <OverviewPage /> : null}
          {tab === "live" ? <LivePage /> : null}
          {tab === "review" ? <ReviewQueuePage /> : null}
          {tab === "trace" ? <TracePage /> : null}
          {tab === "modelops" ? <ModelOpsPage /> : null}
          {tab === "copilot" ? <CopilotPage /> : null}
        </Suspense>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  );
}

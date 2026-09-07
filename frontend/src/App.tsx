import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { OverviewPage } from "./features/overview/OverviewPage";
import { LivePage } from "./features/live/LivePage";
import { TracePage } from "./features/trace/TracePage";
import { ReviewQueuePage } from "./features/review/ReviewQueuePage";
import { ModelOpsPage } from "./features/modelops/ModelOpsPage";
import { CopilotPage } from "./features/copilot/CopilotPage";

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
        {tab === "overview" ? <OverviewPage /> : null}
        {tab === "live" ? <LivePage /> : null}
        {tab === "review" ? <ReviewQueuePage /> : null}
        {tab === "trace" ? <TracePage /> : null}
        {tab === "modelops" ? <ModelOpsPage /> : null}
        {tab === "copilot" ? <CopilotPage /> : null}
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

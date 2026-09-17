// Backend API contract types (mirror of the backend schemas, strict).

export type WsConnectionState = "connecting" | "connected" | "disconnected" | "reconnecting" | "reconnected";

export type QualityResult = "PASS" | "REVIEW" | "FAIL";
export type ProcessStatus = "PENDING" | "COMPLETED" | "FAILED";
export type Severity = "low" | "medium" | "high" | "critical";
export type WsEventType =
  | "inspection.completed"
  | "inspection.failed"
  | "review.created"
  | "review.claimed"
  | "review.resolved";

export type ReviewTaskStatus = "PENDING" | "IN_REVIEW" | "RESOLVED";
export type HumanDecision = "PASS" | "CONFIRM_DEFECT" | "CORRECT_DEFECT" | "OTHER_DEFECT";

export interface Defect {
  id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  bbox_normalized: [number, number, number, number];
  defect_area_px: number;
  defect_area_ratio: number;
  severity: Severity | null;
  matched_rule: string | null;
}

export interface Product {
  product_id: string;
  production_line: string;
  station: string;
  created_at: string;
}

export interface Inspection {
  inspection_id: string;
  product_id: string;
  batch_id: string | null;
  image_url: string | null;
  status: ProcessStatus;
  quality_result: QualityResult | null;
  final_quality_result?: QualityResult | null;
  severity: Severity | null;
  model_name: string | null;
  model_version: string | null;
  rule_version: number | null;
  inference_latency_ms: number | null;
  error_message: string | null;
  created_at: string;
  defects: Defect[];
  product?: Product;
  // Phase 6 anomaly + fusion
  anomaly_score?: number | null;
  anomaly_threshold?: number | null;
  is_anomalous?: boolean | null;
  anomaly_map_url?: string | null;
  anomaly_model_version?: string | null;
  anomaly_regions?: Array<{ bbox_xyxy: number[]; bbox_normalized: number[]; area_ratio: number; region_score: number }> | null;
  fusion_class?: string | null;
  // Phase 7 industrial (three-layer semantics)
  desired_command?: string | null;
  execution_status?: string | null;
  industrial_state?: string | null;
  industrial_final_state?: string | null;
  plc_command?: string | null;
  plc_status?: string | null;
  plc_adapter_type?: string | null;
  plc_reason_code?: string | null;
  plc_latency_ms?: number | null;
  mes_sync_status?: string | null;
}

export interface RealtimeStatus {
  // quality / persisted facts (DB-owned, single coherent snapshot)
  completed_total: number;
  failed_total: number;
  pass_total: number;
  review_total: number;
  fail_total: number;
  total_inspected: number;
  yield_rate: number | null;
  // runtime telemetry (pipeline view, refreshed independently)
  captured_total: number;
  queued_current: number;
  processing_current: number;
  queue_depth: number;
  throughput: number;
  // freshness
  snapshot_at: string;
  telemetry_at: string | null;
  // remaining
  queue_peak_depth: number;
  simulator_running: boolean;
  simulator_interval_ms: number | null;
  worker_count: number | null;
  queue_size: number | null;
  current_throughput: number;
  average_processing_latency_ms: number | null;
  p50_latency_ms: number | null;
  p95_latency_ms: number | null;
  average_inference_latency_ms: number | null;
  uptime_seconds: number;
  ws_client_count: number;
}

export interface InspectionEvent {
  event_id: string;
  event_type: "inspection.completed" | "inspection.failed";
  timestamp: string;
  product_id: string;
  inspection_id: string;
  batch_id: string | null;
  production_line: string;
  station: string;
  process_status: ProcessStatus;
  quality_result: QualityResult | null;
  severity: Severity | null;
  defect_count: number;
  inference_latency_ms: number | null;
  model_version: string | null;
  error_message: string | null;
}

export interface InspectionFilters {
  product_id?: string;
  inspection_id?: string;
  batch_id?: string;
  quality_result?: QualityResult;
  status?: ProcessStatus;
  defect_type?: string;
  production_line?: string;
  station?: string;
  date_from?: string;
  date_to?: string;
  limit?: number;
  offset?: number;
}

export interface ReviewDecision {
  id: string;
  review_task_id: string;
  inspection_id: string;
  reviewer: string;
  ai_quality_result: string;
  ai_defects_snapshot: DefectSnapshot[];
  human_decision: HumanDecision;
  human_label: string | null;
  final_quality_result: QualityResult;
  reason: string | null;
  created_at: string;
}

export interface DefectSnapshot {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  bbox_normalized: [number, number, number, number];
  defect_area_px: number;
  defect_area_ratio: number;
  severity: Severity | null;
  matched_rule: string | null;
}

export interface ReviewTask {
  review_task_id: string;
  inspection_id: string;
  inspection: Inspection | null;
  status: ReviewTaskStatus;
  priority: number;
  assigned_to: string | null;
  claimed_at: string | null;
  resolved_at: string | null;
  version: number;
  ai_quality_result: string;
  ai_defects_snapshot: DefectSnapshot[];
  ai_model_version: string | null;
  ai_rule_version: number | null;
  ai_severity: string | null;
  product_id: string;
  production_line: string;
  station: string;
  batch_id: string | null;
  image_url: string | null;
  decision: ReviewDecision | null;
  // Phase 6 anomaly snapshot (6G)
  anomaly_score?: number | null;
  anomaly_threshold?: number | null;
  is_anomalous?: boolean | null;
  anomaly_regions?: Array<{ bbox_xyxy: number[]; bbox_normalized: number[]; area_ratio: number; region_score: number }> | null;
  anomaly_map_url?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReviewEvent {
  event_id: string;
  event_type: "review.created" | "review.claimed" | "review.resolved";
  timestamp: string;
  review_task_id: string;
  inspection_id: string;
  product_id: string;
  status: ReviewTaskStatus;
  priority: number | null;
  assigned_to: string | null;
  reviewer: string | null;
  human_decision: HumanDecision | null;
  final_quality_result: QualityResult | null;
  top_defect_class: string | null;
  top_confidence: number | null;
  severity: string | null;
  model_version: string | null;
  image_url: string | null;
}

export type WsEvent = InspectionEvent | ReviewEvent;

export interface ReviewMetrics {
  pending_review_count: number;
  pending: number;
  in_review: number;
  resolved: number;
  average_review_wait_time_s: number | null;
  review_rate: number | null;
  defect_confirmation_rate: number | null;
  ai_human_label_agreement_rate: number | null;
  override_rate: number | null;
  corrected_label_count: number;
  pass_overrides: number;
}

export interface TrainingCandidate {
  inspection_id: string;
  image_url: string | null;
  ai_label: string | null;
  human_label: string | null;
  ai_confidence: number | null;
  agreement: boolean;
  review_reason: string | null;
  source_dataset_version: string | null;
  source_model_version: string | null;
  source_deployment_version: string | null;
  timestamp: string;
}

// ---- Phase 8 MLOps ----

export type ModelStatus = "CANDIDATE" | "STAGING" | "PRODUCTION" | "ARCHIVED";

export interface RegistryModel {
  id: string;
  model_name: string;
  model_version: string;
  model_type: "yolo" | "patchcore";
  artifact_uri: string | null;
  artifact_sha256: string | null;
  dataset_version: string | null;
  training_run_id: string | null;
  status: ModelStatus;
  metrics: Record<string, number>;
  domain_validated: boolean;
  promoted_at: string | null;
  notes: string | null;
  created_at: string | null;
}

export interface GateCheck {
  check: string;
  passed: boolean;
  got: number | boolean | null;
  required: number | boolean;
  blocked_by_domain?: boolean;
}

export interface GateResult {
  passed: boolean;
  checks: GateCheck[];
  blocked: string[];
}

export interface ModelMetrics {
  model_version: string | null;
  window_count: number;
  inference_count: number;
  error_count: number;
  error_rate: number | null;
  inference_latency_avg_ms: number | null;
  inference_latency_p95_ms: number | null;
  review_rate: number | null;
  confidence_distribution: number[];
  defect_distribution: Record<string, number>;
  anomaly_score_distribution: number[];
}

export interface HumanFeedback {
  filters: { model_version: string | null; defect_type: string | null; line: string | null; station: string | null };
  resolved: number;
  defect_confirmation_rate: number | null;
  ai_human_label_agreement_rate: number | null;
  pass_override_rate: number | null;
  corrected_label_rate: number | null;
  per_defect: Record<string, { defect_confirmation_rate: number | null; ai_human_label_agreement_rate: number | null; pass_override_rate: number | null; corrected_label_rate: number | null; resolved: number }>;
}

export interface DriftReport {
  model_version: string | null;
  baseline_window: { from: string; to: string; n: number };
  current_window: { from: string; to: string; n: number };
  overall: "NORMAL" | "WARNING" | "CRITICAL";
  signals: Record<string, { score?: number; level: "NORMAL" | "WARNING" | "CRITICAL"; baseline?: number; current?: number; max_delta?: number; baseline_n?: number; current_n?: number }>;
  note: string;
}

// ---- Phase 9 Quality Copilot ----

export interface CopilotToolCall {
  name: string;
  arguments: Record<string, unknown>;
  latency_ms?: number;
}

export interface CopilotLatency {
  llm_latency_ms: number;
  total_latency_ms: number;
  tool_call_count: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number;
}

export interface CopilotEvidence {
  tool: string;
  latency_ms: number;
  time_window?: string;
  [key: string]: unknown;
}

export interface CopilotResponse {
  conversation_id: string;
  message: string;
  evidence: CopilotEvidence[];
  tools_used: string[];
  tool_calls: CopilotToolCall[];
  limitations: string[];
  confidence: "low" | "medium" | "high";
  latency: CopilotLatency;
  safety: { read_only: boolean; write_actions_performed: string[] };
  timestamp: string;
}

export interface CopilotConversation {
  id: string;
  created_at: number;
  updated_at: number;
  turns: Array<{ role: string; content: string; tools: string[] }>;
}

// ---- Error Analysis Pipeline + Model Quality Gate ----

export type QualityGateVerdict = "PASS" | "HOLD" | "NOT_ENFORCED";

export interface QualityGateRule {
  rule: string;
  message: string;
  got?: number | null;
  required?: number | string | null;
  direction?: "min" | "max";
  source?: string | null;
  category?: string;
}

export interface QualityGateEnforcement {
  enforced_model_types: string[];
  require_evaluation_evidence: boolean;
  enabled: boolean;
}

export interface QualityGatePolicySummary {
  policy_id: string;
  policy_sha256: string;
  policy_pinned: boolean;
  policy_status: string;
  policy_path?: string;
  policy_notes?: string[];
  enforcement: QualityGateEnforcement;
  evidence_requirements?: Record<string, boolean>;
  max_evidence_age_days?: number | null;
  model_types?: string[];
  thresholds_used?: Record<string, number>;
  rule_sources?: Record<string, string | null>;
}

export interface QualityGateEvidenceRef {
  evaluation_id?: string | null;
  fingerprint: string | null;
  model_name: string | null;
  model_version: string | null;
  dataset: string | null;
  split: string | null;
  sample_count: number | null;
  threshold: number | null;
  evaluation_time: string | null;
}

export interface QualityGateResult {
  verdict: QualityGateVerdict;
  passed: boolean;
  enforced: boolean;
  allows_promotion: boolean;
  checks: Array<{ check: string; passed: boolean; got: unknown; required: unknown; category: string }>;
  failed_rules: QualityGateRule[];
  failed_rule_codes: string[];
  warnings: QualityGateRule[];
  warning_codes: string[];
  metrics: Record<string, number>;
  policy?: QualityGatePolicySummary;
  evidence?: QualityGateEvidenceRef | null;
  reason?: string;
  timestamp: string;
}

export interface ModelEvaluationMetrics {
  precision: number | null;
  recall: number | null;
  f1: number | null;
  false_accept_rate: number | null;
  false_reject_rate: number | null;
  latency_p95_ms: number | null;
}

export interface ModelEvaluation {
  id: string;
  registry_id: string | null;
  model_name: string;
  model_version: string;
  model_type: string | null;
  task: string;
  dataset: { name: string; split: string };
  sample_count: number;
  threshold: number | null;
  metrics: ModelEvaluationMetrics;
  confusion_matrix: number[][];
  report_sha256: string;
  report_uri: string | null;
  attested_by: string | null;
  evaluation_time: string | null;
  created_at: string | null;
}

export interface ModelEvaluationList {
  model: string;
  count: number;
  evaluations: ModelEvaluation[];
}

export interface QualityGateResponse {
  model: string;
  quality_gate: QualityGateResult;
}

export interface ModelGateResponse {
  model: string;
  gate: GateResult;
  quality_gate: QualityGateResult;
  promotion_allowed: boolean;
}

export interface QualityGatePolicyResponse {
  available: boolean;
  enforcement_enabled: boolean;
  verdicts: QualityGateVerdict[];
  policy?: QualityGatePolicySummary;
  error?: string;
  note?: string;
}

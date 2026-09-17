# Evaluation Evidence 与 Model Quality Gate

回答一个聚合指标永远回答不了的问题：**模型具体错在哪里，以及这份「它是对的」的证据
能不能经得起一份书面政策的审查。**

本阶段交付一条完整的评测证据链，起点是逐样本的预测记录，终点是晋升决策被阻断或放行。
链路上每一个数字都可以被审计者独立重算。

## 1. 为什么需要独立于 promotion gate 的第二道门

仓库里原本已有 `app.mlops.promotion_gate`，它回答「被背书的聚合指标够好吗」。这道门回答的
问题不同：「是否存在一份能证明它的评测，且这份评测能通过质量政策」。

两者必须同时通过，且刻意分开，因为消费的证据不同。promotion gate 读的是聚合指标背书，
quality gate 读的是带混淆矩阵与失败样本清单的逐样本评测报告。把两者合并会让「指标好看」
悄悄替代「证据存在」，而这正是本阶段要堵的漏洞。

## 2. 语义层：positive 的唯一权威定义

工业评测管线最具破坏性的失效模式是「正类」的含义在指标代码、门禁政策、测试与看板之间漂移。
`app.evaluation.semantics` 存在的唯一目的就是让任何人都不必猜。

两个任务族回答同一个车间问题：这个单元发给客户，还是进废品箱。

| 概念 | 统计记法 | 车间记法 |
|---|---|---|
| 实际有缺陷却放行 | False Negative | **False Accept (FA)** |
| 实际完好却拦下 | False Positive | **False Reject (FR)** |

正类因此是「绝不能被放行的东西」。anomaly_detection (PatchCore) 的正类为 ANOMALY，
defect_detection (YOLO) 的正类为 DEFECT。两个族都在 `score >= threshold` 时判为正。

由此得到本仓库固定的错误恒等式：

    false_accept_rate (FAR) = fn / (tp + fn) = 1 - recall      漏检率
    false_reject_rate (FRR) = fp / (fp + tn) = 1 - specificity 误杀率

FA 与 FN 在数值上相同、在责任归属上完全不同。把它当成同义词是本文件要消除的第一类混淆。

## 3. 评测核心（纯函数，无 DB / HTTP / 模型加载 / GPU）

`backend/app/evaluation/` 是七个模块组成的纯计算包。这份纯度是设计约束而非巧合，同一份代码
必须能被离线基准脚本、测试夹具与服务端门禁三处调用，并在三处产出逐字节一致的数字。

| 模块 | 职责 |
|---|---|
| `records.py` | `PredictionRecord`，预测由 `(score, threshold)` 推导 |
| `semantics.py` | positive 定义与 FA/FR 恒等式 |
| `metrics.py` | 混淆矩阵、分类指标、延迟统计 |
| `threshold.py` | 阈值扫描与描述性参考点 |
| `slices.py` | 切片诊断 |
| `report.py` | 结构化评测报告与指纹 |

三条贯穿全包的设计决定值得单独说明。

**预测不由调用方声明。** `PredictionRecord` 从 `(score, threshold)` 推导预测。调用方若同时
传入预测标签，该标签会被校验，不一致即硬错误。这让一份报告无法声称其自身分数支撑不了的预测，
也让阈值扫描成为精确操作而非近似。

**未定义指标返回 `None`。** 分母为零时不返回 0.0，也不返回 NaN。读 `None` 的门禁必须 fail closed，
读 `None` 的看板必须渲染「未定义」。唯一例外是 precision 与 recall 均有定义且同时为零时的 F1，
那是真实的 0.0，与 `steel_patchcore.aggregation.operating_point` 的既有约定一致。

**阈值结论止于描述。** `threshold.py` 的参考点按所满足的准则命名（`best_f1`、`recall_oriented`、
`conservative`），不按它们支撑不了的裁决命名。没有任何模块在此宣布某阈值为「生产最优」，
因为那个判断需要一个经测量的业务成本函数（放行缺陷的代价与报废好件的代价），本仓库没有这样的函数。

## 4. 切片纪律

`slices.py` 只对预测记录上确实存在的字段做切片。支持维度为 `dataset`、`split`、`model_name`、
`model_version`、`defect_type`、`score_range`。

没有光照切片，没有相机切片，没有产线切片，没有材料切片，没有环境切片，因为这些字段在预测记录上
并不存在。虚构它们会产出一张看起来像洞察、实则是装饰的图表。

`score_range` 是唯一的派生维度，切点即工作阈值（低于 / 达到或高于）。这是本仓库唯一具有确定含义的切法。
样本量低于 `min_slice_size` 的切片只报计数，比率保持 `None`。三个样本算出的比率是噪声，
一个读噪声的门禁比没有门禁更危险。

## 5. 评测报告与指纹

`report.py` 产出的是门禁读取、看板渲染、审计者重算的那份制品。三条性质决定其结构。

**自描述。** 报告携带自身的 schema 版本、任务语义与阈值，读者无需猜测「positive」当时指什么。

**可复现。** `fingerprint` 是规范载荷去掉时钟后的 SHA256，相同输入永远得到相同摘要。

**对证据诚实。** 报告包含逐样本失败清单，不只包含比率。因此「召回率 0.61」后面永远可以接上
「以及它漏掉的是这些单元」。

持久化不在此模块的职责内，`app.services.evaluation_service` 负责存储，`app.mlops.quality_gate`
负责裁决。

## 6. Model Quality Gate

`app.mlops.quality_gate` 拥有裁决权。它是一个 `(policy, evidence)` 上的纯函数，性质如下。

- **确定性**。相同证据与相同政策永远得到相同裁决，无模型、无采样、无随机打破平局。
- **可配置**。每一个数字都活在 `backend/config/quality_gate_policy.yaml` 中，进程按 SHA256 固定该文件。
- **可审计**。裁决携带政策身份、政策摘要、被评估的模型版本、证据指纹、每一条失败规则的机器可读与
  人类可读形式，以及时间戳。
- **fail closed**。政策缺失、证据记录缺失、报告结构不完整、指标不可读，一律产出 HOLD。
  不存在「降级为警告」的路径。

判定取值为 `PASS` / `HOLD` / `NOT_ENFORCED` 三者之一。

**没有覆盖路径。** 与晋升政策不同，这里连 tighten-only 的运行时收紧入口都不提供，因为本门禁读取的
证据由受信管线产出，调用方没有正当理由去重新判定它。一个能在运行时被说服放弃自身裁决的门禁
不构成门禁。没有任何 LLM 决定模型是否能上线，裁决是写在政策之上的算术。

门禁读不到政策时，受影响的晋升一律 fail closed，而非静默跳过。

## 7. 政策文件的来源纪律

`source` 在每条规则上都是必填字段，记录该数字从何而来。加载器拒绝缺少该字段的文件。
一个无人能追溯的阈值，是一个在审计中无人能辩护的阈值。

| 规则 | 值 | 来源 |
|---|---|---|
| patchcore recall (min) | 0.60 | `steel_patchcore/aggregation.py` DEVELOPMENT_GATE.anomaly_recall_min |
| patchcore false_accept_rate (max) | 0.40 | 0.60 召回下界的对偶，FAR == 1 - recall |
| patchcore false_reject_rate (max) | 0.10 | 同上 DEVELOPMENT_GATE.normal_fpr_max |
| patchcore latency_p95_ms (max) | 2000.0 | `backend/config/promotion_policy.yaml` |
| yolo recall (min) | 0.60 | `promotion_policy.yaml` thresholds.yolo.recall |
| yolo latency_p95_ms (max) | 120.0 | `promotion_policy.yaml` thresholds.yolo.latency_p95_ms |
| sample_count (min) | 100 | **开发占位值，本仓库尚无抽样方案** |

`sample_count = 100` 在本仓库中没有任何先例定义。它被明确标记为开发占位值，必须在状态升到
`production` 之前用真实抽样方案推导出的数字替换。加载器在任一条规则仍引用演示值时拒绝
`status: production`。

## 8. 强制范围：空是刻意的

`enforcement.enforced_model_types` 当前为 `[]`。这个空列表是一个有意为之且有记录的决定。

本仓库当前注册的模型是 Phase 6 / Phase 8 基准基线，它们从未有逐样本评测被记录。事后向它们
追索一份评测是伪造门禁，而非严格门禁。一旦第一个候选模型有了已存储的评测报告，该列表即被填充，
门禁开始拦截那些无法自证的候选。

在该模型族缺席期间门禁返回 `NOT_ENFORCED`，且该裁决连同政策身份与摘要一并写入治理日志。
豁免因而是可见的，而不是隐藏状态。`GET /api/v1/model-quality-gate/policy` 会暴露
`enforcement_enabled: false`，让读者能看见门禁已配置、看见其摘要、看见强制列表仍为空。

## 9. 提交路径与信任边界

`app.services.evaluation_service` 负责摄取。存储是追加式的，永不更新。

- 报告经 `verify_report` 校验后才可能入库。
- 提交者须持有 pipeline 或 admin 角色。
- 报告中的模型身份必须与注册表条目一致，不一致报 `report_model_mismatch`。
- `evaluation_time` 为必填，缺失报 `report_evaluation_time_missing`，因为证据年龄需要可判定。
- API 侧另有 HMAC 背书：签名覆盖报告的 SHA256，服务端从收到的 body 重算该摘要，
  因此途中被改动的报告无法通过验证。

## 10. API

| 方法与路径 | 角色 | 说明 |
|---|---|---|
| `POST /api/v1/models/{id}/evaluations` | pipeline | 提交评测证据 |
| `GET /api/v1/models/{id}/evaluations` | viewer | 列出该模型版本的证据 |
| `GET /api/v1/models/{id}/quality-gate` | viewer | 试算门禁，不写日志 |
| `GET /api/v1/evaluations/{id}` | viewer | 单份评测及其摘要 |
| `GET /api/v1/model-quality-gate/policy` | viewer | 当前生效的政策及其强制状态 |

试算端点不写入治理日志，其存在是为了回答「此刻门禁会说什么」。真正落进治理日志的门禁评估，
发生在一次真实晋升尝试期间。

## 11. 接入晋升链路

`app.services.registry_service.promote` 在构造晋升决策时会调用
`evaluation_service.quality_gate_for`。当裁决为 held 时，晋升被阻断并写入 `block: "quality_gate_hold"`，
同时附上完整门禁结果。因此评测证据不是一份旁路报告，它是晋升路径上的一个硬性前置条件。

## 12. 数据结构

迁移 `0012_model_evaluations`（`down_revision = 0011_business_audit_log`）新增 `model_evaluations` 表。

没有向 `model_registry` 增加任何列，因为评测证据是一条时间序列（同一版本随时间可能有多份评测），
而非一个字段。门禁读取它所裁决的那个版本的最新一行。

## 13. 验证

| 范围 | 命令 | 结果 |
|---|---|---|
| 评测核心 + 门禁 | `pytest backend/tests/test_evaluation_*.py backend/tests/test_quality_gate.py` | 94 passed |
| CI 应然范围 | `pytest backend/tests inference-service/tests/test_vision_contract.py -m "not integration and not gpu and not opcua and not industrial-e2e and not artifact"` | 293 passed, 1 skipped, 22 deselected |
| 路由鉴权审计 | `pytest backend/tests/test_security_rbac.py` | 17 passed |
| 迁移链 | 0012 → 0011 → … → 0001 | 线性完整，无分叉 |

路由鉴权审计为递归式，新增业务路由若缺少鉴权依赖即失败。本阶段新增的五条评测路由全部带角色依赖。

## 14. Files

- backend/app/evaluation/{records,semantics,metrics,threshold,slices,report,__init__}.py
- backend/app/mlops/quality_gate.py；backend/app/mlops/gate_policy.py
- backend/config/quality_gate_policy.yaml
- backend/app/services/evaluation_service.py；backend/app/services/registry_service.py
- backend/app/api/evaluations.py；backend/app/main.py
- backend/alembic/versions/0012_model_evaluations.py；backend/app/models.py
- backend/tests/{eval_helpers,test_evaluation_lifecycle,test_evaluation_metrics,test_evaluation_report,test_evaluation_threshold,test_quality_gate}.py

## Known Issues

1. 政策 `status` 仍为 `development`，`sample_count` 规则引用占位值。在写出真实抽样方案之前，
   该政策只能约束候选，不能认证产线。
2. `enforcement.enforced_model_types` 为空，因此门禁对当前全部已注册模型返回 `NOT_ENFORCED`。
   豁免有记录且可见，但这意味着本阶段交付的是一个已就位但尚未生效的门禁。
3. `freshness.max_evidence_age_days` 为 `null`，证据年龄检查被禁用。在写出「评测多久算过期」
   的政策之前，陈旧的评测不会被拒绝。
4. 本仓库缺少经测量的业务成本函数，因此阈值结论止于描述性参考点。任何「生产最优阈值」的说法
   在当前证据下都不成立。

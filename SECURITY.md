# Security Policy

## Supported Versions

Security fixes target the default branch and the latest published release. Older release tags are immutable historical snapshots and receive fixes only when explicitly announced.

| Version | Supported |
|---|---|
| `main` | Yes |
| Latest release | Yes |
| Older releases | Best effort |

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability or include credentials, proprietary images, model artifacts, device addresses, or plant topology in a public report.

Use [GitHub private vulnerability reporting](https://github.com/ten10do/industrial-ai-vision-qc/security/advisories/new). Include:

- affected revision, component, and deployment mode;
- reproducible steps or a minimal proof of concept;
- expected and observed security impact;
- whether the issue can create an unsafe PASS, bypass artifact verification, expose data, or alter industrial commands;
- sanitized logs and suggested mitigation, if available.

The maintainer will review the report, request missing evidence when needed, coordinate remediation and disclosure, and credit reporters who wish to be acknowledged. No fixed response or remediation SLA is promised for this reference implementation.

## Security Boundaries

The repository is not a safety-certified control system. Physical interlocks, machine safety, network segmentation, identity, credential management, and site authorization remain the responsibility of the deploying organization.

Security-sensitive changes must preserve these invariants:

- invalid, unavailable, non-finite, or mismatched inference evidence cannot become PASS;
- model manifests and artifacts are verified before loading;
- the frozen D3 model, threshold, whitening state, and memory banks are not mutated online;
- PLC/MES retries remain idempotent;
- logs, issues, tests, and demo assets contain no production credentials or proprietary factory data.

See the [deployment guide](docs/operations/deployment-guide.md), [rollback guide](docs/operations/rollback-guide.md), and [architecture decisions](docs/engineering-decisions/architecture-decisions.md).

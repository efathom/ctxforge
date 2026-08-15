# Security Policy

## Reporting a Vulnerability

We take security issues seriously. If you believe you have found a security vulnerability in ctxforge, please report it responsibly rather than opening a public issue.

**Do not** open a GitHub issue or PR with the details. Instead, report it privately via one of:

- **GitHub**: use the repository's *Private vulnerability reporting* feature (Security → Report a vulnerability), or
- **Email**: `security@efathom.com`

Please include:

- A clear description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept if available
- The affected version(s) / commit(s)
- Any suggested remediation

## What to expect

- You will receive an acknowledgment within a few business days.
- We will validate the report and keep you informed of our assessment and fix timeline.
- Once a fix is available we will publish a security advisory and credit the reporter (unless you prefer to remain anonymous).

## Supported versions

| Version | Supported |
|---------|-----------|
| latest `main` | :white_check_mark: |
| tagged releases | :white_check_mark: (latest release) |

## Security best practices for deployments

When running ctxforge in production:

- **Never commit real credentials** — use environment variables or a secret manager for API keys (`llm.api_key`, vector-store keys, database passwords, etc.).
- **Enable PII redaction** middleware (`pipelines.*` → `pii`) if user data flows through your context, and audit middleware if you need an activity trail.
- **Restrict the executable-skill runtime** (`skills.executable_runtime`) to trusted authors, or use `isolation: subprocess` with resource limits.
- **Review middleware pipelines** (`pipelines.prepare` / `pipelines.record`) for any data you don't want persisted, logged, or forwarded to LLM providers.
- **Pin and review dependencies** — Dependabot updates are enabled, but review them before merging.

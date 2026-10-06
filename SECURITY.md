# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.x     | :white_check_mark: |
| < 3.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within Membrane, please report it responsibly.

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via [GitHub Security Advisories](https://github.com/sachncs/membrane/security/advisories/new).

### What to include

- Type of vulnerability (e.g., buffer overflow, SQL injection, cross-site scripting)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will confirm the vulnerability and determine its impact within 7 days.
- **Resolution**: We aim to release a fix within 30 days of confirmation.
- **Disclosure**: We will coordinate with you on the timing of public disclosure.

## Security Best Practices

When deploying Membrane in production, follow the
[hardening checklist](https://sachncs.github.io/membrane/docs/security/#hardening-checklist)
in the security guide. In short:

- Enable authentication: `membrane serve` refuses public binds without an API keyfile or mTLS.
- Use mTLS (or TLS at a reverse proxy) for all network traffic, including between peers.
- Keep the `--data-dir` key in a secret manager (`--data-key-file`), not on the data volume.
- Run the service as a non-root user (the Docker image already does this).
- Use Redis authentication and TLS when connecting to external Redis instances.
- Keep dependencies updated regularly.
- Restrict network access to only required ports.
- Monitor logs for suspicious activity.

## Dependency Security

Dependencies are pinned in `uv.lock`, and GitHub Dependabot checks them for
updates weekly. Every change also runs pip-audit (known vulnerabilities in the
locked dependencies), bandit (static analysis), gitleaks (committed secrets),
and Trivy (filesystem and dependency scan) in CI.

## Scope

This security policy applies to the official Membrane package distributed through this repository. It does not apply to third-party integrations or forks maintained by others.

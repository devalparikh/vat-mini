# Security Posture Report

Date: 2026-07-10

## Executive summary

VAT Mini has a good security posture for its stated purpose: a local, educational ML sandbox with a static learning site. The review found no embedded secrets, no browser XSS sinks, no network/authentication attack surface, and no known vulnerabilities in the installed application dependencies. The test suite and production frontend build also pass.

The original review found one high-severity checkpoint deserialization issue. It was remediated on 2026-07-10 by enabling PyTorch's restricted `weights_only=True` loader, validating the payload before applying it, and adding a regression test that proves executable serialized objects are rejected.

Overall rating: **good for local use; imported datasets and resource-intensive configs should still be treated as untrusted**.

## Scope and threat model

Reviewed:

- Python 3.11/3.12 CLI, configuration, dataset, checkpoint, training, and evaluation paths.
- React 19 / Vite 6 static learning site.
- Python and npm dependency manifests and installed dependency trees.
- Secret patterns, dangerous deserialization, command execution, path/file handling, browser injection sinks, client storage, cross-origin messaging, and production browser protections.

The repository contains no backend server, database, authentication, authorization, API, cookies, or remote requests. Consequently, common server-side risks such as SQL injection, SSRF, CSRF, broken access control, and session theft are not applicable to the current code.

## High severity

### SEC-001 — Unrestricted checkpoint deserialization can execute arbitrary code

- **Status:** Resolved on 2026-07-10.
- **Rule ID:** PY-DESER-001
- **Severity:** High
- **Location:** `src/vat_mini/checkpoint.py`, `load_checkpoint`, line 58; exposed through `src/vat_mini/cli.py`, lines 51-55 and 66-67.
- **Original evidence:** `torch.load(path, map_location="cpu", weights_only=False)` explicitly enabled unrestricted pickle-compatible loading. Both `train --checkpoint` and `evaluate --checkpoint` accept a caller-selected path.
- **Impact:** A malicious or tampered checkpoint can execute code as the local user as soon as it is loaded. An attacker must first convince the user to download and open the checkpoint or replace a trusted local artifact.
- **Implemented fix:** `load_checkpoint` now uses `weights_only=True`, converts rejected executable objects into a clear safe error, and validates the top-level payload, model state, and optimizer state before applying them.
- **Mitigation:** Until fixed, load only checkpoints created locally by this repository; do not open models from email, shared drives, model hubs, pull-request artifacts, or other untrusted sources. Integrity hashes or signatures can protect trusted distribution channels but do not make pickle safe by themselves.
- **Verification:** All existing checkpoint files can be deserialized in restricted mode, and the training integration test confirms a newly written checkpoint still round-trips into the matching model. `tests/test_checkpoint.py` confirms that a serialized object with an executable reduction is rejected without executing and that malformed payloads are rejected before model loading. Older checkpoints can still fail normal `load_state_dict` compatibility checks if their saved architecture no longer matches the current model.

## Medium severity

### SEC-002 — Python installs are not reproducible or integrity-locked

- **Rule ID:** PY-SUPPLY-001
- **Severity:** Medium
- **Location:** `pyproject.toml`, lines 10-17; `Makefile`, lines 8-10.
- **Evidence:** Runtime and development dependencies use only lower bounds (`numpy>=1.26`, `PyYAML>=6.0`, `torch>=2.2`, `pytest>=8.0`), and `make setup` runs `pip install -e '.[dev]'` without a lock file or hashes.
- **Impact:** Two clean installs can resolve materially different dependency trees. A compromised release, dependency confusion event, breaking major release, or newly vulnerable transitive dependency can enter an environment without a reviewed manifest change.
- **Fix:** Generate and commit a Python lock/constraints file for supported Python/platform combinations, use hashes where practical, and make setup/CI install from that locked input. Add automated dependency auditing.
- **Mitigation:** Build in an isolated environment, use trusted indexes, review resolver output, and retain known-good environment manifests.
- **False positive notes:** Broad ranges are convenient for a learning project and are not a vulnerability on their own. Severity increases if the project is used in CI, shared development environments, or artifact production.

### SEC-003 — The current bootstrap installer has known vulnerabilities

- **Rule ID:** PY-SUPPLY-002
- **Severity:** Medium
- **Location:** local `.venv` bootstrap state; setup flow at `Makefile`, lines 8-10.
- **Evidence:** `pip-audit` identified four advisories in installed `pip 25.3`: `PYSEC-2026-196` / `CVE-2026-8643`, `PYSEC-2026-1796` / `CVE-2026-1703`, `CVE-2026-3219`, and `CVE-2026-6357`. Fixed versions range from pip 26.0 through 26.1.2.
- **Impact:** The advisories affect installation of maliciously crafted distributions, including path-handling and post-install import behavior. Exploitation requires installing an attacker-controlled package, but package installation is exactly the bootstrap trust boundary.
- **Fix:** Upgrade pip to at least 26.1.2 before installing project dependencies, and encode that minimum in the setup workflow.
- **Mitigation:** Install only reviewed packages from trusted indexes and combine the upgrade with a locked dependency set.
- **False positive notes:** `pip` is an environment/bootstrap tool, not an application runtime dependency declared by this project. A newly created environment may contain a different pip version, so CI and developer bootstrap environments should be checked explicitly.

## Low severity / hardening

### SEC-004 — Production browser security headers are not defined in the repository

- **Rule ID:** REACT-HEADERS-001 / REACT-CSP-001
- **Severity:** Low
- **Location:** `learning-site/index.html`, lines 3-12; no deployment/edge configuration is present.
- **Evidence:** The app shell defines basic metadata and a same-origin module script, but the repository has no CSP or hosting configuration for `Content-Security-Policy`, `X-Content-Type-Options`, clickjacking protection, `Referrer-Policy`, or `Permissions-Policy`.
- **Impact:** If the static site is deployed without edge headers, it lacks defense-in-depth against future XSS and can be framed by another site. Current impact is limited because the app renders only constants, uses React escaping, loads no third-party scripts, and handles no sensitive data.
- **Fix:** Configure headers at the eventual host/edge. A suitable starting CSP for the current build is restrictive (`default-src 'self'`; explicitly scope scripts, styles, images, connections, objects, base URIs, and framing) and should be tested against the deployed response.
- **Mitigation:** Keep the frontend free of raw HTML sinks and third-party scripts. Verify headers against the live URL after deployment.
- **False positive notes:** Headers may already be supplied outside this repository. This finding should be closed if runtime response inspection confirms an equivalent policy.

### SEC-005 — Dataset archives and resource-intensive config values are trusted without bounds

- **Rule ID:** PY-INPUT-001
- **Severity:** Low
- **Location:** `src/vat_mini/data.py`, lines 207-220; `src/vat_mini/config.py`, lines 60-90.
- **Evidence:** Existing validation enforces positivity and a few relationships, but does not cap sample counts, image dimensions, sequence length, batch size, worker count, model width/depth, or archive array shapes/dtypes. Existing `.npz` files are loaded and converted to tensors without validating their schema against the resolved config.
- **Impact:** A hostile config or compressed dataset can trigger excessive CPU, memory, disk, or process consumption, or fail deep inside training. This is local denial of service; no remote input path exists today.
- **Fix:** Before allocating or tensor conversion, enforce practical upper bounds and verify archive keys, dtypes, ranks, shapes, finite values, action ranges, and total uncompressed size. Consider refusing archives that do not match the resolved config.
- **Mitigation:** Treat external configs and datasets as untrusted, inspect sizes before use, and run imported experiments in a resource-limited environment.
- **False positive notes:** For locally generated toy datasets, this is a robustness hardening gap rather than an exploitable vulnerability.

## Positive controls observed

- YAML is parsed with `yaml.safe_load`, and unknown configuration fields are rejected.
- NumPy archive loading keeps the safe default `allow_pickle=False`.
- Checkpoints are written atomically through temporary files and replacement.
- The React app contains no `dangerouslySetInnerHTML`, direct HTML injection, `eval`, dynamic navigation, `postMessage`, web storage, network requests, third-party scripts, or service worker.
- Frontend dependencies are exact-versioned and covered by `package-lock.json`.
- No likely secrets, private keys, credentials, `.env` files, or secret-bearing client environment variables were found.
- The Vite production build emits no public source maps by default.

## Verification results

- `pytest -q`: **14 passed**.
- `npm run build`: **passed**.
- `npm audit --json`: **0 known vulnerabilities** across 117 dependencies (including development dependencies), checked 2026-07-10.
- `pip-audit` against the installed Python environment: **0 known vulnerabilities in application/runtime libraries**; four findings in pip 25.3 as described in SEC-003, checked 2026-07-10.
- `pip check`: **no broken requirements**.
- Restricted loading test: every existing repository-generated `.pt` checkpoint loaded successfully with `weights_only=True`.

## Recommended remediation order

1. Fix SEC-001 before accepting or distributing checkpoints outside a fully trusted local workflow.
2. Upgrade pip and add a reproducible Python dependency lock (SEC-002 and SEC-003).
3. Add archive/config resource validation if external datasets or shared experiment configs become part of the workflow.
4. Add and runtime-verify browser security headers when the learning site is deployed.

## Limitations

- This was a source/configuration and dependency audit, not a penetration test or malware analysis of binary artifacts.
- No deployed learning-site URL or edge/CDN configuration was available, so live HTTP headers and TLS were not assessed.
- The Git repository currently has no committed files or history (`git ls-files` and `git log` are empty); secret/history scanning therefore covered only the present working tree, not prior revisions.

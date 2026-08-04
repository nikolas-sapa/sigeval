# Security Policy

## Supported Versions

sigeval is pre-1.0. Only the latest released version is supported with
security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, email niksapa150@gmail.com with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal code sample is ideal).
- Any relevant version/environment details.

You should receive an acknowledgment within a few days. We'll work with you
to understand and address the issue, and to agree on a disclosure timeline
before any public write-up.

## Scope

sigeval is a stdlib-only statistics library that runs inside your test suite
— it does not execute untrusted input, make network calls, or handle secrets
itself. If you find a case where it does something unexpected security-wise
(e.g. via `judge.py`'s handling of a caller-supplied `complete_fn`), that's
in scope.

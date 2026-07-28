# CVM mTLS migration — two-proxy, two-secret attestation provenance

**Status**: Phase I implemented (this branch). Phase II is a code change that removes the legacy path.

## Problem

Attestation endpoints trust the nginx-injected `X-Client-Cert`. A shared proxy secret proves a
request came through a trusted proxy (not forged against `api.chutes.ai`). But two VM
generations coexist during the fleet upgrade, and they don't call the API the same way:

| Endpoint | 1.3.1 base URL | 1.3.1 transport | 1.4.0 |
|---|---|---|---|
| `GET /servers/nonce` | `api.chutes.ai` (VALIDATOR_BASE_URL) | plain TLS, no cert | cvm.chutes.ai, mTLS |
| `POST /servers/boot/attestation` | tdx-attestation (TDX_BASE_URL) | mTLS (throwaway cert) | cvm.chutes.ai, mTLS |
| `POST /servers/{vm}/luks/attest` | tdx-attestation | mTLS (throwaway cert) | cvm.chutes.ai, mTLS |
| `POST /servers/{vm}/luks/confirm` | `api.chutes.ai` | plain TLS, no cert | cvm.chutes.ai, mTLS |
| `provision`, `provision/confirm` | — (don't exist) | — | cvm.chutes.ai, mTLS |

Two consequences:
- 1.3.1 hits **nonce** and **luks/confirm** on `api.chutes.ai`, where **no proxy can inject a
  secret** — so those endpoints can't fail-closed on a proxy secret without breaking the fleet.
- A single secret can't tell "legacy 1.3.1 via attestation-proxy" from "full-mTLS 1.4.0 via
  cvm-proxy", so mTLS can't be enforced only where it's expected.

## Solution: two secrets + named guards

Two proxies, each injecting its own secret; the API discriminates by which matched.

| Proxy | Domain | Header / secret | Fleet |
|---|---|---|---|
| attestation-proxy | tdx-attestation.chutes.ai | `X-Attestation-Proxy-Auth` / `ATTESTATION_PROXY_SECRET` | legacy 1.3.x |
| cvm-proxy | cvm.chutes.ai | `X-Cvm-Proxy-Auth` / `CVM_PROXY_SECRET` | 1.4.0+ |

Each endpoint uses the dependency that matches how both VM generations reach it:

| Dependency | Endpoints | Rule |
|---|---|---|
| `require_attestation_proxy()` | boot/attestation, luks/attest | require **either** secret; 503 if neither configured, 403 if no valid header |
| `require_cvm_proxy()` | provision, provision/confirm | require **cvm** secret; 503 if unconfigured, 403 otherwise (1.4.0-only) |
| `gate_legacy_attestation()` | nonce, luks/confirm | version-gated: allow if via cvm proxy; else look up the VM by caller IP and **reject** if attested `>= tee_mtls_min_version` (a VM that new must use cvm); else allow legacy 1.3.x |

`_get_client_certificate` trusts `X-Client-Cert` only when a request carries a valid secret
from *either* proxy. The public `api.chutes.ai` ingress strips both proxy headers.

## Phase I (this branch) — coexistence

Deployable now so 1.4.0 can be **tested against a live 1.3.1 fleet**:
- API: the named guards above + `CVM_PROXY_SECRET`.
- Charts: **cvm-proxy** (new, fronts all six endpoints, injects `X-Cvm-Proxy-Auth`);
  **attestation-proxy** converted to sed-render + injects `X-Attestation-Proxy-Auth`;
  api-deployment wires both secrets; api-ingress strips provenance headers.

Result: 1.3.1 keeps working on every path (incl. plain-API nonce/confirm); 1.4.0 (all via
cvm-proxy) gets full provenance + cert enforcement; both run simultaneously.

**Rollout:** the secret env vars are wired `optional: true` (the `validator` image is shared across
apps, so a hard env requirement would block them all from starting). Enforcement is at the
per-request guards, which fail closed (503) when a secret is unset — a missing secret rejects
attestation rather than bypassing, without taking the pod down. So the deploy order is: provision
`attestation-proxy-auth` + `cvm-proxy-auth`, roll the proxies (injecting), then the API. Until the
secret is provisioned the guarded endpoints 503; the rest of the API is unaffected.

## Phase II — remove the legacy path (code change)

During the migration, `gate_legacy_attestation` tightens on its own: the moment a VM attests at
`>= tee_mtls_min_version` (default `1.4.0`), its nonce/luks-confirm calls on `api.chutes.ai` are
rejected and it must use the cvm proxy — so VMs move over per-VM as the fleet upgrades, and
`tee_mtls_min_version: "0.0.0"` can force the stragglers. That is an interim safety valve, **not**
the end state.

Once the fleet is fully on 1.4.0, **delete `gate_legacy_attestation` and swap nonce + luks/confirm
to `require_cvm_proxy()`**. Every attestation endpoint then unconditionally requires the cvm
secret. This is deliberate: leaving the legacy branch in place is dead code that a misconfigured
`tee_mtls_min_version` (set too high, or reverted) could silently re-open — removing it eliminates
that regression surface entirely. `tee_mtls_min_version` and the IP lookup can then be dropped too.

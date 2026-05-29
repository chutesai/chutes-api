# Feature Spec: GET /servers/signing-keys API Endpoint

**Date**: 2026-05-28  
**Status**: draft

---

## Context

The `dynamic-signing-keys` feature ([spec](dynamic-signing-keys.md)) removes cosign and Helm PGP keys from the VM image and fetches them dynamically at boot. The `fetch-signing-keys` initramfs init-bottom script performs a `GET` to `VALIDATOR_BASE_URL/servers/signing-keys`, verifies each key's detached PGP signature against the attested root key baked into the image, and writes verified keys to `/run/chutes/signing-keys/` on tmpfs.

This spec defines the API endpoint that must exist on `api.chutes.ai` before any VM image built with the `dynamic-signing-keys` feature can boot.

- **Service affected**: validator API (`api.chutes.ai`)
- **Consumer**: initramfs `fetch-signing-keys` script on every booting chutes-miner-vm

---

## Design Decisions

- **Public endpoint, no authentication**: Cosign and Helm public keys are not secrets. Any authenticated endpoint would block VMs that haven't yet completed their TDX attestation flow. The PGP signature chain provides integrity, not confidentiality.
- **Single GET, JSON bundle**: One request returns all keys and their signatures atomically. This eliminates TOCTOU issues (no partial key set on the VM) and minimizes boot-time network calls.
- **Base64-encoded binary data**: Both the key bytes and PGP signature bytes are binary. Base64 encoding makes them safe in JSON without escaping issues. The consumer does `base64 -d` before writing to disk.
- **Detached PGP signatures (binary, base64-encoded)**: `gpg --detach-sign` (no `--armor`) produces a compact binary packet. The initramfs `gpgv` verifies binary detached signatures directly. Wrapping in base64 for transport, then decoding before `gpgv`, is the simplest path.
- **Key names as bundle map keys**: Using the key name (`cosign/chutes.pub`, `helm-pubkey.gpg`, etc.) as the map key means the consumer can write each key to `$OUTPUT_DIR/<key-name>` without any hardcoded path mapping. Adding a new key type in the future is purely additive.
- **Multiple cosign keys for rotation overlap**: Serving both old and new `cosign/chutes.pub` simultaneously during a rotation window would require the map key to be unique (e.g., `cosign/chutes-v2.pub`). The current spec uses fixed names and assumes the VM fleet reboots within the rotation window. If overlap is needed, key names can be extended (e.g., `cosign/chutes-1.pub`, `cosign/chutes-2.pub`) — the consumer iterates all keys in the bundle.
- **`version` field**: Allows the consumer to reject bundles from a future incompatible schema without crashing. Currently `1`. Bump only on breaking schema changes.
- **No caching headers**: VMs fetch this at each boot, not continuously. The response can be served with short or no TTL. A CDN may cache it but the consumer does not rely on freshness guarantees.

---

## API Changes

- **New endpoint**: `GET /servers/signing-keys`
- **Auth**: None (public)
- **HTTP method**: GET
- **Content-Type**: `application/json`

### Response schema

```json
{
  "version": 1,
  "keys": {
    "<key-name>": "<base64-encoded raw key bytes>"
  },
  "signatures": {
    "<key-name>": "<base64-encoded raw detached PGP signature bytes>"
  }
}
```

**Current key names** (must match exactly — the initramfs consumer validates these are present):

| Key name | Content |
|---|---|
| `cosign/chutes.pub` | Cosign public key for `localregistry.chutes.ai` and wildcard registry |
| `cosign/dockerhub.pub` | Cosign public key for `docker.io/parachutes` org |
| `helm-pubkey.gpg` | GPG public key for Helm chart provenance verification |

The consumer asserts that all three of these names exist in the bundle. Additional keys may be present and will be verified and installed — this enables additive rotation without breaking old VMs.

### Example

```json
{
  "version": 1,
  "keys": {
    "cosign/chutes.pub": "dW50cnVzdGVkIGNvbW1lbnQ6...==",
    "cosign/dockerhub.pub": "dW50cnVzdGVkIGNvbW1lbnQ6...==",
    "helm-pubkey.gpg": "mQINBGQ...=="
  },
  "signatures": {
    "cosign/chutes.pub": "owGbwMvMwCU2...==",
    "cosign/dockerhub.pub": "owGbwMvMwCU2...==",
    "helm-pubkey.gpg": "owGbwMvMwCU2...=="
  }
}
```

### Error responses

The consumer treats any non-2xx response as a retryable failure. After 3 failed attempts with 2-second backoff, the VM powers off. Keep the endpoint always available; there is no graceful degradation for VMs that cannot reach it.

---

## Signing Workflow (Key Bundle Production)

This is the **offline operation** performed whenever a key is rotated or initially published. The root PGP **private** key must be available (offline machine or HSM). After signing, the private key goes back to secure storage.

### One-time setup (generate signatures for existing keys)

```bash
# Sign each leaf key with the root PGP private key.
# --detach-sign: produces a detached signature (not inline)
# No --armor: produces compact binary packet (base64-encoded for the bundle)
gpg --detach-sign -o chutes.pub.sig     ~/.cosign/chutes.pub
gpg --detach-sign -o dockerhub.pub.sig  ~/.cosign/dockerhub.pub
gpg --detach-sign -o helm-pubkey.gpg.sig ~/.chutes/helm-pubkey.gpg
```

### Build the JSON bundle

```bash
#!/bin/bash
set -euo pipefail

bundle_entry() {
    local name="$1"
    local key_file="$2"
    local sig_file="$3"
    local key_b64 sig_b64
    key_b64=$(base64 -w0 < "$key_file")
    sig_b64=$(base64 -w0 < "$sig_file")
    printf '"%s": "%s"' "$name" "$key_b64"
    printf '|'
    printf '"%s": "%s"' "$name" "$sig_b64"
}

CHUTES=$(bundle_entry "cosign/chutes.pub"    ~/.cosign/chutes.pub    chutes.pub.sig)
DOCKER=$(bundle_entry "cosign/dockerhub.pub" ~/.cosign/dockerhub.pub dockerhub.pub.sig)
HELM=$(bundle_entry   "helm-pubkey.gpg"      ~/.chutes/helm-pubkey.gpg helm-pubkey.gpg.sig)

IFS='|'
read -r k1 s1 <<< "$CHUTES"
read -r k2 s2 <<< "$DOCKER"
read -r k3 s3 <<< "$HELM"

jq -n \
  --argjson version 1 \
  --argjson keys "{$k1, $k2, $k3}" \
  --argjson sigs "{$s1, $s2, $s3}" \
  '{"version": $version, "keys": $keys, "signatures": $sigs}'
```

Upload the resulting JSON to the API's backing store (database, object store, or config file — implementation choice).

### Key rotation workflow (post-rollout)

1. Generate the new key pair (e.g., new cosign key pair).
2. On the offline/HSM machine: sign the new public key with the root PGP private key:
   ```bash
   gpg --detach-sign -o new-chutes.pub.sig new-chutes.pub
   ```
3. Rebuild the JSON bundle with the new key replacing the old one (or add it alongside the old one with a new name like `cosign/chutes-2.pub` if you need overlap).
4. Publish the updated bundle to the API. VMs pick it up on next reboot — no image rebuild required.

---

## Goal

Success = VMs running the `dynamic-signing-keys` image boot successfully, fetching and verifying keys from this endpoint. Specifically:

1. `GET /servers/signing-keys` returns HTTP 200 with valid JSON containing `version`, `keys`, and `signatures`.
2. All three required key names (`cosign/chutes.pub`, `cosign/dockerhub.pub`, `helm-pubkey.gpg`) are present in both `keys` and `signatures`.
3. Each signature in `signatures` verifies against the corresponding entry in `keys` using `gpgv` and the root PGP public key at `root-signing-key.gpg`.
4. The endpoint is reachable from inside a TDX VM over public HTTPS at boot time (same network path as `VALIDATOR_BASE_URL`).
5. The endpoint returns within 30 seconds (matching `fetch-signing-keys` `TIMEOUT`).

---

## Constraints

- The root PGP private key must never be present in the API service or its deployment environment. Only signed bundles (public key bytes + detached signatures) are stored and served.
- The endpoint must be served over HTTPS. The initramfs `curl` call uses the system CA bundle at `/etc/ssl/certs/ca-certificates.crt`.
- The endpoint must be available before the first VM with this image version boots. There is no fallback — unreachable API = VM powers off.
- No request body, no query parameters, no auth headers. Plain `GET`.
- Response must be valid JSON parseable by `jq` (the version available in the initramfs, which supports all standard filters used by the consumer).

---

## Failure Conditions

- Endpoint is live but returns a bundle missing one of the three required keys — all VMs with this image will fail to boot.
- Signatures in the bundle were generated with the wrong key (not the root PGP key baked into the image) — all VMs with this image will power off at the signature verification step.
- Endpoint returns HTTP 200 with malformed JSON — `jq` parse fails, VM powers off.
- Key or signature values are not valid base64 — `base64 -d` fails, VM powers off.
- Endpoint is unreachable or returns non-2xx after 3 retries — VM powers off.

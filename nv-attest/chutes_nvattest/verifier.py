from typing import Dict

from nv_attestation_sdk.attestation import Attestation, Devices, Environment


class NvVerifier:

    def attest(self, nonce: str, evidence: list[Dict[str, str]]) -> bool:
        _client = Attestation()
        _client.add_verifier(Devices.GPU, Environment.REMOTE, "", "")
        _client.set_nonce(nonce)
        result = _client.attest(evidence)
        return result

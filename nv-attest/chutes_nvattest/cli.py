import json
import sys
from pathlib import Path

import typer

from chutes_nvattest.verifier import NvVerifier

app = typer.Typer(no_args_is_help=True)


def attest_gpu(
    nonce: str = typer.Option(..., help="The nonce to expect in the evidence"),
    evidence: Path = typer.Option(..., help="File containing evidence json"),
):
    with open(evidence) as fh:
        content = fh.read()
        _evidence = json.loads(content)
    _verifier = NvVerifier()
    result = _verifier.attest(nonce, _evidence)
    if result:
        sys.exit(0)
    else:
        sys.exit(1)


app.command(name="attest-gpu", help="Attest Nvidia GPU evidence.")(attest_gpu)

if __name__ == "__main__":
    app()

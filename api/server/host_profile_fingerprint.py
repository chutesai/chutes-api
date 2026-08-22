"""
Print the topology fingerprint of a discover-profile.sh document.

    python -m api.server.host_profile_fingerprint <profile.json>   # or - for stdin

For the measurement generator. Submission-driven topologies already carry their fingerprint as the
bucket object key; this covers seed / build-time topologies that were never submitted, so the
generator can stamp `fingerprint` onto a hardware entry without reimplementing the algorithm. A
second implementation that drifts silently breaks accepted-detection.
"""

import sys
import json

from api.server.schemas import HostProfile


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip(), file=sys.stderr)
        return 2

    try:
        raw = sys.stdin.read() if argv[1] == "-" else open(argv[1]).read()
        print(HostProfile(**json.loads(raw)).fingerprint)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

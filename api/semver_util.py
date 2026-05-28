import re
import semver


def semcomp(input_version: str, target_version: str) -> int:
    """
    Semver comparison with cleanup.
    """
    if not input_version:
        input_version = "0.0.0"
    re_match = re.match(r"^([0-9]+\.[0-9]+\.[0-9]+).*", input_version)
    clean_version = re_match.group(1) if re_match else "0.0.0"
    return semver.compare(clean_version, target_version)

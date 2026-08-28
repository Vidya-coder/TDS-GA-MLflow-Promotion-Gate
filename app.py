from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any
from datetime import datetime, timezone
import math
import re


app = FastAPI()

SAFE_INT_MAX = 9007199254740991

VERSION_RE = re.compile(r"^[1-9][0-9]*$")

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

REQUIRED_POLICY_FIELDS = {
    "datasetDigest",
    "schemaDigest",
    "maxAgeSeconds",
    "accuracyFloor",
    "requiredSlices",
    "maxLatencyMs",
    "maxSizeBytes",
    "minImprovement",
}


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def is_safe_nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_unit_interval(value: Any) -> bool:
    return is_finite_number(value) and 0 <= float(value) <= 1


def parse_timestamp(value: Any):
    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    try:
        normalized = value

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        dt = datetime.fromisoformat(normalized)

        if dt.tzinfo is None:
            return None

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


def validate_version_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    if not VERSION_RE.fullmatch(value):
        return False

    try:
        number = int(value)
    except (ValueError, OverflowError):
        return False

    return 1 <= number <= SAFE_INT_MAX


def add_gate(failures: dict, version: str, code: str):
    failures.setdefault(version, [])

    if code not in failures[version]:
        failures[version].append(code)


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    if not REQUIRED_POLICY_FIELDS.issubset(policy.keys()):
        return False

    dataset_digest = policy.get("datasetDigest")
    schema_digest = policy.get("schemaDigest")

    if not isinstance(dataset_digest, str) or not dataset_digest:
        return False

    if not isinstance(schema_digest, str) or not schema_digest:
        return False

    if not is_safe_nonnegative_int(policy.get("maxAgeSeconds")):
        return False

    if not is_unit_interval(policy.get("accuracyFloor")):
        return False

    required_slices = policy.get("requiredSlices")

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return False

        if not is_unit_interval(floor):
            return False

    max_latency = policy.get("maxLatencyMs")

    if not is_finite_number(max_latency):
        return False

    if float(max_latency) < 0:
        return False

    if not is_safe_nonnegative_int(policy.get("maxSizeBytes")):
        return False

    if not is_unit_interval(policy.get("minImprovement")):
        return False

    return True


def timestamp_age_seconds(as_of: datetime, created_at: datetime):
    """
    Calculate age without constructing timedelta from policy values.
    This avoids OverflowError when maxAgeSeconds is near the
    JavaScript safe-integer limit.
    """
    return (as_of - created_at).total_seconds()


def evaluate_version(
    version_obj: Any,
    as_of: datetime,
    policy: dict,
    failures: dict,
):
    """
    Returns:
        (valid_id, version_id, eligible)
    """

    if not isinstance(version_obj, dict):
        return False, None, False

    version = version_obj.get("version")

    if not validate_version_id(version):
        return (
            False,
            version if isinstance(version, str) else None,
            False,
        )

    evaluation = version_obj.get("evaluation")

    if not isinstance(evaluation, dict):
        add_gate(
            failures,
            version,
            "MISSING_EVALUATION",
        )
        return True, version, False

    codes = []

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    created_at = parse_timestamp(
        evaluation.get("createdAt")
    )

    if created_at is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created_at > as_of:
            codes.append("FUTURE_EVALUATION")
        else:
            age = timestamp_age_seconds(
                as_of,
                created_at,
            )

            if age > policy["maxAgeSeconds"]:
                codes.append("STALE_EVALUATION")

    # ------------------------------------------------------------
    # Artifact lineage
    # ------------------------------------------------------------

    registered_artifact = version_obj.get(
        "artifactDigest"
    )

    evaluated_artifact = evaluation.get(
        "artifactDigest"
    )

    if (
        not isinstance(registered_artifact, str)
        or not registered_artifact
        or evaluated_artifact != registered_artifact
    ):
        codes.append("ARTIFACT_MISMATCH")

    # ------------------------------------------------------------
    # Dataset lineage
    # ------------------------------------------------------------

    if (
        evaluation.get("datasetDigest")
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # ------------------------------------------------------------
    # Schema lineage
    # ------------------------------------------------------------

    if (
        evaluation.get("schemaDigest")
        != policy["schemaDigest"]
    ):
        codes.append("SCHEMA_MISMATCH")

    # ------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    accuracy_finite = is_finite_number(accuracy)
    latency_finite = is_finite_number(latency)
    size_finite = is_finite_number(size)

    if (
        not accuracy_finite
        or not latency_finite
        or not size_finite
    ):
        codes.append("NON_FINITE")

    # ------------------------------------------------------------
    # Metric ranges
    # ------------------------------------------------------------

    if accuracy_finite:
        if not (0 <= float(accuracy) <= 1):
            codes.append("METRIC_RANGE")

    if latency_finite:
        if float(latency) < 0:
            codes.append("METRIC_RANGE")

    size_is_valid_integer = (
        isinstance(size, int)
        and not isinstance(size, bool)
        and 0 <= size <= SAFE_INT_MAX
    )

    if size_finite and not size_is_valid_integer:
        codes.append("METRIC_RANGE")

    # ------------------------------------------------------------
    # Aggregate gates
    # ------------------------------------------------------------

    if is_unit_interval(accuracy):
        if (
            float(accuracy)
            < float(policy["accuracyFloor"])
        ):
            codes.append("ACCURACY_FLOOR")

    if (
        latency_finite
        and float(latency) >= 0
    ):
        if (
            float(latency)
            > float(policy["maxLatencyMs"])
        ):
            codes.append("LATENCY_LIMIT")

    if size_is_valid_integer:
        if size > policy["maxSizeBytes"]:
            codes.append("SIZE_LIMIT")

    # ------------------------------------------------------------
    # Required slices
    # ------------------------------------------------------------

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        for name in policy["requiredSlices"]:
            codes.append(
                f"MISSING_SLICE:{name}"
            )
    else:
        for name, floor in policy["requiredSlices"].items():

            if name not in slices:
                codes.append(
                    f"MISSING_SLICE:{name}"
                )
                continue

            value = slices[name]

            if not is_finite_number(value):
                codes.append("NON_FINITE")
                continue

            if not (0 <= float(value) <= 1):
                codes.append(
                    f"SLICE_RANGE:{name}"
                )
                continue

            if float(value) < float(floor):
                codes.append(
                    f"SLICE_FLOOR:{name}"
                )

    # Unique and deterministic ordering.
    codes = sorted(set(codes))

    for code in codes:
        add_gate(
            failures,
            version,
            code,
        )

    return (
        True,
        version,
        len(codes) == 0,
    )


def normalize_failures(failures: dict):
    result = {}

    for version, codes in failures.items():
        result[version] = sorted(set(codes))

    def version_sort_key(item):
        version = item[0]

        # Valid canonical numeric versions first,
        # sorted numerically.
        if validate_version_id(version):
            return (
                0,
                int(version),
            )

        # Invalid string version IDs must still be returned
        # deterministically without attempting int(version).
        if isinstance(version, str):
            return (
                1,
                version.encode("utf-8"),
            )

        return (
            2,
            str(version).encode("utf-8"),
        )

    return dict(
        sorted(
            result.items(),
            key=version_sort_key,
        )
    )


def make_response(
    action: str,
    champion_version: str,
    selected_version,
    eligible_versions,
    failures,
    alias_mutation,
    evidence,
):
    return {
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected_version,
        "eligibleVersions": eligible_versions,
        "failedGates": normalize_failures(
            failures
        ),
        "aliasMutation": alias_mutation,
        "evidence": evidence,
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/promote")
async def promote(request: Request):

    # ------------------------------------------------------------
    # Parse JSON explicitly.
    #
    # This ensures malformed JSON does not become an unhandled
    # FastAPI exception/500.
    # ------------------------------------------------------------

    try:
        payload = await request.json()
    except Exception:
        return invalid_input()

    # ------------------------------------------------------------
    # Top-level request validation
    # ------------------------------------------------------------

    if not isinstance(payload, dict):
        return invalid_input()

    if "policy" not in payload:
        return invalid_input()

    if "versions" not in payload:
        return invalid_input()

    if not isinstance(payload["versions"], list):
        return invalid_input()

    if "championVersion" not in payload:
        return invalid_input()

    if not isinstance(
        payload["championVersion"],
        str,
    ):
        return invalid_input()

    if "asOf" not in payload:
        return invalid_input()

    policy = payload["policy"]

    if not validate_policy(policy):
        return invalid_input()

    as_of = parse_timestamp(
        payload["asOf"]
    )

    if as_of is None:
        return invalid_input()

    champion_version = payload[
        "championVersion"
    ]

    versions = payload["versions"]

    failures = {}

    # ------------------------------------------------------------
    # First pass.
    #
    # Reject invalid/noncanonical IDs and duplicate occurrences
    # before constructing the lookup map.
    # ------------------------------------------------------------

    seen = set()
    normalized_versions = []

    for item in versions:

        if not isinstance(item, dict):
            continue

        version = item.get("version")

        if not validate_version_id(version):

            if isinstance(version, str):
                add_gate(
                    failures,
                    version,
                    "INVALID_VERSION",
                )

            continue

        if version in seen:
            add_gate(
                failures,
                version,
                "DUPLICATE_VERSION",
            )
            continue

        seen.add(version)
        normalized_versions.append(item)

    # ------------------------------------------------------------
    # Champion must itself be canonical.
    # ------------------------------------------------------------

    if not validate_version_id(
        champion_version
    ):
        return make_response(
            "block",
            champion_version,
            None,
            [],
            failures,
            None,
            None,
        )

    # ------------------------------------------------------------
    # Construct lookup map only after duplicate and invalid
    # occurrences have been rejected.
    # ------------------------------------------------------------

    version_map = {
        item["version"]: item
        for item in normalized_versions
    }

    # Champion must identify one listed version.
    if champion_version not in version_map:
        return make_response(
            "block",
            champion_version,
            None,
            [],
            failures,
            None,
            None,
        )

    # ------------------------------------------------------------
    # Evaluate every valid unique version.
    # ------------------------------------------------------------

    eligible = []
    eligible_set = set()

    for item in normalized_versions:

        valid_id, version, is_eligible = (
            evaluate_version(
                item,
                as_of,
                policy,
                failures,
            )
        )

        if valid_id and is_eligible:
            eligible.append(item)
            eligible_set.add(version)

    # ------------------------------------------------------------
    # Champion evidence must be valid.
    # ------------------------------------------------------------

    if champion_version not in eligible_set:

        eligible_versions = sorted(
            eligible_set,
            key=lambda value: int(value),
        )

        return make_response(
            "block",
            champion_version,
            None,
            eligible_versions,
            failures,
            None,
            None,
        )

    # ------------------------------------------------------------
    # Deterministic ranking:
    #
    # 1. accuracy descending
    # 2. latency ascending
    # 3. size ascending
    # 4. numeric version ascending
    # ------------------------------------------------------------

    def ranking_key(item):
        evaluation = item["evaluation"]

        return (
            -float(evaluation["accuracy"]),
            float(evaluation["latencyMs"]),
            int(evaluation["sizeBytes"]),
            int(item["version"]),
        )

    eligible.sort(
        key=ranking_key
    )

    eligible_versions = [
        item["version"]
        for item in eligible
    ]

    champion = version_map[
        champion_version
    ]

    champion_evaluation = champion[
        "evaluation"
    ]

    # ------------------------------------------------------------
    # Find the highest-ranked eligible challenger.
    # ------------------------------------------------------------

    challengers = [
        item
        for item in eligible
        if item["version"]
        != champion_version
    ]

    # No challenger -> retain champion.
    if not challengers:
        return make_response(
            "retain",
            champion_version,
            champion_version,
            eligible_versions,
            failures,
            None,
            champion_evaluation,
        )

    challenger = challengers[0]

    challenger_evaluation = challenger[
        "evaluation"
    ]

    # ------------------------------------------------------------
    # Improvement is rounded to 12 decimal places BEFORE
    # comparing with minImprovement.
    # ------------------------------------------------------------

    improvement = round(
        float(
            challenger_evaluation["accuracy"]
        )
        - float(
            champion_evaluation["accuracy"]
        ),
        12,
    )

    if improvement >= float(
        policy["minImprovement"]
    ):
        return make_response(
            "promote",
            champion_version,
            challenger["version"],
            eligible_versions,
            failures,
            {
                "alias": "champion",
                "version": challenger[
                    "version"
                ],
            },
            challenger_evaluation,
        )

    # Improvement insufficient -> retain champion.
    return make_response(
        "retain",
        champion_version,
        champion_version,
        eligible_versions,
        failures,
        None,
        champion_evaluation,
    )
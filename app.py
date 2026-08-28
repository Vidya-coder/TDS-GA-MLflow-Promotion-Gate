from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from typing import Any
from datetime import datetime, timedelta, timezone
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
    return (
        is_finite_number(value)
        and 0 <= float(value) <= 1
    )


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
    except ValueError:
        return False

    return 1 <= number <= SAFE_INT_MAX


def add_failure(
    failures: dict[str, list[str]],
    version: str,
    code: str,
) -> None:
    failures.setdefault(version, [])

    if code not in failures[version]:
        failures[version].append(code)


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False

    if not REQUIRED_POLICY_FIELDS.issubset(policy.keys()):
        return False

    dataset_digest = policy["datasetDigest"]
    schema_digest = policy["schemaDigest"]

    if not isinstance(dataset_digest, str) or not dataset_digest:
        return False

    if not isinstance(schema_digest, str) or not schema_digest:
        return False

    if not is_safe_nonnegative_int(policy["maxAgeSeconds"]):
        return False

    if not is_unit_interval(policy["accuracyFloor"]):
        return False

    required_slices = policy["requiredSlices"]

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():
        if not isinstance(name, str):
            return False

        if not is_unit_interval(floor):
            return False

    if not is_finite_number(policy["maxLatencyMs"]):
        return False

    if float(policy["maxLatencyMs"]) < 0:
        return False

    if not is_safe_nonnegative_int(policy["maxSizeBytes"]):
        return False

    if not is_unit_interval(policy["minImprovement"]):
        return False

    return True


def evaluate_version(
    version_obj: dict,
    as_of: datetime,
    policy: dict,
    failures: dict[str, list[str]],
) -> bool:
    """
    Evaluate one unique, canonical version.

    Returns True only when every gate passes.
    """

    version = version_obj["version"]

    evaluation = version_obj.get("evaluation")

    if evaluation is None or not isinstance(evaluation, dict):
        add_failure(
            failures,
            version,
            "MISSING_EVALUATION",
        )
        return False

    codes: list[str] = []

    # ---------------------------------------------------------
    # Timestamp / evidence age
    # ---------------------------------------------------------

    created_at = parse_timestamp(
        evaluation.get("createdAt")
    )

    if created_at is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created_at > as_of:
            codes.append("FUTURE_EVALUATION")
        elif created_at < (
            as_of - timedelta(
                seconds=policy["maxAgeSeconds"]
            )
        ):
            codes.append("STALE_EVALUATION")

    # ---------------------------------------------------------
    # Artifact lineage
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Dataset lineage
    # ---------------------------------------------------------

    if (
        evaluation.get("datasetDigest")
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # ---------------------------------------------------------
    # Schema lineage
    # ---------------------------------------------------------

    if (
        evaluation.get("schemaDigest")
        != policy["schemaDigest"]
    ):
        codes.append("SCHEMA_MISMATCH")

    # ---------------------------------------------------------
    # Numeric metrics
    # ---------------------------------------------------------

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    metrics_finite = (
        is_finite_number(accuracy)
        and is_finite_number(latency)
        and is_finite_number(size)
    )

    if not metrics_finite:
        codes.append("NON_FINITE")

    # ---------------------------------------------------------
    # Metric ranges
    # ---------------------------------------------------------

    if is_finite_number(accuracy):
        if not 0 <= float(accuracy) <= 1:
            codes.append("METRIC_RANGE")

    if is_finite_number(latency):
        if float(latency) < 0:
            codes.append("METRIC_RANGE")

    if is_finite_number(size):
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > SAFE_INT_MAX
        ):
            codes.append("METRIC_RANGE")

    # ---------------------------------------------------------
    # Aggregate gates
    # ---------------------------------------------------------

    if is_unit_interval(accuracy):
        if float(accuracy) < float(
            policy["accuracyFloor"]
        ):
            codes.append("ACCURACY_FLOOR")

    if is_finite_number(latency):
        if float(latency) >= 0:
            if float(latency) > float(
                policy["maxLatencyMs"]
            ):
                codes.append("LATENCY_LIMIT")

    if (
        isinstance(size, int)
        and not isinstance(size, bool)
        and 0 <= size <= SAFE_INT_MAX
    ):
        if size > policy["maxSizeBytes"]:
            codes.append("SIZE_LIMIT")

    # ---------------------------------------------------------
    # Required slices
    # ---------------------------------------------------------

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        for name in policy["requiredSlices"]:
            codes.append(f"MISSING_SLICE:{name}")
    else:
        for name, floor in policy["requiredSlices"].items():
            if name not in slices:
                codes.append(f"MISSING_SLICE:{name}")
                continue

            value = slices[name]

            if not is_finite_number(value):
                codes.append("NON_FINITE")
                continue

            if not 0 <= float(value) <= 1:
                codes.append(f"SLICE_RANGE:{name}")
                continue

            if float(value) < float(floor):
                codes.append(f"SLICE_FLOOR:{name}")

    # ---------------------------------------------------------
    # Canonical gate-code output
    # ---------------------------------------------------------

    for code in sorted(set(codes)):
        add_failure(
            failures,
            version,
            code,
        )

    return len(codes) == 0


def normalize_failures(
    failures: dict[str, list[str]]
) -> dict[str, list[str]]:
    normalized = {}

    for version in sorted(
        failures.keys(),
        key=lambda value: int(value),
    ):
        normalized[version] = sorted(
            set(failures[version])
        )

    return normalized


def rank_key(version_obj: dict):
    evaluation = version_obj["evaluation"]

    return (
        -float(evaluation["accuracy"]),
        float(evaluation["latencyMs"]),
        int(evaluation["sizeBytes"]),
        int(version_obj["version"]),
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/promote")
async def promote(
    payload: Any = Body(...)
):
    # =========================================================
    # Request validation
    # =========================================================

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

    if not validate_policy(payload["policy"]):
        return invalid_input()

    as_of = parse_timestamp(payload["asOf"])

    if as_of is None:
        return invalid_input()

    policy = payload["policy"]
    champion_version = payload["championVersion"]
    versions = payload["versions"]

    failures: dict[str, list[str]] = {}

    # =========================================================
    # Pass 1:
    # validate IDs and reject duplicates BEFORE lookup map
    # =========================================================

    seen: set[str] = set()
    valid_versions: list[dict] = []

    for item in versions:
        if not isinstance(item, dict):
            # No canonical version ID exists, therefore there
            # is no version-specific gate to attach.
            continue

        version = item.get("version")

        if not validate_version_id(version):
            if isinstance(version, str):
                add_failure(
                    failures,
                    version,
                    "INVALID_VERSION",
                )
            continue

        if version in seen:
            add_failure(
                failures,
                version,
                "DUPLICATE_VERSION",
            )
            continue

        seen.add(version)
        valid_versions.append(item)

    # =========================================================
    # Champion must itself be canonical
    # =========================================================

    if not validate_version_id(champion_version):
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": normalize_failures(failures),
            "aliasMutation": None,
            "evidence": None,
        }

    # =========================================================
    # Construct lookup map ONLY after duplicate validation
    # =========================================================

    version_map = {
        item["version"]: item
        for item in valid_versions
    }

    # Champion must be one of the listed versions.
    if champion_version not in version_map:
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": normalize_failures(failures),
            "aliasMutation": None,
            "evidence": None,
        }

    # =========================================================
    # Evaluate all valid unique versions
    # =========================================================

    eligible: list[dict] = []

    for item in valid_versions:
        if evaluate_version(
            item,
            as_of,
            policy,
            failures,
        ):
            eligible.append(item)

    # =========================================================
    # Champion evidence must be valid
    # =========================================================

    champion = version_map[champion_version]

    champion_is_eligible = any(
        item["version"] == champion_version
        for item in eligible
    )

    eligible.sort(key=rank_key)

    eligible_versions = [
        item["version"]
        for item in eligible
    ]

    if not champion_is_eligible:
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": normalize_failures(failures),
            "aliasMutation": None,
            "evidence": None,
        }

    champion_evaluation = champion["evaluation"]

    # =========================================================
    # Find best eligible challenger
    # =========================================================

    challengers = [
        item
        for item in eligible
        if item["version"] != champion_version
    ]

    # No challenger -> retain current champion.
    if not challengers:
        return {
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": eligible_versions,
            "failedGates": normalize_failures(failures),
            "aliasMutation": None,
            "evidence": champion_evaluation,
        }

    challenger = challengers[0]
    challenger_evaluation = challenger["evaluation"]

    # =========================================================
    # Improvement calculation
    # =========================================================

    improvement = round(
        float(challenger_evaluation["accuracy"])
        - float(champion_evaluation["accuracy"]),
        12,
    )

    # =========================================================
    # Promotion
    # =========================================================

    if improvement >= float(
        policy["minImprovement"]
    ):
        return {
            "action": "promote",
            "championVersion": champion_version,
            "selectedVersion": challenger["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": normalize_failures(failures),
            "aliasMutation": {
                "alias": "champion",
                "version": challenger["version"],
            },
            "evidence": challenger_evaluation,
        }

    # =========================================================
    # Retain
    # =========================================================

    return {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": eligible_versions,
        "failedGates": normalize_failures(failures),
        "aliasMutation": None,
        "evidence": champion_evaluation,
    }

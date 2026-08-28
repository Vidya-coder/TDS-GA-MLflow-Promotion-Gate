from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from typing import Any
from datetime import datetime, timezone
import json
import math
import re


app = FastAPI()


# JavaScript Number.MAX_SAFE_INTEGER
SAFE_INT_MAX = 9007199254740991


# Canonical positive safe-integer version:
# "1", "2", "10", ...
# Never "01", "+1", "1.0", etc.
VERSION_RE = re.compile(r"^[1-9][0-9]*$")


# Exact policy fields required by the specification.
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


# -------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------

def invalid_input():
    """
    Every malformed top-level request must return exactly:

        HTTP 400
        {"error":"INVALID_INPUT"}
    """
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


def make_response(
    action: str,
    champion_version: str,
    selected_version: Any,
    eligible_versions: list[str],
    failures: dict[str, list[str]],
    alias_mutation: Any,
    evidence: Any,
):
    """
    Construct the response in the exact required shape.
    """

    return {
        "action": action,
        "championVersion": champion_version,
        "selectedVersion": selected_version,
        "eligibleVersions": eligible_versions,
        "failedGates": normalize_failures(failures),
        "aliasMutation": alias_mutation,
        "evidence": evidence,
    }


def is_safe_nonnegative_int(value: Any) -> bool:
    """
    Non-negative safe integer.

    bool is deliberately rejected because bool is a subclass of int
    in Python.
    """

    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def is_finite_number(value: Any) -> bool:
    """
    Accept finite int/float values, but reject bool and NaN/Infinity.
    """

    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError, OverflowError):
        return False


def is_unit_interval(value: Any) -> bool:
    """
    Finite number in [0, 1].
    """

    if not is_finite_number(value):
        return False

    value = float(value)

    return 0.0 <= value <= 1.0


# -------------------------------------------------------------------
# Timestamp handling
# -------------------------------------------------------------------

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def parse_timestamp(value: Any):
    """
    Accept exactly:

        YYYY-MM-DDTHH:mm:ss
        YYYY-MM-DDTHH:mm:ss.s
        YYYY-MM-DDTHH:mm:ss.ss
        YYYY-MM-DDTHH:mm:ss.sss

    followed by:

        Z
        +HH:mm
        -HH:mm

    Return UTC-aware datetime or None.
    """

    if not isinstance(value, str):
        return None

    if not TIMESTAMP_RE.fullmatch(value):
        return None

    normalized = value

    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(normalized)
    except (ValueError, OverflowError):
        return None

    if dt.tzinfo is None:
        return None

    return dt.astimezone(timezone.utc)


# -------------------------------------------------------------------
# Version handling
# -------------------------------------------------------------------

def validate_version_id(value: Any) -> bool:
    """
    Version IDs must be canonical positive safe-integer strings.
    """

    if not isinstance(value, str):
        return False

    if not VERSION_RE.fullmatch(value):
        return False

    try:
        number = int(value)
    except (ValueError, OverflowError):
        return False

    return 1 <= number <= SAFE_INT_MAX


# -------------------------------------------------------------------
# Failure handling
# -------------------------------------------------------------------

def add_gate(
    failures: dict[str, list[str]],
    version: str,
    code: str,
):
    """
    Add a gate code without duplication.
    """

    failures.setdefault(version, [])

    if code not in failures[version]:
        failures[version].append(code)


def normalize_failures(
    failures: dict[str, list[str]]
):
    """
    Sort gate codes and version IDs deterministically.

    Version IDs are numeric strings, so sort numerically.
    """

    result = {}

    for version, codes in failures.items():
        result[version] = sorted(set(codes))

    return dict(
        sorted(
            result.items(),
            key=lambda item: (
                int(item[0])
                if validate_version_id(item[0])
                else item[0]
            ),
        )
    )


# -------------------------------------------------------------------
# Policy validation
# -------------------------------------------------------------------

def validate_policy(policy: Any) -> bool:
    """
    Validate the supplied policy as a whole.

    Important:
    - Missing policy is INVALID_INPUT.
    - Present but invalid policy becomes INVALID_POLICY gate(s).
    """

    if not isinstance(policy, dict):
        return False

    if not REQUIRED_POLICY_FIELDS.issubset(policy.keys()):
        return False

    # datasetDigest
    dataset_digest = policy.get("datasetDigest")

    if (
        not isinstance(dataset_digest, str)
        or dataset_digest == ""
    ):
        return False

    # schemaDigest
    schema_digest = policy.get("schemaDigest")

    if (
        not isinstance(schema_digest, str)
        or schema_digest == ""
    ):
        return False

    # maxAgeSeconds
    if not is_safe_nonnegative_int(
        policy.get("maxAgeSeconds")
    ):
        return False

    # accuracyFloor
    if not is_unit_interval(
        policy.get("accuracyFloor")
    ):
        return False

    # requiredSlices
    required_slices = policy.get("requiredSlices")

    if not isinstance(required_slices, dict):
        return False

    for name, floor in required_slices.items():

        if not isinstance(name, str):
            return False

        if not is_unit_interval(floor):
            return False

    # maxLatencyMs
    max_latency = policy.get("maxLatencyMs")

    if not is_finite_number(max_latency):
        return False

    if float(max_latency) < 0:
        return False

    # maxSizeBytes
    if not is_safe_nonnegative_int(
        policy.get("maxSizeBytes")
    ):
        return False

    # minImprovement
    if not is_unit_interval(
        policy.get("minImprovement")
    ):
        return False

    return True


# -------------------------------------------------------------------
# Version evaluation
# -------------------------------------------------------------------

def evaluate_version(
    version_obj: Any,
    as_of: datetime,
    policy: dict,
    failures: dict[str, list[str]],
):
    """
    Evaluate one canonical, unique version.

    Returns:

        (valid_id, version_id, eligible)

    Once a version has been structurally identified, every applicable
    gate is collected rather than stopping at the first failure.
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

    # Missing or non-object evaluation is specifically
    # MISSING_EVALUATION.
    if evaluation is None or not isinstance(evaluation, dict):

        add_gate(
            failures,
            version,
            "MISSING_EVALUATION",
        )

        return True, version, False

    codes = []

    # ---------------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------------

    created_at_raw = evaluation.get("createdAt")
    created_at = parse_timestamp(created_at_raw)

    if created_at is None:

        codes.append("INVALID_TIMESTAMP")

    else:

        if created_at > as_of:
            codes.append("FUTURE_EVALUATION")

        else:
            age_seconds = (
                as_of - created_at
            ).total_seconds()

            if age_seconds > float(
                policy["maxAgeSeconds"]
            ):
                codes.append("STALE_EVALUATION")

    # ---------------------------------------------------------------
    # Artifact lineage
    # ---------------------------------------------------------------

    registered_artifact = version_obj.get(
        "artifactDigest"
    )

    evaluated_artifact = evaluation.get(
        "artifactDigest"
    )

    if (
        not isinstance(
            registered_artifact,
            str,
        )
        or not registered_artifact
        or evaluated_artifact
        != registered_artifact
    ):
        codes.append("ARTIFACT_MISMATCH")

    # ---------------------------------------------------------------
    # Dataset lineage
    # ---------------------------------------------------------------

    if (
        evaluation.get("datasetDigest")
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # ---------------------------------------------------------------
    # Schema lineage
    # ---------------------------------------------------------------

    if (
        evaluation.get("schemaDigest")
        != policy["schemaDigest"]
    ):
        codes.append("SCHEMA_MISMATCH")

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    accuracy_finite = is_finite_number(
        accuracy
    )

    latency_finite = is_finite_number(
        latency
    )

    size_finite = is_finite_number(
        size
    )

    if (
        not accuracy_finite
        or not latency_finite
        or not size_finite
    ):
        codes.append("NON_FINITE")

    # ---------------------------------------------------------------
    # Metric ranges
    # ---------------------------------------------------------------

    if accuracy_finite:

        accuracy_float = float(accuracy)

        if not (
            0.0
            <= accuracy_float
            <= 1.0
        ):
            codes.append("METRIC_RANGE")

    if latency_finite:

        latency_float = float(latency)

        if latency_float < 0:
            codes.append("METRIC_RANGE")

    # sizeBytes must be a non-negative safe integer.
    if size_finite:

        if not is_safe_nonnegative_int(size):
            codes.append("METRIC_RANGE")

    # ---------------------------------------------------------------
    # Aggregate accuracy gate
    # ---------------------------------------------------------------

    if is_unit_interval(accuracy):

        if (
            float(accuracy)
            < float(
                policy["accuracyFloor"]
            )
        ):
            codes.append(
                "ACCURACY_FLOOR"
            )

    # ---------------------------------------------------------------
    # Aggregate latency gate
    # ---------------------------------------------------------------

    if (
        is_finite_number(latency)
        and float(latency) >= 0
    ):

        if (
            float(latency)
            > float(
                policy["maxLatencyMs"]
            )
        ):
            codes.append(
                "LATENCY_LIMIT"
            )

    # ---------------------------------------------------------------
    # Aggregate size gate
    # ---------------------------------------------------------------

    if is_safe_nonnegative_int(size):

        if (
            size
            > policy["maxSizeBytes"]
        ):
            codes.append(
                "SIZE_LIMIT"
            )

    # ---------------------------------------------------------------
    # Required slices
    # ---------------------------------------------------------------

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):

        for name in policy[
            "requiredSlices"
        ]:

            codes.append(
                f"MISSING_SLICE:{name}"
            )

    else:

        for (
            name,
            floor,
        ) in policy[
            "requiredSlices"
        ].items():

            if name not in slices:

                codes.append(
                    f"MISSING_SLICE:{name}"
                )

                continue

            value = slices[name]

            if not is_finite_number(value):

                codes.append(
                    "NON_FINITE"
                )

                continue

            value_float = float(value)

            if not (
                0.0
                <= value_float
                <= 1.0
            ):

                codes.append(
                    f"SLICE_RANGE:{name}"
                )

                continue

            if (
                value_float
                < float(floor)
            ):

                codes.append(
                    f"SLICE_FLOOR:{name}"
                )

    # ---------------------------------------------------------------
    # Deterministic code ordering
    # ---------------------------------------------------------------

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


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------------------------------------------------
# Promotion endpoint
# -------------------------------------------------------------------

@app.post("/promote")
async def promote(request: Request):

    # ---------------------------------------------------------------
    # Parse JSON body ourselves so malformed JSON returns exactly
    # the required INVALID_INPUT response rather than FastAPI's
    # default validation structure.
    # ---------------------------------------------------------------

    try:
        raw_body = await request.body()

        payload = json.loads(
            raw_body.decode("utf-8")
        )

    except Exception:
        return invalid_input()

    # ---------------------------------------------------------------
    # Top-level request validation
    # ---------------------------------------------------------------

    if not isinstance(payload, dict):
        return invalid_input()

    # Missing policy -> INVALID_INPUT.
    if "policy" not in payload:
        return invalid_input()

    # versions must exist and be an array.
    if "versions" not in payload:
        return invalid_input()

    if not isinstance(
        payload["versions"],
        list,
    ):
        return invalid_input()

    # championVersion must exist and be a string.
    if "championVersion" not in payload:
        return invalid_input()

    if not isinstance(
        payload["championVersion"],
        str,
    ):
        return invalid_input()

    # asOf is required.
    if "asOf" not in payload:
        return invalid_input()

    champion_version = (
        payload["championVersion"]
    )

    versions = payload["versions"]

    policy = payload["policy"]

    # asOf itself is request-level validation.
    as_of = parse_timestamp(
        payload["asOf"]
    )

    if as_of is None:
        return invalid_input()

    failures = {}

    # ---------------------------------------------------------------
    # Policy validation
    #
    # IMPORTANT:
    #
    # Missing policy => 400.
    #
    # Invalid supplied policy => INVALID_POLICY gate.
    # ---------------------------------------------------------------

    policy_valid = validate_policy(
        policy
    )

    # ---------------------------------------------------------------
    # First structural pass over versions.
    #
    # Invalid and duplicate versions are rejected BEFORE creating
    # the lookup map.
    # ---------------------------------------------------------------

    seen = set()

    normalized_versions = []

    for item in versions:

        if not isinstance(item, dict):
            continue

        version = item.get("version")

        # Non-string / noncanonical version.
        if not validate_version_id(
            version
        ):

            if isinstance(
                version,
                str,
            ):
                add_gate(
                    failures,
                    version,
                    "INVALID_VERSION",
                )

            continue

        # Duplicate occurrence.
        if version in seen:

            add_gate(
                failures,
                version,
                "DUPLICATE_VERSION",
            )

            continue

        seen.add(version)

        normalized_versions.append(
            item
        )

    # ---------------------------------------------------------------
    # If policy itself is invalid, it is a gate failure on every
    # valid unique input version.
    #
    # Structural INVALID_VERSION / DUPLICATE_VERSION failures are
    # retained as well.
    # ---------------------------------------------------------------

    if not policy_valid:

        for item in normalized_versions:

            version = item["version"]

            add_gate(
                failures,
                version,
                "INVALID_POLICY",
            )

        return make_response(
            action="block",
            champion_version=champion_version,
            selected_version=None,
            eligible_versions=[],
            failures=failures,
            alias_mutation=None,
            evidence=None,
        )

    # ---------------------------------------------------------------
    # Champion version itself must be canonical.
    # ---------------------------------------------------------------

    if not validate_version_id(
        champion_version
    ):

        return make_response(
            action="block",
            champion_version=champion_version,
            selected_version=None,
            eligible_versions=[],
            failures=failures,
            alias_mutation=None,
            evidence=None,
        )

    # ---------------------------------------------------------------
    # Construct lookup map ONLY after invalid and duplicate versions
    # have been rejected.
    # ---------------------------------------------------------------

    version_map = {
        item["version"]: item
        for item in normalized_versions
    }

    # Champion must identify a listed version.
    if champion_version not in version_map:

        return make_response(
            action="block",
            champion_version=champion_version,
            selected_version=None,
            eligible_versions=[],
            failures=failures,
            alias_mutation=None,
            evidence=None,
        )

    # ---------------------------------------------------------------
    # Evaluate all valid unique versions.
    # ---------------------------------------------------------------

    eligible = []

    eligible_set = set()

    for item in normalized_versions:

        (
            valid_id,
            version,
            is_eligible,
        ) = evaluate_version(
            item,
            as_of,
            policy,
            failures,
        )

        if (
            valid_id
            and is_eligible
        ):

            eligible.append(item)

            eligible_set.add(
                version
            )

    # ---------------------------------------------------------------
    # Champion evidence must itself be valid.
    # ---------------------------------------------------------------

    if (
        champion_version
        not in eligible_set
    ):

        eligible_versions = sorted(
            eligible_set,
            key=lambda value: int(value),
        )

        return make_response(
            action="block",
            champion_version=champion_version,
            selected_version=None,
            eligible_versions=eligible_versions,
            failures=failures,
            alias_mutation=None,
            evidence=None,
        )

    # ---------------------------------------------------------------
    # Deterministic ranking:
    #
    # 1. accuracy descending
    # 2. latency ascending
    # 3. size ascending
    # 4. numeric version ascending
    # ---------------------------------------------------------------

    def ranking_key(item):

        evaluation = item[
            "evaluation"
        ]

        return (
            -float(
                evaluation["accuracy"]
            ),
            float(
                evaluation["latencyMs"]
            ),
            int(
                evaluation["sizeBytes"]
            ),
            int(
                item["version"]
            ),
        )

    eligible.sort(
        key=ranking_key
    )

    eligible_versions = [
        item["version"]
        for item in eligible
    ]

    # ---------------------------------------------------------------
    # Current champion evidence.
    # ---------------------------------------------------------------

    champion = version_map[
        champion_version
    ]

    champion_evaluation = champion[
        "evaluation"
    ]

    # ---------------------------------------------------------------
    # Find the best eligible challenger.
    # ---------------------------------------------------------------

    challengers = [
        item
        for item in eligible
        if item["version"]
        != champion_version
    ]

    # No challenger -> retain champion.
    if not challengers:

        return make_response(
            action="retain",
            champion_version=champion_version,
            selected_version=champion_version,
            eligible_versions=eligible_versions,
            failures=failures,
            alias_mutation=None,
            evidence=champion_evaluation,
        )

    challenger = challengers[0]

    challenger_evaluation = challenger[
        "evaluation"
    ]

    # ---------------------------------------------------------------
    # Improvement:
    #
    # challenger accuracy - champion accuracy
    #
    # Round to 12 decimal places BEFORE comparison.
    # ---------------------------------------------------------------

    improvement = round(
        float(
            challenger_evaluation[
                "accuracy"
            ]
        )
        - float(
            champion_evaluation[
                "accuracy"
            ]
        ),
        12,
    )

    # ---------------------------------------------------------------
    # Promote if improvement meets policy.
    # ---------------------------------------------------------------

    if (
        improvement
        >= float(
            policy["minImprovement"]
        )
    ):

        return make_response(
            action="promote",
            champion_version=champion_version,
            selected_version=challenger[
                "version"
            ],
            eligible_versions=eligible_versions,
            failures=failures,
            alias_mutation={
                "alias": "champion",
                "version": challenger[
                    "version"
                ],
            },
            evidence=challenger_evaluation,
        )

    # ---------------------------------------------------------------
    # Otherwise retain current champion.
    # ---------------------------------------------------------------

    return make_response(
        action="retain",
        champion_version=champion_version,
        selected_version=champion_version,
        eligible_versions=eligible_versions,
        failures=failures,
        alias_mutation=None,
        evidence=champion_evaluation,
    )
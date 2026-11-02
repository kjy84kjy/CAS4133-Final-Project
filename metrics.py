# metrics.py

import re


def extract_final_answer(response):
    """
    Prefer explicit format:
    Final Answer: 123

    Fallback:
    Use the last number in the response.
    """
    if response is None:
        return None

    response = str(response)

    match = re.search(
        r"Final Answer:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).replace(",", "")

    nums = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        response,
    )

    if nums:
        return nums[-1].replace(",", "")

    return None


def has_valid_format(response):
    """
    Valid if the response contains explicit Final Answer format.
    """
    if response is None:
        return False

    response = str(response)

    return bool(
        re.search(
            r"Final Answer:\s*-?\d+(?:,\d{3})*(?:\.\d+)?",
            response,
            flags=re.IGNORECASE,
        )
    )


def normalize_number(x):
    if x is None:
        return None

    try:
        value = float(str(x).replace(",", "").strip())

        if value.is_integer():
            return str(int(value))

        return str(value)

    except Exception:
        return str(x).strip()


def is_correct(pred, gold):
    return normalize_number(pred) == normalize_number(gold)


def compute_metrics(results):
    total = len(results)

    if total == 0:
        return {
            "accuracy": 0.0,
            "format_error_rate": 0.0,
            "avg_tokens": 0.0,
        }

    correct = 0
    format_errors = 0
    total_tokens_proxy = 0

    for r in results:
        if r["correct"]:
            correct += 1

        if not r["valid_format"]:
            format_errors += 1

        total_tokens_proxy += len(str(r["response"]).split())

    accuracy = correct / total
    format_error_rate = format_errors / total
    avg_tokens = total_tokens_proxy / total

    return {
        "accuracy": accuracy,
        "format_error_rate": format_error_rate,
        "avg_tokens": avg_tokens,
    }


def compute_final_score(
    metrics,
    lambda_format=0.2,
    lambda_token=0.1,
    token_baseline=300,
    mode="multi_objective",
):
    """
    mode="accuracy_only":
        vanilla OPRO baseline

    mode="multi_objective":
        ours
    """
    if mode == "accuracy_only":
        return metrics["accuracy"]

    normalized_token_cost = metrics["avg_tokens"] / token_baseline

    return (
        metrics["accuracy"]
        - lambda_format * metrics["format_error_rate"]
        - lambda_token * normalized_token_cost
    )
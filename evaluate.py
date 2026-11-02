# evaluate.py

from utils import build_solver_prompt
from metrics import (
    extract_final_answer,
    has_valid_format,
    is_correct,
    compute_metrics,
    compute_final_score,
)


def evaluate_prompt(
    generator,
    instruction,
    dataset,
    max_new_tokens=160,
    temperature=0.0,
    lambda_format=0.2,
    lambda_token=0.1,
    token_baseline=300,
    score_mode="multi_objective",
    batch_size=8,
    desc=None,
):
    prompts = [
        build_solver_prompt(instruction, ex["question"])
        for ex in dataset
    ]

    responses = generator.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        show_progress=True,
        desc=desc or "Evaluating prompt",
        system_prompt=(
            "You are a precise math problem solver. "
            "Follow the required final answer format exactly."
        ),
        use_chat_template=True,
    )

    results = []

    for ex, response in zip(dataset, responses):
        pred = extract_final_answer(response)
        correct = is_correct(pred, ex["gold"])
        valid_format = has_valid_format(response)

        results.append({
            "question": ex["question"],
            "gold": ex["gold"],
            "response": response,
            "pred": pred,
            "correct": correct,
            "valid_format": valid_format,
        })

    metrics = compute_metrics(results)

    final_score = compute_final_score(
        metrics,
        lambda_format=lambda_format,
        lambda_token=lambda_token,
        token_baseline=token_baseline,
        mode=score_mode,
    )

    return {
        "instruction": instruction,
        "metrics": metrics,
        "score": final_score,
        "results": results,
    }
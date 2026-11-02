# main.py

import os
import csv
import json
from huggingface_hub import login

from config import *
from data import load_gsm8k, load_svamp
from model import HFGenerator
from evaluate import evaluate_prompt
from optimizer import build_optimizer_prompt, parse_candidate_prompts

from getpass import getpass
from huggingface_hub import login


def setup_huggingface_auth():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if token is None:
        print("[Warning] HF_TOKEN is not set. Skipping Hugging Face login.")
        print("[Warning] Gated models such as Llama may not be accessible.")
        return None

    login(token=token)
    print("[Info] Hugging Face login successful.")
    return token


def load_seed_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simplify_eval_result(result, source):
    return {
        "source": source,
        "instruction": result["instruction"],
        "metrics": result["metrics"],
        "score": result["score"],
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_history_csv(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "source",
            "score",
            "accuracy",
            "format_error_rate",
            "avg_tokens",
            "instruction",
        ])

        for h in history:
            writer.writerow([
                h["source"],
                h["score"],
                h["metrics"]["accuracy"],
                h["metrics"]["format_error_rate"],
                h["metrics"]["avg_tokens"],
                h["instruction"],
            ])


def print_result(name, result):
    print("Metrics:", result["metrics"])
    print("Final score:", result["score"])


def main():
    
    # hf_token = getpass("Enter your Hugging Face token: ")
    # login(token=hf_token)

    print("[Info] Loading dataset...")

    if DATASET_NAME.lower() == "gsm8k":
        train_data, test_data = load_gsm8k(
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
        )
    elif DATASET_NAME.lower() == "svamp":
        train_data, test_data = load_svamp(
            train_size=TRAIN_SIZE,
            test_size=TEST_SIZE,
        )
    else:
        raise ValueError(f"Unknown dataset: {DATASET_NAME}")

    print("[Info] Example data:")
    print(train_data[0])

    print("[Info] Loading model...")
    generator = HFGenerator(MODEL_NAME)

    seed_prompts = load_seed_prompts(SEED_PROMPT_PATH)

    history = []

    run_dir = os.path.join(RESULT_DIR, RUN_NAME)

    # ============================================================
    # Stage 1: Evaluate seed prompts
    # ============================================================

    print("=" * 80)
    print("[Stage 1] Evaluating seed prompts")

    for p in seed_prompts:
        print("=" * 80)
        print("Prompt:", p["name"])
        print(p["prompt"])

        result = evaluate_prompt(
            generator=generator,
            instruction=p["prompt"],
            dataset=train_data,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            lambda_format=LAMBDA_FORMAT,
            lambda_token=LAMBDA_TOKEN,
            token_baseline=TOKEN_BASELINE,
            score_mode=SCORE_MODE,
            batch_size=BATCH_SIZE,
            desc=f"Evaluating {p['name']}",
        )

        print_result(p["name"], result)

        history.append(
            simplify_eval_result(
                result,
                source=p["name"],
            )
        )

        save_json(history, f"{run_dir}/history_seed.json")
        save_history_csv(history, f"{run_dir}/history_seed.csv")

    # ============================================================
    # Stage 2: OPRO-style prompt optimization
    # ============================================================

    print("=" * 80)
    print("[Stage 2] OPRO-style prompt optimization")

    for it in range(NUM_ITERATIONS):
        print("=" * 80)
        print(f"Optimization Iteration {it + 1}/{NUM_ITERATIONS}")

        optimizer_input = build_optimizer_prompt(
            history,
            num_candidates=NUM_CANDIDATES_PER_ITER,
        )

        optimizer_response = generator.generate(
            [optimizer_input],
            max_new_tokens=96,
            temperature=0.2,
            batch_size=1,
            show_progress=False,
            desc="Generating candidate prompts",
            system_prompt=(
                "You are a prompt optimizer. "
                "Return only the requested JSON array and nothing else."
            ),
            use_chat_template=True,
        )[0]

        print("[Optimizer raw response]")
        print(optimizer_response)

        candidates = parse_candidate_prompts(
            optimizer_response,
            max_candidates=NUM_CANDIDATES_PER_ITER,
        )

        if len(candidates) == 0:
            print("[Warning] No valid candidate prompts parsed. Skipping this iteration.")
            continue

        print("[Parsed candidate prompts]")
        for idx, candidate in enumerate(candidates, start=1):
            print(f"[Candidate {idx}] {candidate}")

        for idx, candidate in enumerate(candidates, start=1):
            print("-" * 80)
            print(f"Evaluating Iter {it + 1} Candidate {idx}")
            print(candidate)

            result = evaluate_prompt(
                generator=generator,
                instruction=candidate,
                dataset=train_data,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                lambda_format=LAMBDA_FORMAT,
                lambda_token=LAMBDA_TOKEN,
                token_baseline=TOKEN_BASELINE,
                score_mode=SCORE_MODE,
                batch_size=BATCH_SIZE,
                desc=f"Iter {it + 1} Candidate {idx}",
            )

            print_result(f"iter_{it + 1}_candidate_{idx}", result)

            history.append(
                simplify_eval_result(
                    result,
                    source=f"iter_{it + 1}_candidate_{idx}",
                )
            )

            save_json(history, f"{run_dir}/history_iter_{it + 1}.json")
            save_history_csv(history, f"{run_dir}/history_iter_{it + 1}.csv")

    # ============================================================
    # Stage 3: Select best prompt
    # ============================================================

    best = max(history, key=lambda x: x["score"])

    print("=" * 80)
    print("[Stage 3] Best prompt selected")
    print("Source:", best["source"])
    print("Best prompt:")
    print(best["instruction"])
    print("Validation metrics:", best["metrics"])
    print("Validation score:", best["score"])

    # ============================================================
    # Stage 4: Evaluate best prompt on test set
    # ============================================================

    print("=" * 80)
    print("[Stage 4] Evaluating best prompt on test set")

    test_result = evaluate_prompt(
        generator=generator,
        instruction=best["instruction"],
        dataset=test_data,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        lambda_format=LAMBDA_FORMAT,
        lambda_token=LAMBDA_TOKEN,
        token_baseline=TOKEN_BASELINE,
        score_mode=SCORE_MODE,
        batch_size=BATCH_SIZE,
        desc="Evaluating best prompt on test set",
    )

    print("Test metrics:", test_result["metrics"])
    print("Test score:", test_result["score"])

    final_output = {
        "best_prompt": best,
        "test_metrics": test_result["metrics"],
        "test_score": test_result["score"],
        "history": history,
    }

    save_json(final_output, f"{run_dir}/final_result.json")
    save_history_csv(history, f"{run_dir}/history_final.csv")

    print("=" * 80)
    print("[Done] Results saved.")
    print(f"- {run_dir}/final_result.json")
    print(f"- {run_dir}/history_final.csv")


if __name__ == "__main__":
    main()
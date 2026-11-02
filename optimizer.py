# optimizer.py

import json
import re


def build_optimizer_prompt(history, num_candidates=2):
    sorted_history = sorted(history, key=lambda x: x["score"], reverse=True)

    history_text = ""

    for i, item in enumerate(sorted_history[:8], start=1):
        metrics = item["metrics"]

        history_text += f"""Previous Prompt {i}
Instruction: {item["instruction"]}
Score: {item["score"]:.4f}
Accuracy: {metrics["accuracy"]:.4f}
Format Error Rate: {metrics["format_error_rate"]:.4f}
Average Output Tokens: {metrics["avg_tokens"]:.2f}

"""

    return f"""You are an expert prompt optimizer.

Your task is to generate better prompt instructions for math word problem solving.

We evaluate each prompt using:
- accuracy,
- final answer format compliance,
- output length.

A good prompt should:
1. improve mathematical correctness,
2. force the model to end with exactly one final line: Final Answer: <number>
3. avoid unnecessarily long responses.

Previous prompts and scores are shown below. Higher score is better.

{history_text}

Generate exactly {num_candidates} new prompt instructions.

You must follow these rules:
- Output ONLY one valid JSON array of strings.
- Do not output more than one JSON array.
- Do not use markdown.
- Do not use code blocks.
- Do not explain your reasoning.
- Do not include examples.
- Each string must be one concise prompt instruction.
- Each prompt must explicitly require the final line format: Final Answer: <number>.
- Stop immediately after the JSON array.

Output exactly like this:
[
  "Solve the problem carefully. End with exactly one line: Final Answer: <number>.",
  "Compute the answer concisely. The final line must be exactly: Final Answer: <number>."
]
"""


def clean_candidate_prompt(prompt):
    prompt = str(prompt).strip()

    prompt = prompt.replace("```python", "")
    prompt = prompt.replace("```json", "")
    prompt = prompt.replace("```", "")

    prompt = re.sub(r"^\s*[-*]\s*", "", prompt)
    prompt = re.sub(r"^\s*\d+\.\s*", "", prompt)

    stop_markers = [
        "### Step",
        "Step 1:",
        "Step 2:",
        "The given prompts",
        "These prompts aim",
        "Strengths",
        "Weaknesses",
        "# Prompt Optimization",
    ]

    for marker in stop_markers:
        idx = prompt.find(marker)
        if idx != -1:
            prompt = prompt[:idx].strip()

    prompt = re.sub(r"\s+", " ", prompt).strip()
    prompt = prompt.strip(" ,")
    prompt = prompt.strip('"')
    prompt = prompt.strip("'")

    return prompt


def is_valid_candidate(prompt):
    if not prompt:
        return False

    word_count = len(prompt.split())

    if word_count < 6:
        return False

    if word_count > 80:
        return False

    lower = prompt.lower()

    if "final answer" not in lower:
        return False

    bad_phrases = [
        "previous prompt",
        "accuracy:",
        "format error",
        "average output",
        "score:",
        "step 1",
        "step 2",
        "analyze the given",
        "json array",
    ]

    if any(x in lower for x in bad_phrases):
        return False

    return True


def extract_first_json_array(text):
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch != "[":
            continue

        try:
            parsed, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, list):
            return parsed

    return None


def parse_candidate_prompts(text, max_candidates=2):
    candidates = []

    # 1. Best case: parse the first valid JSON array only
    parsed = extract_first_json_array(text)

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str):
                candidates.append(item)

    # 2. Fallback: parse Prompt 1: ... style
    if not candidates:
        pattern = r"(?:\*\*)?\s*Prompt\s*\d+\s*(?:\*\*)?\s*:?\s*(.+?)(?=\n\s*(?:\*\*)?\s*Prompt\s*\d+\s*(?:\*\*)?\s*:|\Z)"
        matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(matches)

    cleaned = []
    seen = set()

    for c in candidates:
        c = clean_candidate_prompt(c)

        if not is_valid_candidate(c):
            continue

        key = c.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(c)

        if len(cleaned) >= max_candidates:
            break

    return cleaned
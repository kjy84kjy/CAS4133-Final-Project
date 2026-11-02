# data.py

import re
from datasets import load_dataset


def extract_gsm8k_gold(answer_text):
    """
    GSM8K answer format:
    ... #### 16
    """
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)

    if match is None:
        return None

    return match.group(1)


def convert_gsm8k_example(example):
    question = example["question"]
    answer_text = example["answer"]
    gold = extract_gsm8k_gold(answer_text)

    return {
        "question": question,
        "gold": gold,
        "raw_answer": answer_text,
    }


def load_gsm8k(train_size=40, test_size=50, seed=42):
    dataset = load_dataset("gsm8k", "main")

    train = dataset["train"].shuffle(seed=seed).select(range(train_size))
    test = dataset["test"].shuffle(seed=seed).select(range(test_size))

    train_data = [convert_gsm8k_example(x) for x in train]
    test_data = [convert_gsm8k_example(x) for x in test]

    return train_data, test_data


def load_svamp(train_size=40, test_size=50, seed=42):
    """
    Optional second dataset for later multiple-datasets experiment.
    Run only after GSM8K pipeline is stable.
    """
    dataset = load_dataset("ChilleD/SVAMP")

    data = dataset["train"].shuffle(seed=seed)
    total_size = train_size + test_size
    data = data.select(range(total_size))

    converted = []

    for x in data:
        question = str(x["Body"]) + " " + str(x["Question"])
        gold = str(x["Answer"])

        converted.append({
            "question": question,
            "gold": gold,
            "raw_answer": gold,
        })

    train_data = converted[:train_size]
    test_data = converted[train_size:train_size + test_size]

    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = load_gsm8k(train_size=2, test_size=2)
    print(train_data[0])
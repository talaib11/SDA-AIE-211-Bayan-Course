"""Lab 3B: inspect and run the supplied QA smoke set."""

import json
from pathlib import Path


DATA_PATH = Path("data/eval/qa_smoke_set.json")


def main():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    total = 0
    answerable = 0
    unanswerable = 0

    for item in payload["data"]:
        for paragraph in item["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1

                if qa.get("is_impossible", False):
                    unanswerable += 1
                else:
                    answerable += 1

    print(f"Total QA smoke questions: {total}")
    print(f"Answerable questions: {answerable}")
    print(f"Unanswerable questions: {unanswerable}")

    if total != 12:
        raise AssertionError(
            f"Expected 12 smoke questions, found {total}"
        )

    if answerable != 9 or unanswerable != 3:
        print(
            "WARNING: supplied smoke-set data does not match "
            "the README contract of 9 answerable + 3 unanswerable."
        )


if __name__ == "__main__":
    main()
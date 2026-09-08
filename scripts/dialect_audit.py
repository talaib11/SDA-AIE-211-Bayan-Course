"""Lab 4: audit dialect mix over the Arabic slice."""

import pandas as pd


DATA_PATH = "data/raw/bayan_feedback.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    arabic = df[df["lang"] == "ar"].copy()

    counts = arabic["dialect_region"].value_counts()
    percentages = arabic["dialect_region"].value_counts(
        normalize=True
    ) * 100

    print("Arabic rows:", len(arabic))
    print()

    print("Dialect distribution:")
    for region in counts.index:
        print(
            f"{region}: {counts[region]} "
            f"({percentages[region]:.1f}%)"
        )


if __name__ == "__main__":
    main()
"""Lab 3A: dataset construction and split integrity."""

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit


def build_topic_dataset(
    path="data/raw/bayan_feedback.csv",
    random_state=42,
):
    df = pd.read_csv(path)

    # First split: 70% train, 30% temporary.
    first_split = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        first_split.split(
            df,
            y=df["topic"],
            groups=df["citizen_group_id"],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split remaining 30% into 20% validation and 10% test.
    second_split = GroupShuffleSplit(
        n_splits=1,
        train_size=2 / 3,
        random_state=random_state,
    )

    valid_idx, test_idx = next(
        second_split.split(
            temp_df,
            y=temp_df["topic"],
            groups=temp_df["citizen_group_id"],
        )
    )

    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df,
                preserve_index=False,
            ),
            "validation": Dataset.from_pandas(
                valid_df,
                preserve_index=False,
            ),
            "test": Dataset.from_pandas(
                test_df,
                preserve_index=False,
            ),
        }
    )
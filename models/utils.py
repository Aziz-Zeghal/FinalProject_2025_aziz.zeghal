"""Utility functions for all recommender systems."""

import os
import pandas as pd


def get_data_path() -> str:
    """
    Get the path to the data directory.

    Returns:
        str: Path to the data directory.
    """
    DATA_PATH = "/kaggle/input/kuairec/KuaiRec 2.0/data"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = f"{os.getcwd()}/KuaiRec/data"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = f"{os.getcwd()}/../KuaiRec/data"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = f"{os.getcwd()}/KuaiRec 2.0/data"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = f"{os.getcwd()}/../KuaiRec 2.0/data"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("KuaiRec dataset not found. Please check the path.")

    return DATA_PATH


def matrix_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the data matrix into a more usable format.

    Args:
        df (pd.DataFrame): Either small_matrix or big_matrix.

    Returns:
        pd.DataFrame: Cleaned DataFrame.

    """
    # Date is time in a weird format

    # Time and Date are duplicated of timestamp, we can drop them
    df.drop(columns=["time", "date"], inplace=True)
    # Not a problem, we want to keep the data for the density
    df = df.astype(
        {
            "user_id": "int32",
            "video_id": "int32",
            "play_duration": "int32",
            "timestamp": "int64",
            "watch_ratio": "float32",
        },
        errors="ignore",
    )

    # Drop duplicates
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df[df["timestamp"] >= 0]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def my_describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Custom describe for datasets containing user_id and video_id

    Args:
        df (pd.DataFrame): DataFrame to describe (should contain user_id and video_id columns).

    Returns:
        pd.DataFrame: Description of the DataFrame.
    """
    print(f"Shape of the small matrix: {df.shape}")
    unique_users = df["user_id"].nunique()
    unique_posts = df["video_id"].nunique()
    print(f"Number of unique users: {unique_users}")
    print(f"Number of unique posts: {unique_posts}")
    print(f"Matrix sparsity: {len(df) / (unique_posts * unique_users) * 100}%")
    return df.describe()

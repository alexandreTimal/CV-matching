import logging

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


def load_dataset_fit() -> pd.DataFrame:
    """charge le dataset resume-job-description-fit depuis huggingface"""
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")

    dfs = [ds[split].to_pandas() for split in ds]
    df = pd.concat(dfs, ignore_index=True)

    logger.info("Dataset chargé: %d paires", len(df))
    logger.info("Colonnes: %s", list(df.columns))

    return df


def group_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split train/test en groupant par CV pour éviter le data leakage"""
    cv_col = 'resume_text'
    groups = df[cv_col].factorize()[0]

    n_unique = len(set(groups))
    logger.info("CVs uniques: %d pour %d paires", n_unique, len(df))

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_cvs = set(train_df[cv_col].values)
    test_cvs = set(test_df[cv_col].values)
    overlap = train_cvs & test_cvs
    logger.info("Train: %d | Test: %d", len(train_df), len(test_df))
    logger.info("Overlap CVs train/test: %d (doit être 0)", len(overlap))

    return train_df, test_df

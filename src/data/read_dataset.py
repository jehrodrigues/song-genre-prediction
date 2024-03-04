# -*- coding: utf-8 -*-
"""
Script used to read datasets files.
"""
import pandas as pd
import logging
from typing import Tuple
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads datasets."""
    path = project_dir / 'data' / 'processed'
    if path.exists():
        try:
            df_train = pd.read_csv(path / 'train.csv', delimiter=",",
                                   header=0, encoding='utf-8', engine='python')

            df_test = pd.read_csv(path / 'test.csv', delimiter=",",
                                  header=0, encoding='utf-8', engine='python')
        except pd.errors.EmptyDataError:
            logging.error(f'file is empty and has been skipped.')
        return df_train, df_test


class DatasetReader:
    """Handles dataset reading"""

    def __init__(self):
        pass

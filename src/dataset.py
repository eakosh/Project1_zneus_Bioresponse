import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from torch.utils.data import Dataset
from config import *


class BioresponseDataset(Dataset):
    """Dataset for bioresponse data with preprocessing options."""

    def __init__(self,
                 path: str = DATA_PATH,
                 preprocess: bool = True,
                 remove_duplicates: bool = True,
                 remove_zero_cols: bool = True,
                 remove_const_cols: bool = True,
                 remove_low_variance_cols: bool = True,
                 remove_outliers: bool = True,
                 variance_threshold: float = 0.01):
        """Load dataset and optionally preprocess it."""
        self.path = path
        self.df = pd.read_csv(path, sep=",")
        self.remove_duplicates = remove_duplicates
        self.remove_zero_cols = remove_zero_cols
        self.remove_const_cols = remove_const_cols
        self.remove_low_variance_cols = remove_low_variance_cols
        self.variance_threshold = variance_threshold
        self.remove_outliers = remove_outliers

        if preprocess:
            self.preprocess()

        self.X = self.df.drop(columns=['target'])
        self.y = self.df['target']

    def __len__(self):
        """Return dataset length."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return one sample (features, target)."""
        x = self.X.iloc[idx]
        y = self.y.iloc[idx]
        return x, y

    def preprocess(self):
        """Clean dataset by removing duplicates, constants, and low-variance features."""
        if self.remove_duplicates:
            self.df = self.df.drop_duplicates().reset_index(drop=True)

        if self.remove_zero_cols:
            zero_cols = [c for c in self.df.columns if self.df[c].notna().all() and (self.df[c] == 0).all()]
            self.df.drop(columns=zero_cols, inplace=True)

        if self.remove_const_cols:
            const_cols = [c for c in self.df.columns if self.df[c].dropna().nunique() <= 1]
            self.df.drop(columns=const_cols, inplace=True)

        if self.remove_low_variance_cols:
            features = [c for c in self.df.columns if c != 'target']
            X_temp, y = self.df[features], self.df['target']
            variance = X_temp.var()
            low_var = variance[variance < self.variance_threshold].index.tolist()
            if low_var:
                selector = VarianceThreshold(self.variance_threshold)
                X_temp = selector.fit_transform(X_temp)
                selected = [f for f, keep in zip(features, selector.get_support()) if keep]
                self.df = pd.concat([pd.DataFrame(X_temp, columns=selected, index=self.df.index), y], axis=1)
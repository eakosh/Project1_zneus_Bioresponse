import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch

from config import *
from dataset import BioresponseDataset


class DataModule:
    """Handles data loading, preprocessing, feature selection, normalization, and creaing DataLoader."""

    def __init__(self,
                 normalization: str = NORMALIZATION_METHOD,
                 stratify: bool = STRATIFY,
                 random_state: int = RANDOM_STATE,
                 test_size: float = TEST_SIZE,
                 val_size: float = VAL_SIZE,
                 feature_selection: bool = APPLY_FEATURE_SELECTION,
                 feature_selection_method: str = FEATURE_SELECTION_METHOD,
                 num_features: int = TOP_N_FEATURES,
                 batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS,
                 shuffle: bool = SHUFFLE,
                 variance_threshold: float = VARIANCE_THRESHOLD):
        """Initialize data module and preprocessing options."""
        self.dataset = BioresponseDataset(
            path=DATA_PATH,
            preprocess=PREPROCESS,
            remove_duplicates=REMOVE_DUPLICATES,
            remove_zero_cols=REMOVE_ZERO_COLUMNS,
            remove_const_cols=REMOVE_CONSTANT_COLUMNS,
            remove_low_variance_cols=REMOVE_LOW_VARIANCE_COLUMNS,
            variance_threshold=variance_threshold,
        )
        self.normalization = normalization
        self.stratify = stratify
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.feature_selection = feature_selection
        self.feature_selection_method = feature_selection_method
        self.num_features = num_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.data_split()

        if feature_selection:
            self.apply_feature_selection()
        if normalization:
            self.normalize()

    def setup(self):
        """Create TensorDatasets and DataLoaders."""
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train.values, dtype=torch.float32),
            torch.tensor(self.y_train.values, dtype=torch.float32).unsqueeze(1))
        self.val_dataset = TensorDataset(
            torch.tensor(self.X_val.values, dtype=torch.float32),
            torch.tensor(self.y_val.values, dtype=torch.float32).unsqueeze(1))
        self.test_dataset = TensorDataset(
            torch.tensor(self.X_test.values, dtype=torch.float32),
            torch.tensor(self.y_test.values, dtype=torch.float32).unsqueeze(1))

        self.dataloader_train = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                           shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
        self.dataloader_val = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=self.num_workers)
        self.dataloader_test = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                          shuffle=False, num_workers=self.num_workers)
        self.num_x = self.X_train.shape[1]

    def data_split(self):
        """Split dataset into train, validation, and test sets."""
        stratify = self.dataset.y if self.stratify else None
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.dataset.X, self.dataset.y,
            test_size=self.test_size, random_state=self.random_state, stratify=stratify)

        val_size = self.val_size / (1 - self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=self.random_state,
            stratify=y_temp if self.stratify else None)

    def apply_feature_selection(self):
        """Select top features using correlation, MI, or RandomForest."""
        sel_sets = []
        if 'corr' in self.feature_selection_method:
            corr = self.X_train.corrwith(self.y_train).abs().sort_values(ascending=False)
            sel_sets.append(corr.head(self.num_features).index.tolist())

        if 'mi' in self.feature_selection_method:
            mi = mutual_info_classif(self.X_train, self.y_train, random_state=self.random_state)
            mi_s = pd.Series(mi, index=self.X_train.columns).sort_values(ascending=False)
            sel_sets.append(mi_s.head(self.num_features).index.tolist())

        if 'rf' in self.feature_selection_method:
            rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            imp = pd.Series(rf.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
            sel_sets.append(imp.head(self.num_features).index.tolist())

        if sel_sets:
            self.selected_features = list(set().union(*sel_sets))
            self.X_train = self.X_train[self.selected_features].reset_index(drop=True)
            self.X_val = self.X_val[self.selected_features].reset_index(drop=True)
            self.X_test = self.X_test[self.selected_features].reset_index(drop=True)

    def normalize(self):
        """Normalize features (MinMax or Standard)."""
        if self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization == 'standard':
            self.scaler = StandardScaler()

        self.scaler.fit(self.X_train)
        self.X_train = pd.DataFrame(self.scaler.transform(self.X_train), index=self.X_train.index)
        self.X_val = pd.DataFrame(self.scaler.transform(self.X_val), index=self.X_val.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), index=self.X_test.index)
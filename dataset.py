import numpy as np
import pandas as pd
import os
import ast
import torch
from torch.utils.data import Dataset
import xarray as xr
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_static_features(feature_data, geography_metadata, save_path=None):
    """
    Load static geography features and align them to dataset gid ordering.
    """
    static_df = pd.read_csv(feature_data).copy()

    if "msoa" not in static_df.columns:
        raise ValueError("Static feature CSV must include a 'msoa' column")

    msoa_map = pd.Series(geography_metadata.index, index=geography_metadata["MSOA"])
    static_df["gid"] = static_df["msoa"].map(msoa_map).astype("Int64")

    unmapped = static_df["gid"].isna()
    if unmapped.any():
        print(f"[WARNING] Dropping {unmapped.sum()} static rows with MSOAs not found in geo_metadata.")
    static_df = static_df[~unmapped].copy()

    expected_gids = set(geography_metadata.index.tolist())
    found_gids = set(static_df["gid"].astype(int).tolist())
    missing_gids = sorted(expected_gids - found_gids)
    if len(missing_gids) > 0:
        raise ValueError(f"Static features are missing {len(missing_gids)} gids from geo_metadata.")

    if "lat_long" in static_df.columns:
        parsed = static_df["lat_long"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )
        static_df["latitude"] = parsed.apply(lambda x: x[0])
        static_df["longitude"] = parsed.apply(lambda x: x[1])

    preferred_columns = [
        "latitude",
        "longitude",
        "iomd_centile",
        "num_students",
        "num_residents_in_msoa",
        "population_density",
        "num_households_of_size_1",
        "num_households_of_size_2",
        "num_households_of_size_3",
        "num_households_of_size_4",
        "num_households_of_size_5",
        "num_households_of_size_6",
        "num_households_of_size_7",
        "num_households_of_size_8",
        "num_carehome_residents",
    ]

    selected = [c for c in preferred_columns if c in static_df.columns]
    if len(selected) == 0:
        excluded = {"msoa", "region", "lat_long", "gid"}
        selected = [c for c in static_df.columns if c not in excluded]

    static_df = static_df[["gid"] + selected].copy()
    static_df["gid"] = static_df["gid"].astype(int)
    static_df = static_df.sort_values("gid")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        static_df.to_csv(os.path.join(save_path, "processed_static_features.csv"), index=False)

    values = static_df[selected].to_numpy(dtype=np.float32)
    return xr.DataArray(
        values,
        coords={"gid": static_df["gid"].to_numpy(), "static_features": selected},
        dims=["gid", "static_features"],
    )


class DatesetDirectory():
    """
    Main data container that loads simulation data, applies scaling, and manages train/test/val splits.
    
    This class handles:
    - Loading time-series data from CSV with geographic indexing
    - Applying StandardScaler to temporal features (fitted on training data only)
    - Applying MinMaxScaler to static spatial features
    - Splitting simulations into train/test/validation sets
    - Supporting both count data (Negative Binomial) and continuous data (LogNormal)
    """
    
    def __init__(self, data_csv, metadata_csv, static_features_csv, split,
                 columns_for_prediction,
                 columns_for_context,
                 column_with_date='date',
                 column_with_geography='MSOA',
                 prediction_distribution_types=None,
                 continuous_prediction_columns=None,
                 temporal_scalers=None,
                 spatial_scaler=None,
                 fit_scalers=True,
                 inference_only=False,
                 min_timestep=0,
                 output_dir=None,
                 max_sims=None
    ):
        """
        Initialize the dataset directory.
        
        Args:
            data_csv: Path to main time-series CSV (columns: sim, date, MSOA, targets...)
            metadata_csv: Path to simulation metadata CSV (columns: sim, ...)
            static_features_csv: Path to geographic features CSV
            split: List of [train_count, test_count, val_count] or ratios
            columns_for_prediction: List of column names to predict
            columns_for_context: List of column names to use as context features
            column_with_date: Name of date column in data CSV
            column_with_geography: Name of geography column in data CSV
            prediction_distribution_types: List of 'nb' or 'lognormal' per prediction column
            continuous_prediction_columns: List of columns that should use LogNormal
            temporal_scalers: Pre-fitted scalers for temporal features (for inference)
            spatial_scaler: Pre-fitted scaler for spatial features (for inference)
            fit_scalers: Whether to fit scalers (False for inference with pre-fitted)
            inference_only: If True, all data goes to test set
            min_timestep: Minimum timestep to include in datasets (e.g., 30 for R_t)
            max_sims: Maximum number of simulations to include
        """
        # Store min_timestep for use in ProcessedData
        self.min_timestep = min_timestep
        
        self.columns_for_prediction = columns_for_prediction
        self.columns_for_context = columns_for_context
        self.columns_with_data = (
            list(self.columns_for_prediction)
            + [c for c in self.columns_for_context if c not in self.columns_for_prediction]
        )
        self.output_dir = output_dir

        print(f"Values for prediction {self.columns_for_prediction}")
        print(f"Data used for context {self.columns_for_context}")
        print(f"Columns with data {self.columns_with_data}")
        print(f"Minimum timestep: {self.min_timestep}")

        # Determining which distribution to use for each prediction column
        if prediction_distribution_types is not None: 
            if len(prediction_distribution_types) != len(self.columns_for_prediction):
                raise ValueError("prediction_distribution_types must match columns_for_prediction length")
            self.prediction_distribution_types = list(prediction_distribution_types)
        else:
            # If we don't specify, we assume all are 'nb' unless continuous_prediction_columns is specified
            types = ['nb'] * len(self.columns_for_prediction)
            if continuous_prediction_columns is not None:
                cont_set = set(continuous_prediction_columns)
                for i, name in enumerate(self.columns_for_prediction):
                    if name in cont_set:
                        types[i] = 'lognormal'
            self.prediction_distribution_types = types

        # Load data and rename relevant columns
        df_data = pd.read_csv(data_csv)
        old_names = [column_with_date, column_with_geography]
        new_names = ['date', 'MSOA']
        df_data.rename(columns=dict(zip(old_names, new_names)), inplace=True)

        # Change data format in main df
        df_data['date_format'] = pd.to_datetime(df_data['date'])

        self.max_sims = max_sims
        selected_sims = None
        if self.max_sims is not None:
            unique_sims = sorted(df_data['sim'].unique())
            selected_sims = unique_sims[: int(self.max_sims)]
            df_data = df_data[df_data['sim'].isin(selected_sims)].copy()
            print(f"Applying max_sims={self.max_sims}: using {len(selected_sims)} simulations")

        # Create geographic metadata
        df_data['MSOA'] = df_data['MSOA'].astype(str)
        unique_MSOA = sorted(df_data['MSOA'].unique())
        self.df_geo_metadata = pd.DataFrame({
            'gid': range(len(unique_MSOA)),
            'MSOA': unique_MSOA
        })
        geo_metadata_path = 'geo_metadata.csv'
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            geo_metadata_path = os.path.join(self.output_dir, 'geo_metadata.csv')
        self.df_geo_metadata.to_csv(geo_metadata_path, index=False)
        print(f"Saved geo_metadata to: {geo_metadata_path}")

        # Create start-time metadata (for day-of-week calculation)
        df_start_time_metadata = df_data.groupby(by='sim', as_index=False)['date_format'].min()
        df_start_time_metadata['tid'] = 0
        df_start_time_metadata['dow'] = df_start_time_metadata['date_format'].dt.weekday
        self.df_start_time_metadata = df_start_time_metadata

        # Add ids into main data frame
        # tid = number of days since the minimum date in each simulation
        group_min_dates = df_start_time_metadata.set_index('sim')['date_format']
        df_data['tid'] = (df_data['date_format'] - df_data['sim'].map(group_min_dates)).dt.days
        df_data['gid'] = df_data['MSOA'].map(self.df_geo_metadata.set_index('MSOA')['gid'])

        self.num_time_steps = len(df_data['tid'].unique())
        df_data.set_index(['sim', 'gid', 'tid'], inplace=True)
        self.df_data = df_data.sort_values(by=['sim', 'gid', 'tid'])

        # Add index to auxiliary tables
        self.df_geo_metadata.set_index(keys="gid", inplace=True)

        # Load auxiliary tables
        self.df_sim_metadata = pd.read_csv(metadata_csv).set_index('sim').sort_values(by='sim')
        self.xr_static_features = load_static_features(
            feature_data=static_features_csv, 
            geography_metadata=self.df_geo_metadata,
            save_path=self.output_dir
        )
        self.xr_static_features = self.xr_static_features.sortby('gid')
        self.num_static_features = self.xr_static_features.shape[1]

        self.num_simulations = len(self.df_sim_metadata)
        self.num_geographies = len(self.df_geo_metadata)
        self.num_data_variables = len(self.columns_with_data)

        # Create scalers or reuse provided ones
        self.spatial_scaler = spatial_scaler if spatial_scaler is not None else MinMaxScaler()
        if temporal_scalers is not None:
            if len(temporal_scalers) != len(self.columns_with_data):
                raise ValueError("temporal_scalers length must match columns_with_data length")
            self.temporal_scalers = temporal_scalers
        else:
            self.temporal_scalers = [StandardScaler() for _ in self.columns_with_data]

        # Scale static features (fit if requested/needed)
        if fit_scalers and spatial_scaler is None:
            scaled_values = self.spatial_scaler.fit_transform(self.xr_static_features.values)
        else:
            scaled_values = self.spatial_scaler.transform(self.xr_static_features.values)
            
        self.xr_static_features_scaled = xr.DataArray(
            scaled_values,
            coords=self.xr_static_features.coords,
            dims=self.xr_static_features.dims
        )
        self.xr_static_features_scaled = self.xr_static_features_scaled.sortby('gid')

        # Create split from input
        if inference_only:
            split = np.array(split, dtype=int)
            if split.sum() == 0:
                split = np.array([0, self.num_simulations, 0], dtype=int)
        else:
            split = np.array(split)
            split = np.round(split * (self.num_simulations / split.sum())).astype(int)
            split[-1] = self.num_simulations - split[:-1].sum()
            if split[0] == 0:
                raise ValueError(f"split leads to an empty training set.")
            if split[1] == 0:
                raise ValueError(f"split leads to an empty testing set.")
            if split[2] == 0:
                raise ValueError(f"split leads to an empty validation set.")
        self.split = split

        # Split simulations into training, testing, validation
        np.random.seed(42)
        labels = ['training'] * self.split[0] + ['testing'] * self.split[1] + ['validation'] * self.split[2]
        np.random.shuffle(labels)
        self.df_sim_metadata['split'] = labels

        # Save split sims
        self.train_sims = self.df_sim_metadata[self.df_sim_metadata['split'] == 'training'].index.get_level_values('sim').to_numpy()
        self.test_sims = self.df_sim_metadata[self.df_sim_metadata['split'] == 'testing'].index.get_level_values('sim').to_numpy()
        self.val_sims = self.df_sim_metadata[self.df_sim_metadata['split'] == 'validation'].index.get_level_values('sim').to_numpy()

        # Fit/apply temporal scalers (fit only on training data)
        train_mask = self.df_data.index.get_level_values('sim').isin(self.train_sims)
        for col, scaler in zip(self.columns_with_data, self.temporal_scalers):
            if fit_scalers and temporal_scalers is None:
                values_training = self.df_data.loc[train_mask, col].to_numpy().reshape(-1, 1)
                scaler.fit(values_training)
            self.df_data[col + '_scaled'] = scaler.transform(self.df_data[[col]].values)

        # Create list of scalers aligned to columns_for_prediction
        self.scalers_for_prediction = [
            self.temporal_scalers[self.columns_with_data.index(col)]
            for col in self.columns_for_prediction
        ]

        print("Converting data to xarray and tensors (once)...")
        # Pre-convert all data to xarray and then tensors to share between splits
        self.list_xr_data = [self.df_data[col + '_scaled'].to_xarray() 
                             for col in self.columns_with_data]
        self.raw_list_xr_data = [self.df_data[col].to_xarray()
                                 for col in self.columns_with_data]
        
        self.all_sims = self.list_xr_data[0].sim.values.tolist()
        self.sim_id_to_idx = {}
        for i, sim in enumerate(self.all_sims):
            self.sim_id_to_idx[sim] = i
            self.sim_id_to_idx[str(sim)] = i

        # Convert all data to [num_vars, sim, gid, tid] tensors
        # These will be shared by all ProcessedData instances
        self.data_tensor = torch.stack([torch.from_numpy(xr.values).float() for xr in self.list_xr_data])
        self.raw_data_tensor = torch.stack([torch.from_numpy(xr.values).float() for xr in self.raw_list_xr_data])
        
        # Pre-convert static features [gid, static_features] -> [num_geographies, num_static_features]
        self.static_features_tensor = torch.from_numpy(self.xr_static_features_scaled.values).float()

        print("Data loaded and pre-processed.")
        print(f"Data split into training ({self.split[0]} instances), "
              f"testing ({self.split[1]} instances), "
              f"validation ({self.split[2]} instances)...")

    def resolve_sim_idx(self, sim_id):
        """Resolve simulation identifiers coming from numpy/pandas containers."""
        if sim_id in self.sim_id_to_idx:
            return self.sim_id_to_idx[sim_id]

        candidates = []
        if hasattr(sim_id, "item"):
            try:
                candidates.append(sim_id.item())
            except Exception:
                pass

        candidates.append(str(sim_id))

        try:
            candidates.append(int(sim_id))
        except (TypeError, ValueError):
            pass

        for candidate in candidates:
            if candidate in self.sim_id_to_idx:
                return self.sim_id_to_idx[candidate]

        raise KeyError(sim_id)

    def get_raw_data_split(self):
        """Get raw data with split labels."""
        df_data = pd.merge(self.df_data, self.df_sim_metadata, on='sim', how='left')
        df_data = df_data.reset_index()
        return df_data
    
    def get_column_as_xarray(self, column, type):
        """Get a specific column as xarray for a given split type."""
        if type == "training":
            sims = self.train_sims
        elif type == "testing":
            sims = self.test_sims
        elif type == "validation":
            sims = self.val_sims
        
        list_xr_data = self.df_data[column].to_xarray()
        data = list_xr_data[sims]
        return data

        
class ProcessedData(Dataset):
    """
    PyTorch Dataset for iterating over simulation data with context windows.
    
    This class provides:
    - Iteration over (sim, timestep) pairs
    - Context window construction with optional zero-padding
    - Configurable minimum timestep (e.g., skip first 30 days for R_t)
    """
    
    def __init__(self, data_directory: DatesetDirectory, past_context_size, type, loss_type="nll", multistep_k=1):
        """
        Initialize the processed dataset.
        
        Args:
            data_directory: DatesetDirectory instance with loaded data
            past_context_size: Number of past timesteps to include in context
            type: One of "training", "testing", "validation"
            loss_type: "nll" for raw counts, "mse" for scaled values
            multistep_k: Number of future timesteps to predict (defaults to 1)
        """
        self.data_directory = data_directory
        self.df_start_time_metadata = data_directory.df_start_time_metadata
        self.xr_static_features_scaled = data_directory.xr_static_features_scaled
        self.num_time_steps = data_directory.num_time_steps
        self.past_context_size = past_context_size
        self.columns_for_prediction = data_directory.columns_for_prediction
        self.columns_for_context = data_directory.columns_for_context
        self.columns_with_data = data_directory.columns_with_data
        self.loss_type = loss_type
        self.multistep_k = multistep_k
        
        # Get min_timestep from data_directory (default 0 for backward compatibility)
        self.min_timestep = getattr(data_directory, 'min_timestep', 0)
        
        # Extracts data and shapes it into an xarray cube (dim: sim, gid, tid)
        # Reference pre-converted tensors and mappings from data_directory
        self.sim_id_to_idx = data_directory.sim_id_to_idx
        self.data_tensor = data_directory.data_tensor
        self.raw_data_tensor = data_directory.raw_data_tensor
        self.static_features_tensor = data_directory.static_features_tensor

        # DOW mapping: sim_id -> start_dow
        raw_dow_dict = self.df_start_time_metadata.set_index('sim')['dow'].to_dict()
        self.sim_to_start_dow = {}
        for k, v in raw_dow_dict.items():
            self.sim_to_start_dow[k] = v
            self.sim_to_start_dow[int(k)] = v
            self.sim_to_start_dow[str(k)] = v

        # Number of geographies (M)
        self.num_geographies = data_directory.num_geographies
        
        # Index mappings for columns
        self.ctx_col_indices = [self.columns_with_data.index(c) for c in self.columns_for_context]
        self.pred_col_indices = [self.columns_with_data.index(c) for c in self.columns_for_prediction]

        # Compatibility with get_testing_data
        self.list_xr_data = data_directory.list_xr_data
        self.raw_list_xr_data = data_directory.raw_list_xr_data

        # Saves simulations per dataset
        if type == "training":
            self.sims = data_directory.train_sims
        elif type == "testing":
            self.sims = data_directory.test_sims
        elif type == "validation":
            self.sims = data_directory.val_sims
        else:
            raise ValueError(f"invalid 'type': {type}")

        # Calculate number of valid timesteps per simulation
        # If min_timestep > 0, we skip early timesteps (e.g., for R_t which needs history)
        # Re-adjust valid timesteps so that we have enough future targets for multistep prediction
        self.valid_timesteps = self.num_time_steps - self.min_timestep - (self.multistep_k - 1)
        
        if self.valid_timesteps <= 0:
            raise ValueError(
                f"min_timestep ({self.min_timestep}) >= num_time_steps ({self.num_time_steps}). "
                f"No valid timesteps available."
            )
        
        self.number_items = self.valid_timesteps * len(self.sims)
        
        print(f"ProcessedData [{type}]: {len(self.sims)} sims × {self.valid_timesteps} timesteps "
              f"(t={self.min_timestep} to t={self.num_time_steps-1}) = {self.number_items} items")

    def __len__(self):
        return self.number_items

    def __getitem__(self, idx):
        """
        Get a single training/testing example.
        
        Returns:
            Tuple of (static_features, current_t, current_dow, *contexts, *targets)
            where:
            - static_features: [num_msoa, num_static_features]
            - current_t: scalar timestep
            - current_dow: scalar day of week (0-6)
            - contexts: list of [num_msoa, context_size] tensors
            - targets: list of [num_msoa] tensors
        """
        # Calculate which simulation and timestep this index corresponds to
        sim_id = self.sims[idx // self.valid_timesteps]
        sim_idx = self.data_directory.resolve_sim_idx(sim_id)
        
        # current_t starts from min_timestep, not 0
        current_t = (idx % self.valid_timesteps) + self.min_timestep
        
        # Calculate day of week
        resolved_sim_id = self.data_directory.all_sims[sim_idx]
        start_dow_sim = self.sim_to_start_dow[resolved_sim_id]
        current_dow = (start_dow_sim + current_t) % 7

        # Build context values [M, context_size]
        context_values_list = []
        start_tid = current_t - self.past_context_size
        
        for var_idx in self.ctx_col_indices:
            # Efficient tensor indexing: [sim, gid, tid]
            var_data = self.data_tensor[var_idx, sim_idx] # [M, all_tids]
            
            if start_tid >= 0:
                # Normal slice
                context_tensor = var_data[:, start_tid:current_t]
            else:
                # Pad with zeros if window starts before tid=0
                num_padding = abs(start_tid)
                context_tensor = var_data.new_zeros((self.num_geographies, self.past_context_size))
                data_part = var_data[:, 0:current_t]
                context_tensor[:, num_padding:] = data_part
            
            context_values_list.append(context_tensor)

        # Build current targets [M] or [M, multistep_k]
        # self.loss_type determines if we use raw or scaled data
        active_tensor = self.data_tensor if self.loss_type == "mse" else self.raw_data_tensor
        
        current_values_list = []
        for var_idx in self.pred_col_indices:
            if self.multistep_k == 1:
                targets_k = active_tensor[var_idx, sim_idx, :, current_t]
            else:
                targets_k = active_tensor[var_idx, sim_idx, :, current_t : current_t + self.multistep_k]
            current_values_list.append(targets_k)

        return tuple([self.static_features_tensor, current_t, current_dow] + context_values_list + current_values_list)


def get_testing_data(testing_dataset: ProcessedData, dataset_directory: DatesetDirectory):
    """
    Extract ground truth arrays for scoring during testing.
    
    Returns:
        truth_scaled: np.ndarray of shape [num_time_steps, num_msoas, num_predictions]
        truth_unscaled: np.ndarray of same shape (inverse transformed)
    
    Note: Returns data for ALL timesteps (0 to num_time_steps-1), not just valid ones.
          The testing loop handles the min_timestep offset.
    """
    truth_scaled_list, truth_unscaled_list = [], []

    # Loop in the order of the prediction cols
    for idx, col_name in enumerate(dataset_directory.columns_for_prediction):
        idx_in_data = dataset_directory.columns_with_data.index(col_name)

        # Get the scaled truth for this column → [num_time_steps, num_msoas]
        xr_col = testing_dataset.list_xr_data[idx_in_data]
        ts_scaled = (
            xr_col
            .where(xr_col.sim == testing_dataset.sims[0], drop=True)
            .squeeze(dim="sim")
            .transpose("tid", "gid")
            .values
        )
        truth_scaled_list.append(ts_scaled)

        # Unscale it
        scaler = dataset_directory.scalers_for_prediction[idx]
        flat = ts_scaled.reshape(-1, 1)
        unflat = scaler.inverse_transform(flat).reshape(ts_scaled.shape)
        truth_unscaled_list.append(unflat)

    # Stack so axis=-1 aligns with columns_for_prediction
    truth_scaled = np.stack(truth_scaled_list, axis=-1)
    truth_unscaled = np.stack(truth_unscaled_list, axis=-1)

    return truth_scaled, truth_unscaled

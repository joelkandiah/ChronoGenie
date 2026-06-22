from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2
from torch_geometric.data import Data
import torch
import os
import numpy as np
import pandas as pd
import xarray as xr
from torch_geometric.utils import to_undirected
import csv
import torch
from datetime import datetime

def save_msoa_pairs(data, output_name):
    """
    Save the unique undirected MSOA edge list from a PyG graph to CSV.

    Inputs:
        data:
            A torch_geometric.data.Data graph object that must contain:
            - data.edge_index: Tensor of shape [2, E] giving edges as (source_node, target_node)
            - data.msoa_names: List/array of MSOA names where msoa_names[node_id] gives the label for that node_id

        output_name:
            Path to the CSV file to write.

    Output:
        Writes a CSV with columns ["msoa_i", "msoa_j"].
        Each row is one unique undirected edge (so (2,5) and (5,2) are treated as the same edge).
        Self-loops are skipped.
        Returns: None
    """

    # Get the MSOA names from the graph object, if it doesn't exist it will throw an error.
    if not hasattr(data, "msoa_names") or data.msoa_names is None:
        raise AttributeError(
            "save_msoa_pairs expects `data.msoa_names` to exist and be non-None. "
            "Attach MSOA names when building the graph (e.g. in get_graph(...))."
        )

    # Convert MSOA names into strings
    labels = [str(x) for x in data.msoa_names]

    ##########################################
    # Getting all the unique undirectd edges #
    ##########################################
    # edge_index is expected to be [2, E] where the first row is sources and second row is targets.
    i, j = data.edge_index

    # Storing all the edges using a set so we remove any duplicates
    pairs = set()

    # Iterate over all directed edges (a -> b) in edge_index.
    for a, b in zip(i.tolist(), j.tolist()):
        # We don't want to record self-loop so skipping here.
        if a == b:
            continue

        # Make sure this edge is always written in the same order (small index first).
        # Example: if we see (5,2), we rewrite it as (2,5). If we see (2,5), it stays (2,5).
        # This way (2,5) and (5,2) are treated as the same undirected edge and don't get saved twice.
        i, j = (a, b) if a < b else (b, a)

        # Add the undirected pair to the set.
        pairs.add((i, j))

    ##############################
    # Write the edge list to CSV #
    ##############################
    # Open the output CSV file for writing.
    with open(output_name, "w", newline="") as f:  # newline="" avoids extra blank lines on Windows when using csv.writer.
        # Creating a CSV writer object.
        writer = csv.writer(f)

        # Write the header row.
        writer.writerow(["msoa_i", "msoa_j"])

        # Sort pairs for deterministic output order.
        for i, j in sorted(pairs):
            # Convert node indices (i,j) into the chosen labels and write the row.
            writer.writerow([labels[i], labels[j]])

def load_static_features(feature_data, geography_metadata, save_path=None):
    """
    Load MSOA-level static features from CSV, align them to the model's internal gid indexing,
    and return them as an xarray.DataArray.

    Inputs:
        feature_data:
            Path to a CSV containing static features per MSOA. Must include:
            - "msoa" column (MSOA code/name)
            - "lat_long" column (string that evaluates to [lat, lon])
            - plus other feature columns used below

        geography_metadata:
            A pandas DataFrame indexed by `gid`, with at least an "MSOA" column.
            It should look like this example:

            gid,MSOA
            0,E02001684
            1,E02001688
            2,E02001689
            3,E02001691
            4,E02001692
            5,E02001693
            6,E02001694
            7,E02001695

            The `gid` is the internal node id, and `geography_metadata["MSOA"]` gives the MSOA code/name for that node id.

    Output:
        xr.DataArray with:
            - dims: ['gid', 'static_features']
            - coords:
                gid = model node ids (aligned to graph / dataset ordering)
                static_features = column names of the chosen features
    """

    # Read the static feature CSV into a pandas DataFrame.
    feature_data = pd.read_csv(feature_data).copy()

    #############################
    # Map MSOA to internal gids #
    ##############################
    # Creating a lookup table to map MSOA to gid
    # geography_metadata.index is gid, and geography_metadata["MSOA"] contains the MSOA labels.
    msoa_map = pd.Series(geography_metadata.index, index=geography_metadata['MSOA'])
    # Add a gid column to the features table by mapping feature_data["msoa"] through msoa_map.
    feature_data['gid'] = feature_data['msoa'].map(msoa_map).astype('Int64')
    
    # Drop rows where we couldn't map "msoa" to a gid (keeps only MSOAs present in geography_metadata).
    #feature_data = feature_data[feature_data['gid'].notna()]

    ##################################
    # Checking MSOA to gid alignment #
    ##################################
    # The "gid" will be NaN for any static feature rows whose "msoa" value does not exist in geo_metadata
    unmapped = feature_data['gid'].isna()
    # If there are any unmapped rows, we raise a warning
    # NOTE: To be honest, this could be an error......
    if unmapped.any():
        print(
            f"[WARNING] Dropping {unmapped.sum()} static rows with MSOAs not found in geo_metadata. "
        )

    # Keep the rows that successfully mapped to a gid.
    feature_data = feature_data[~unmapped].copy()

    ######################################################
    # Check that there are static features for every gid #
    ######################################################
    # The dataset expects one static-feature row per gid in geo_metadata.
    expected_gids = set(geography_metadata.index.tolist())
    # The gids we actually have after mapping.
    found_gids = set(feature_data['gid'].astype(int).tolist())
    # Any gids that are expected but missing means we would be missing nodes/features downstream.
    missing_gids = sorted(expected_gids - found_gids)

    # If we are missing any gids we raise an Error.
    if len(missing_gids) > 0:
        raise ValueError(
            f"Static features are missing {len(missing_gids)} gids from geo_metadata. "
        )

    ###############################
    # Features and feature groups #
    ###############################
    # The CSV stores lat/long as a string like "[52.1, -0.2]".
    lat_long = "lat_long"
    feature_data[lat_long] = feature_data[lat_long].apply(eval) # Convert the lat/long string into a Python list.

    # Split the lat_long list into two separate numeric columns.
    feature_data['latitude'] = feature_data[lat_long].apply(lambda x: x[0])
    feature_data['longitude'] = feature_data[lat_long].apply(lambda x: x[1])

    # NOTE: (We don't currently use these feature grouping, but they are legacy and I am keeping them here in case they may be useful) #
    columns_to_drop = [lat_long, "region", "msoa"]
    columns_hospital = [col for col in feature_data.columns if col.startswith("hospital")]
    columns_carehomes = [col for col in feature_data.columns if col.startswith("num_carehome")]
    columns_school = [col for col in feature_data.columns if col.startswith("school")]
    columns_university = [col for col in feature_data.columns if col.startswith("university")]
    columns_students = ["num_students"]
    columns_companies = [col for col in feature_data.columns if col.startswith("num_companies")]
    columns_workers = [col for col in feature_data.columns if col.startswith("num_workers")]
    columns_ethnicity = [col for col in feature_data.columns if col.startswith("ethnicity")]
    columns_households = [col for col in feature_data.columns if col.startswith("num_households")]
    columns_iomd = [col for col in feature_data.columns if col.startswith("iomd")]
    columns_other = ["area_of_msoa", "num_residents_in_msoa", "population_density"]

    ####################################
    # THE FEATURE SET WE CURRENTLY USE #
    ####################################
    keep_columns = [
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
        "gid"
    ]

    # Timestamp for saving a copy of the processed static features to disk.
    timestamp = datetime.now().strftime("%d_%m_%y")

    # Keeping only the columns we want.
    feature_data = feature_data[keep_columns]

    ########################################
    # Save a copy for debugging/inspection #
    ########################################
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        static_features_path = os.path.join(save_path, 'processed_static_features.csv')
    else:
        timestamp = datetime.now().strftime("%d_%m_%y")
        os.makedirs(f"STATIC_FEATURES_{timestamp}", exist_ok=True)
        static_features_path = f"STATIC_FEATURES_{timestamp}/static_features.csv"
    feature_data.to_csv(static_features_path)
    print(f"Saved processed static features to: {static_features_path}")

    # Printing summary
    print(f"NUMBER OF LOCATION SPECIFIC STATIC FEATURES: {len(feature_data.columns)}")
    print(f"FEATURES: {feature_data.columns}")

    #####################################
    # Convert the DataFrame into xarray #
    #####################################
    # Reset index so we can safely set gid as the index next (avoid carrying an old index through).
    feature_data = feature_data.reset_index()

    # Set gid as the index so rows align to the model's node ordering.
    feature_data.set_index(keys='gid', inplace=True)

    # Drop the old index column created by reset_index().
    feature_data = feature_data.drop(columns=["index"])

    # Convert the pandas DataFrame to an xarray.DataArray.
    # - rows: gid
    # - columns: static feature names
    feature_data = xr.DataArray(
        feature_data,
        dims=['gid', 'static_features'],
        coords={
            'gid': feature_data.index,
            'static_features':feature_data.columns
        }
    )

    # Return the final xarray DataArray of static features.
    return feature_data

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points on Earth using the Haversine formula.

    Inputs:
        lat1, lon1:
            Latitude and longitude of point 1 in degrees.
        lat2, lon2:
            Latitude and longitude of point 2 in degrees.

    Output:
        Distance between the two points in kilometres.
    """

    # Convert input coordinates from degrees to radians (trig functions expect radians).
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in latitude and longitude (in radians).
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula:
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Approximate radius of the Earth in kilometres.
    r = 6371

    # Arc length = radius * angular distance.
    return r * c

def get_graph(scaled_features,
              unscaled_features, 
              edge_haversine_threshold: float,
              use_k_nearest_features: bool,
              use_k_nearest_distance: bool,
              knn_k: int,
              msoa_lookup: pd.DataFrame,
              save_path: str
              ):
    
    """
    GRAPH CONSTRUCTION

    This function builds a PyTorch Geometric graph where each node represents a single MSOA
    (ordered by `gid`) and edges represent neighbourhood relationships between MSOAs.

    Neighbourhoods can be constructed using one or more of the following rules:

    1) Feature-space k-nearest neighbours:
    - Uses Euclidean distance in scaled static feature space
    - Connects each MSOA to its k nearest neighbours based on static features

    2) Geographic-distance k-nearest neighbours:
    - Uses Haversine distance (km) computed from latitude/longitude
    - Connects each MSOA to its k nearest geographic neighbours

    3) Geographic distance threshold:
    - Connects any pair of MSOAs whose Haversine distance is <= threshold (km)

    If multiple modes are enabled, the union of all resulting undirected edges is used.
    Self-loops are excluded.

    Edge weights are based on Haversine distance and are normalised and inverted so
    closer MSOAs have higher edge weights.

    Inputs:
        scaled_features:
            xarray.DataArray of static features scaled to [0,1], indexed by `gid`
            (dims: ['gid', 'static_features'])

        unscaled_features:
            xarray.DataArray of raw static features (must include latitude/longitude),
            indexed by `gid` in the same order as `scaled_features`

        edge_haversine_threshold:
            Maximum Haversine distance (km) for connecting two MSOAs.
            If None, threshold-based edges are not added.

        use_k_nearest_features:
            If True, add k-nearest-neighbour edges using Euclidean distance in
            scaled static feature space.

        use_k_nearest_distance:
            If True, add k-nearest-neighbour edges using Haversine distance in km.

        knn_k:
            Number of neighbours to use for kNN-based graph construction.

        msoa_lookup:
            Mapping from `gid` to MSOA code/name.
            Expected to be either:
            - pandas Series indexed by gid, or
            - dict of the form {gid: msoa_name}

        save_path:
            Directory where diagnostic outputs (e.g. MSOA edge list, kNN tables)
            will be written.

    Output:
        graph_data:
            torch_geometric.data.Data object with:
            - edge_index: [2, E] tensor of undirected edges
            - edge_attr: [E] tensor of edge weights (normalised, higher = closer)
            - num_nodes: number of MSOAs
            - pos: [num_nodes, 2] tensor of (lon, lat) coordinates
            - gid: tensor mapping node index -> gid
            - msoa_names: list mapping node index -> MSOA name
    """
    # Ensure both datasets are sorted by gid
    unscaled_sorted = unscaled_features.sortby('gid')
    scaled_sorted = scaled_features.sortby('gid')

    # Default (prevents UnboundLocalError in degenerate cases, e.g. num_nodes <= 1)
    undirected_edges_count = 0

    # A number of checks to ensure data integrity and get_graph parameters are valid
    assert np.array_equal(unscaled_sorted.gid.values, scaled_sorted.gid.values), \
        "There is a gid mismatch between scaled and unscaled features"
    
    # If both flags are True, we will take the union of k-NN in feature space
    # and k-NN in geographic (haversine) distance space.

    if not use_k_nearest_features and not use_k_nearest_distance and (
        edge_haversine_threshold is None
    ):
        raise ValueError(
            "You must choose at least one graph construction mode: "
            "set use_k_nearest_features=True, use_k_nearest_distance=True, "
            "or provide a edge_haversine_threshold that is not None."
        )

    if (use_k_nearest_features or use_k_nearest_distance) and (knn_k is None or knn_k < 1):
        raise ValueError("knn_k must be a positive integer when using k-nearest neighbour graph construction.")

    # Extracting features needed for graph construction
    lat = unscaled_sorted.sel(static_features="latitude").values
    lon = unscaled_sorted.sel(static_features="longitude").values
    features = [c for c in unscaled_sorted.coords['static_features'].values if c not in ['gid']]
    xr_node_features = scaled_sorted.sel(static_features=features)
    num_nodes = len(scaled_sorted)

    edge_index, edge_weights = [], []

    ############################################################
    # Using k-nearest neighbours based on static feature space #
    ############################################################
    if use_k_nearest_features:
        knn_features = xr_node_features.values
        if num_nodes > 1 and knn_k > 0: # Checking we have mmore than 1 node and knn_k is positive
            num_k = min(knn_k + 1, num_nodes) # We +1 so we can drop self and still keep knn_k
            neighbours = NearestNeighbors(n_neighbors=num_k, metric='euclidean') # Using Euclidean distance in feature space
            neighbours.fit(knn_features) # Fitting the model
            distances, index = neighbours.kneighbors(knn_features) # Getting distances and indices of neighbours

            pairs = set() # Using a set to avoid duplicate undirected edges
            for i in range(num_nodes): # Iterationg over each node
                # Remove self and keep exact knn_k neighbours in the order returned (nearest-first)
                node_i_neighours = [int(j) for j in index[i] if int(j) != i][:knn_k]
                
                for j in node_i_neighours: # Iterate over all neighbours of node i
                    a, b = (i, j) if i < j else (j, i) # By checking i<j we ensure undirected edges are only added once
                    if a != b: # Avoiding self-loops
                        pairs.add((a, b)) # Adding undirected edge

            # Convert pairs set to edge_index and edge_weights based on haversine distance
            edge_index = [[a, b] for (a, b) in pairs] # Converting to list of lists for edge_index
            edge_weights = [haversine(lat[a], lon[a], lat[b], lon[b]) for (a, b) in pairs] # Haversine distances as weights
            undirected_edges_count = len(pairs) # Counting unique undirected edges
        
            # Saving k-nearest neighbour features to CSV 
            gids = scaled_sorted.gid.values # Getting gids in node order
            unscaled_vals = unscaled_sorted.sel(static_features=features).values

            rows = []
            for i in range(num_nodes): # Iterating over each node
                rank = 0 # Rank of neighbour for node i
                # Iterating over neighbours of node i and their euclidean distances
                for j, euclidean_distance in zip(index[i], distances[i]):  
                    if j == i: # Skipping self-loop
                        continue
                    rank += 1 # Incrementing rank after skipping self-loop

                    if rank > knn_k:
                        break
                    
                    row = {
                        "rank": int(rank),
                        "euclidean_distance": float(euclidean_distance),
                        "haversine_km": float(haversine(lat[i], lon[i], lat[j], lon[j]))
                    }

                    row["node_msoa"] = msoa_lookup.loc[int(gids[i])]
                    row["nbr_msoa"]  = msoa_lookup.loc[int(gids[j])]

                    # Getting the static features for both node and neighbour
                    for c_idx, cname in enumerate(features):
                        row[f"node_{cname}"] = unscaled_vals[i, c_idx]
                        row[f"nbr_{cname}"] = unscaled_vals[j, c_idx]
                    rows.append(row)

            df_knn = pd.DataFrame(rows)
            knn_features_file = "knn_neighbours_features"
            save_path_knn_features = os.path.join(save_path, knn_features_file)
            os.makedirs(os.path.dirname(save_path_knn_features) or ".", exist_ok=True)
            df_knn.to_csv(save_path_knn_features + ".gz", index=False, compression="gzip")

            # If distance-based kNN is also requested, take the union with distance kNN
            if use_k_nearest_distance:
                # Compute full haversine matrix D
                D = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
                for i, j in combinations(range(num_nodes), 2):
                    d = haversine(lat[i], lon[i], lat[j], lon[j])
                    D[i, j] = d
                    D[j, i] = d

                # Build kNN by geographic distance
                N = min(max(int(knn_k), 1), num_nodes - 1)
                pairs_dist = set()
                for i in range(num_nodes):
                    order = np.argsort(D[i])
                    nbrs = [int(j) for j in order if j != i][:N]
                    for j in nbrs:
                        a, b = (i, j) if i < j else (j, i)
                        if a != b:
                            pairs_dist.add((a, b))

                # Union of feature-space kNN and distance-space kNN
                # pairs: undirected edges from feature-space kNN (Euclidean on X)
                # pairs_dist: undirected edges from distance-space kNN (Haversine on lat/lon)
                #
                # Both are sets of tuples (a, b)
                #
                # The union means keeps an edge if it was selected by either metric.
                pairs_union = pairs.union(pairs_dist)
                # Converting the set to a list of pairs format
                edge_index = [[a, b] for (a, b) in pairs_union]
                # Assigning haversine distances as weights based on the full distance matrix D
                edge_weights = [D[a, b] for (a, b) in pairs_union]
                undirected_edges_count = len(pairs_union)
            else:
                # Degenerate case: no edges can be formed
                edge_index, edge_weights = [], []
                undirected_edges_count = 0
    
    else:
        # Computing haversine distance matrix between all nodes
        D = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64) # Initialising distance matrix
        for i, j in combinations(range(num_nodes), 2): # Iterating over all unique node pairs
            d = haversine(lat[i], lon[i], lat[j], lon[j]) # Computing haversine distance
            D[i, j] = d # Storing distance for one direction
            D[j, i] = d # Storing distance for the other direction

        edge_index, edge_weights = [], []

        ##########################################################
        # Using k-nearest neighbours based on haversine distance #
        ##########################################################
        if use_k_nearest_distance:
            N = min(max(int(knn_k), 1), num_nodes - 1)
            pairs = set()
            for i in range(num_nodes): # Iterating over each node i
                order = np.argsort(D[i]) # Order the distances ascendingly
                nbrs = [int(j) for j in order if j != i][:N] # Getting top N neighbours excluding self
                for j in nbrs: # Iterate over all neighbours of node i
                    a, b = (i, j) if i < j else (j, i) # Getting undirected edge
                    if a != b: # Avoiding self-loops
                        pairs.add((a, b)) # Adding undirected edge
            
            # Convert pairs set to edge_index and edge_weights based on haversine distance
            edge_index = [[a, b] for (a, b) in pairs] 
            edge_weights = [D[a, b] for (a, b) in pairs]

        #########################################
        # Using haversine distance thresholding #
        #########################################
        if edge_haversine_threshold is not None: 
            for i, j in combinations(range(num_nodes), 2): # Iterating over all unique node pairs
                d = D[i, j] # Getting haversine distance 
                if d <= edge_haversine_threshold: # If within threshold, add edge
                    edge_index.append([i, j]) 
                    edge_weights.append(d)

        undirected_edges_count = len(edge_index)

    # Handle case where no edges are created (threshold = 0)
    if not edge_index:
        # Creating an empty edge tensor with shape [2,E] where E=0
        # PyGeometric expects a edge_index shaped [2,num_edges] even when empty.
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weights = torch.zeros((0,), dtype=torch.float) # Empty tensor for edge weights/distance 0
        edge_attr   = torch.zeros((0,1), dtype=torch.float) # Empty edge attributes [E,1] -> [0,1] 1 is the shape here..
    else:
        # Convert to PyTorch tensors
        # Converts the list of pairs [[i,j],...] shape [E,2] into a tensor then transposing to [2,E]
        edge_index_og = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # Converting the list of Harvesine distnaces into a float tensor of shape [E]
        edge_weights_og = torch.tensor(edge_weights, dtype=torch.float)

        # Normalising [0,1]
        eps = 1e-6 # Avoid divide by zero (you never know)
        den = (edge_weights_og.max() - edge_weights_og.min()).clamp_min(eps)
        norm = (edge_weights_og - edge_weights_og.min()) / den 
        edge_weights_og = 1.0 - norm # inverted so closer nodes have higher weights
    
        # Converting to undirected graph (PyG expects undirected graphs to have both directions explicitly)
        edge_index, edge_attr = to_undirected(
            edge_index_og,
            edge_attr=edge_weights_og,
            num_nodes=num_nodes
        )

    # Logging graph statistics (this will only apply for undirected graphs)
    possible_pairs = num_nodes * (num_nodes - 1) // 2 # Number of possible undirected edges
    density = undirected_edges_count / possible_pairs if possible_pairs > 0 else 0.0 # Proportion of present possible undirected edges
    avg_degree = 2 * undirected_edges_count / num_nodes if num_nodes > 0 else 0.0 # Average number of neighbours per node

    print(
        f"Unique undirected edges: {undirected_edges_count}/{possible_pairs}"
        f"(Density (Proportion of present possible undirected edges) = {density:.3f} "
        f"Average number of neighbours per node = {avg_degree:.2f})"
    )

    # Creating PyTorch Geometric Data object
    graph_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    graph_data.pos = torch.tensor(np.column_stack([lon, lat]), dtype=torch.float)

    gids = scaled_sorted.gid.values.astype(int)
    graph_data.gid = torch.as_tensor(gids, dtype=torch.long)

    # Attach real MSOA names in node order
    if isinstance(msoa_lookup, pd.Series):
        try:
            names = [msoa_lookup.loc[g] for g in gids]
        except KeyError:
            names = list(msoa_lookup.reindex(gids).values)
    elif isinstance(msoa_lookup, dict):
        names = [msoa_lookup.get(int(g), str(int(g))) for g in gids]
    else:
        raise TypeError(
            "msoa_lookup must be a pandas Series indexed by gid or a dict {gid: name}"
        )
    graph_data.msoa_names = names
    graph_data.msoa_to_nid = {name: i for i, name in enumerate(names)}
    
    # MSOA pairs save path
    pairs_save_path = os.path.join(save_path, "graph_msoa_pairs")
    os.makedirs(os.path.dirname(pairs_save_path) or ".", exist_ok=True)
    save_msoa_pairs(graph_data, pairs_save_path + ".csv")

    return graph_data
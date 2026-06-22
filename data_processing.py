import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_feature_data(msoa_features, dont_drop=[]):
    """
    Preprocess a MSOA-level feature DataFrame so it is ready to be handled by GENIE.

    What this does:
        1) Drops a default set of "non-feature" identifier columns (unless explicitly kept)
        2) Detects non-numeric (object) columns and encodes them into integers using LabelEncoder

    Inputs:
        msoa_features:
            pandas DataFrame of MSOA-level features (one row per MSOA).

        dont_drop:
            Optional list of column names that should NOT be dropped even if they are in the default drop list.

    Output:
        Returns a modified pandas DataFrame where:
            - selected columns have been dropped
            - selected object columns have been label-encoded into integer codes
    """

    ###########################################
    # Define which columns we drop by default #
    ###########################################
    # These are columns that are usually identifiers 
    default_drop_columns = ["msoa", "lat_long", "areas_in_msoa", "associated_city"]

    #########################################
    # Handle "dont_drop" overrides properly #
    #########################################
    # If the user passed dont_drop, remove those columns from the drop list.
    # Example: dont_drop=["msoa"] -> we will keep the "msoa" column.
    if dont_drop:
        for item in dont_drop:
            if item in default_drop_columns:
                default_drop_columns.remove(item)

    #########################################
    # Drop columns we consider non-features #
    #########################################
    # Drop columns if they exist; errors="ignore" means missing columns won't raise.
    msoa_features = msoa_features.drop(columns=default_drop_columns, errors="ignore")

    ##################################################
    # Identify non-numeric feature columns to encode #
    ##################################################
    # We exclude either "associated_city" (if present) or "msoa" from encoding,
    # because these are identifiers rather than numeric features.
    exclude_cols = ["associated_city" if "associated_city" in msoa_features.columns else "msoa"]

    # Also exclude anything explicitly protected by dont_drop.
    exclude_cols.extend(dont_drop)

    # Find all object columns (strings/categorical stored as object).
    non_numeric_cols = msoa_features.select_dtypes(include=["object"]).columns

    # Filter out identifier columns (exclude_cols).
    non_numeric_cols = [col for col in non_numeric_cols if col not in exclude_cols]

    # Force these known "code" columns to be encoded as well (even if not detected above).
    # (These columns are likely stored as strings or mixed types.)
    non_numeric_cols.extend([
        "hospital_code_1", "hospital_code_2", "hospital_code_3",
        "university_UKPRN_1", "university_UKPRN_2", "university_UKPRN_3",
        "school_urn_1", "school_urn_2", "school_urn_3"
    ])

    ##########################################
    # Encode categorical columns as integers #
    ##########################################
    # Convert each selected non-numeric column into integer category IDs.
    # LabelEncoder assigns:
    #   the smallest code to the first sorted label, etc.
    for col in non_numeric_cols:
        le = LabelEncoder()
        msoa_features[col] = le.fit_transform(msoa_features[col])

    # Return the processed feature DataFrame.
    return msoa_features

# SUNDERLAND EXAMPLE

# DEFINE THE MSOAS OF THE AREA OF INTEREST
sunderland_msoas = ['E02001791','E02001792','E02001793','E02001794','E02001795','E02001796','E02001798','E02001801','E02001802','E02001803','E02001804','E02001805','E02001806','E02001808','E02001811','E02001812','E02001813','E02001814','E02001816','E02001817','E02001818','E02001819','E02001821']

# LOAD THE RAW EXTRACTED FEATURES OF ALL MSOAS
static_features_data = pd.read_csv("DATA/msoa_feature_data.csv")
sim_data_static_features_big = static_features_data[static_features_data['msoa'].isin(sunderland_msoas)] # Keep all the data of only the MSOAs of interest.
sim_data_static_features_big.to_csv("DATA/static_features_SUNDERLAND.csv",index=False) # Save that data

# PROCESS THE STATIC FEATURES
avanti_msoa_features = pd.read_csv("DATA/static_features_SUNDERLAND.csv")
preprocessed_data = preprocess_feature_data(avanti_msoa_features, dont_drop=["msoa", "lat_long"])
preprocessed_data.to_csv("DATA/processed_msoa_features_SUNDERLAND.csv", index=False)

current_data = pd.read_csv("DATA/static_features_12_10_2025.csv") # The data which we are currently using (in terms of the columns)
new_data = pd.read_csv("DATA/processed_msoa_features_SUNDERLAND.csv")
# Get all the column names except 'msoa' from current_data and drop all other columns in new_data
columns_to_keep = ['msoa'] + [col for col in current_data.columns if col != 'msoa']
filtered_new_data = new_data[columns_to_keep]
# Validate that the columns match
assert list(filtered_new_data.columns) == list(current_data.columns), "Column names do not match!"
# Save the filtered new data to a new CSV file
filtered_new_data.to_csv("DATA/static_features_SUNDERLAND_filtered.csv", index=False)


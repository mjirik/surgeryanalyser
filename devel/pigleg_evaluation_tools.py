import pandas as pd

def rows_with_missing_values_for_stitch(dfs, stitch_id:int):
    """Return a list of rows with missing values."""
    # return dfs[dfs[f"Needle holder stitch {stitch_id} length [m]"].notna()]
    return dfs[dfs[f"Stitch {stitch_id} duration [s]"].notna()]

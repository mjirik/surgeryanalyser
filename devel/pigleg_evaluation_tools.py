import pandas as pd
from typing import List

def rows_with_missing_values_for_stitch(dfs, stitch_id:int):
    """Return a list of rows with missing values."""
    # return dfs[dfs[f"Needle holder stitch {stitch_id} length [m]"].notna()]
    return dfs[dfs[f"Stitch {stitch_id} duration [s]"].notna()]



#  Replace "," with "."
def replace_comma_with_dot(dfs: pd.DataFrame) -> pd.DataFrame:
    # find the columns where its values contain single "," and digits
    dfs.columns[dfs.apply(lambda x: (x.dtype is str) and (x.str.contains(",\d").any()))]
    cols = []
    for col in dfs.columns:
        try:
            if (dfs[col].str.contains(",\d").any()):
                # convert , to . and change the column type to float
                dfs[col] = dfs[col].str.replace(",", ".").astype(float)
                cols.append(col)
        except:
            pass

    return dfs


def convert_bool_to_numeric(dfs:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
    for col in cols:
        uni = dfs[col].unique()
        # print(uni)

        if (len(list(uni)) == 2) and (True in uni) and (False in uni):
            dfs[col] = dfs[col].astype(float)
        else:
            print(f"skipped: {col}")

    return dfs


def convert_object_to_numeric(dfst:pd.DataFrame) -> pd.DataFrame:
    # turn object columns to float
    for col in dfst.columns[dfst.dtypes == object]:
        try:
            dfst[col] = dfst[col].astype(float)
        except Exception as e:
            import traceback
            print(f"Error in {col}")

    return dfst


def new_dataframe_with_one_row_per_stitch(dfs: pd.DataFrame, stitch_count:int = 5, keep_cols=None) -> pd.DataFrame:
    """Return a new dataframe with one row per stitch."""
    # create a new dataframe

    if keep_cols is None:
        keep_cols = keep_cols or ["filename", "annotation_annotation_annotation"]
    list_of_dfs = []
    df_stitches = pd.DataFrame()

    for searched_stitch_id in range(0, stitch_count):
        cols = []
        pattern = f'titch {searched_stitch_id}' + r'( |\b|$)'
        filtered_columns = dfs.filter(regex=pattern)
        print(filtered_columns.columns)
        cols.extend(list(filtered_columns.columns))
        # cols.extend(list(dfs.columns[dfs.columns.str.contains(f"titch {searched_stitch_id} ")]))
        # cols.extend(list(dfs.columns[dfs.columns.str.contains(f"Stitch {searched_stitch_id}")]))
        # cols.extend(list(dfs.columns[dfs.columns.str.contains(f"Stitch{searched_stitch_id}")]))

        cols.extend(list(dfs.filter(regex=f'Knot {searched_stitch_id}' + r'( |\b|$)').columns))
        cols.extend(list(dfs.filter(regex=f'knot {searched_stitch_id}' + r'( |\b|$)').columns))

        cols.extend(list(dfs.filter(regex=f'Piercing {searched_stitch_id}' + r'( |\b|$)').columns))
        cols.extend(list(dfs.filter(regex=f'piercing {searched_stitch_id}' + r'( |\b|$)').columns))

        dfone = dfs[keep_cols + cols]
        dfone = dfone.copy()
        rename = {col: col.replace(f"titch {searched_stitch_id}", "titch") for col in cols }
        dfone = dfone.rename(columns=rename)
        dfone["stitch_id"] = searched_stitch_id
        df_stitches = pd.concat([df_stitches, dfone], axis=0)


    dfst = df_stitches
    dfst = dfst.reset_index(drop=True)
    return dfst



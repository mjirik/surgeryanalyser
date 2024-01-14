import numpy as np
from pathlib import Path
import json
import datetime
import time
import pandas as pd
import plotly.express as px
from typing import Optional, Union
from loguru import logger
try:
    import data_tools
    import media_tools
except ImportError:
    from . import data_tools
    from .media_tools import crop_square




def get_media_path(base_path="."):
    first_result_path = next(Path(base_path).glob("./**/results.json"))
    # results_list = Path(base_path).glob("../**/results.json")
    # print(len(results_list))
    odir = first_result_path.parents[2]
    return odir

def read_one_result(opath:Path)->dict:
    opath = Path(opath)
    results_path = opath / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            loaded_results = json.load(f)
        fn = opath / "meta.json"
        if fn.exists():
            with open(fn) as f:
                meta = json.load(f)

            loaded_results.update(data_tools.flatten_dict(meta))
        else:
            return None
        # loaded_results["Uploaded at"] = uploaded_file.uploaded_at
        # fix typo
        if "Stichtes linearity score" in loaded_results:
            loaded_results["Stitches linearity score"] = loaded_results.pop("Stichtes linearity score")

        if "Stitches linearity score" in loaded_results:
            loaded_results["Stitches linearity score [%]"] = loaded_results["Stitches linearity score"] * 100

        if "Stitches parallelism score" in loaded_results:
            loaded_results["Stitches parallelism score [%]"] = loaded_results["Stitches parallelism score"] * 100

        return loaded_results
    else:
        return None


def calculate_normalization(base_path=".", df:Optional[pd.DataFrame]=None,
                            filename_contains:str="Einzelknopfnaht",
                            )->dict:

    if df is None:

        results_list = list(Path(base_path).glob("./**/results.json"))
        # print(len(results_list))
        odir = results_list[0].parents[2]
        all_df_path = odir / "all_measured_data.csv"

        rows = []
        for i, results_path in enumerate(results_list):

            loaded_results = read_one_result(results_path.parent)
            if loaded_results is not None:
                loaded_results["i"] = i
                rows.append(loaded_results)

        df = pd.DataFrame(rows)
        if "processed_at" not in df.keys():
            df["processed_at"] = datetime.datetime.now().isoformat()

        # df.to_csv(all_df_path)
    else:
        odir = get_media_path(base_path)

    normalization_path = odir / "normalization.json"

    # calculate_normalization
    # filename contain "Einzelknopfnaht"
    dfs = df[df["filename_full"].str.contains(filename_contains, na=False)]
    # ]
    #
    #
    normalization = dfs.median(numeric_only=True, skipna=True)
    norm_dct = normalization.to_dict()
    with open(normalization_path, "w") as f:
        json.dump(norm_dct, f, indent=4)
    # normalization.to_csv(normalization_path, index=False)
    return norm_dct


def make_plot_with_metric(one_record:dict, normalization:dict, cols:list, show:bool=False, filename=None)->px.bar:
    # Assuming df_one is your DataFrame and normalization is a dictionary with normalization factors
    # cols = ["Needle holder visibility [s]", "Forceps visibility [s]"]
    # Assuming df_one is your DataFrame and normalization is a dictionary with normalization factors
    # cols = ["Needle holder visibility [s]", "Forceps visibility [s]"]


    xs = []
    ys = []
    hover_data = []
    distance = []
    for col in cols:
        colname = col if "[%]" in col else col # + f" norm"

        if (col in one_record) and (one_record[col] is not None):
            if "[%]" in col:
                # if "[%]" in col then there is no recalculation
                val = one_record[col]
                logger.debug(f"val kept. val: {val}, {col=}")
            elif (col in normalization) and (normalization[col] is not None):
                val = 100 * one_record[col] / normalization[col]
                logger.debug(f"val normalized. val: {val}, {col=}")
            else:
                continue
        else:
            continue

        xs.append(colname)
        ys.append(val)
        distance.append(abs(100 - val))
        hover_data.append(one_record[col])

    # Include original values in the melted DataFrame
    # df_one_melted = df_one.melt(id_vars=["filename_full"] + hover_data, value_vars=xs, var_name="Tool", value_name="Normalized Value")
    df_one_melted = pd.DataFrame({
        "Metric": xs,
        "Normalized [%]": ys,
        "Value": hover_data,
        "distance": distance
    })

    # # Create the bar plot
    # fig = px.bar(df_one_melted, x="Metric", y="Normalized Value [%]", text='Original Value', hover_data=["Metric"])
    # create the same bar plot just color the columns based on distance of the normalized value from 100
    # fig = px.bar(df_one_melted, x="Metric", y="Normalized Value [%]", text='Original Value', hover_data=["Metric"], color='distance',
    #              color_continuous_scale=["green", "yellow", "red"],
    #              )
    # create the same graph, just hide the colorbar
    fig = px.bar(df_one_melted, x="Metric", y="Normalized [%]", text='Value', hover_data=["Metric"], color='distance',
                 color_continuous_scale=["green", "yellow", "red"],
                 # color_continuous_midpoint=0,
                 )


    fig.update_traces(texttemplate='%{text:.2s}')
    fig.update_coloraxes(showscale=False)
    if show:
        fig.show()
    if filename:
        fig.write_html(filename)
    return fig



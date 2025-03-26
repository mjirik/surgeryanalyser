import pandas
import plotly.express as px
from django.conf import settings
from pathlib import Path
from scipy.optimize import fsolve
import scipy.stats
import pandas as pd
import plotly.figure_factory as ff
from loguru import logger
from . import pigleg_evaluation_tools as pet

from typing import Optional, Union, List
import json



ADD_ADVANCED_STUDENTS_TO_EXPERT = False

with open(Path(__file__).parent / "suggestions.json") as f:
    suggestions = json.load(f)

class StitchDataFrame():
    def __init__(self,
                 relevant_column="mean_movement_annotation",
                 filename_patch="*all_stitches_with_human_annotations*.xlsx",
                 ):
        # normalization_path = settings.MEDIA_ROOT / "generated/normalization.json"
        # normalization_path = "resources/"
        # Check the devel dir to generate these files
        fn = sorted(list((Path(settings.MEDIA_ROOT) / "generated_stats").glob(filename_patch)))[-1]
        # fn = sorted(list(Path("resources").glob("*list_of_all_stitches*.xlsx")))[-1]
        # relevant_column = "AI movement evaluation [%]"
        logger.debug(f"Reading {fn}. {fn.exists()=}")

        dfst = pandas.read_excel(fn)

        annotators = ["Carina", "Ana", "Mira"]
        dfst[relevant_column] = dfst[annotators].mean(axis=1)

        # check devel/2024-09_make_visual_for_students.ipynb
        dfst = add_group_column(dfst, movement_evaluation_col=relevant_column, expert_threshold=3.7894)
        self.dfst:pd.DataFrame = dfst.copy().reset_index(drop=True)
        self.my_dfst:Optional[pd.DataFrame] = None


    def get_suggestions(self, col_name, my_value) -> List[str]:
        col_suggestion = suggestions.get(col_name, [])
        # print(col_suggestion)

        if "thresholds" in col_suggestion:
            thresholds = col_suggestion["thresholds"]
            if my_value < thresholds[0]:
                suggestions_list = col_suggestion["low"]
            elif my_value < thresholds[1]:
                suggestions_list = col_suggestion["moderate"]
            else:
                suggestions_list = col_suggestion["high"]
        else:
            suggestions_list = []

        return suggestions_list

    def get_figs_to_html(self, stitch_id: int, col_names: Optional[List[str]] = None) -> List[str]:
        if col_names is None:

            col_names = [
                "Stitch duration [s]",
                # "Needle holder stitch area presence [%]",
                # "Needle holder stitch median area presence [%]",
                "Needle holder stitch length [m]",
                # "Needle holder stitch visibility [%]",
                "Needle holder stitch velocity above threshold",
                # "Forceps stitch area presence [%]",
                # "Forceps stitch median area presence [%]",
                # "Knot duration [s]",
                # "Needle holder to forceps stitch below threshold [s]"
            ]
        htmls = []
        thresholds = None
        for col_name in col_names:
            my_value = self.my_dfst[self.my_dfst["stitch_id"] == stitch_id][col_name].values[0]

            bin_size = None
            if col_name == "Stitch duration [s]":
                thresholds = [60, 90]
            elif col_name == "Needle holder stitch length [m]":
                bin_size = 0.1
            elif col_name == "Needle holder stitch velocity above threshold":
                thresholds = [17, 25]

            color = "black"
            if thresholds:
                if my_value > thresholds[1]:
                    color = "red"
                elif my_value > thresholds[0]:
                    color = "orange"
                else:
                    color = "green"

            fig = get_distplot(self.dfst, col_name, my_value, annotation_text=f"You={my_value:.2f}",
                               bin_size=bin_size, my_value_color=color,
                               thresholds=thresholds,
                               )
            html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            htmls.append(
                {"title": col_name, "html": html,
                    "suggestions": self.get_suggestions(col_name, my_value),
                 "color": color,
                 }
                )

        return htmls

    def my_values_by_dict(self, results: dict):
        logger.debug(f"{results=}")
        self.results = results
        novy = {}
        novy.update(self.results)

        df_novy = pd.DataFrame(novy, index=[0])

        self.my_dfst = pet.new_dataframe_with_one_row_per_stitch(
            df_novy,
            keep_cols=[],
            # keep_cols=["filename", "annotation_annotation_annotation"]
        )




def add_group_column(dfst, movement_evaluation_col, expert_threshold):
    dfst["Group"] = "expert"
    if ADD_ADVANCED_STUDENTS_TO_EXPERT:
        dfst["Group"][dfst[movement_evaluation_col] < expert_threshold] = "student"
    else:
        dfst["Group"][dfst["done_by_expert"]==False] = "student"
    return dfst


def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x), x0)


def find_threshold(dfst, col_name):
    med = dfst[col_name].median()

    # find the threshold beteween experts and students based on KDE model of each group descriped by done_by_expert
    kde_expert = scipy.stats.gaussian_kde(dfst[dfst["Group"]=='expert'][col_name].dropna())
    kde_student = scipy.stats.gaussian_kde(dfst[dfst["Group"]=='student'][col_name].dropna())
    # find the intersection of the two KDEs
    intersection = findIntersection(kde_expert, kde_student, med)
    return intersection[0]

def get_distplot(dfst, col_name, my_value, annotation_text="You", bin_size:Optional[float]=None, my_value_color="green",
                 thresholds:Optional[list]=None,
                 ):
    fig = ff.create_distplot([dfst[dfst["Group"]=="student"][col_name].dropna(), dfst[dfst["Group"]=='expert'][col_name].dropna()],
                             group_labels=['student', 'expert'],
                             # show_hist=True,
                             show_rug=True,
                             curve_type='kde',
                             # histnorm='probability density'
                             # histnorm='probability'

                             )

    if thresholds and len(thresholds) > 0:
        fig.add_vrect(
            x0=0, x1=thresholds[0],  # uprav si dle potřeby
            fillcolor="green",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )
    if thresholds and len(thresholds) > 1:
        fig.add_vrect(
            x0=thresholds[0], x1=thresholds[1],  # uprav si dle potřeby
            fillcolor="orange",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )
        fig.add_vrect(
            x0=thresholds[1], x1=float("inf"),  # uprav si dle potřeby
            fillcolor="red",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )

    if bin_size is not None:
        # Loop over the traces and update the histogram bin size
        for trace in fig.data:
            if trace.type == 'histogram':
                # Update the xbins property
                trace.xbins = dict(size=bin_size)

    # hide students
    for trace in fig.data:
        if trace.name == 'student':
            trace.visible = 'legendonly'

    # description of x-axis
    fig.update_xaxes(title_text=col_name)
    # legend
    fig.update_layout(legend_title_text='Group')
    # comparison_value
    fig.add_vline(x=my_value, line_width=3, line_dash="dash", line_color=my_value_color, annotation_text=annotation_text, annotation_position="top right")
    # arrow to my_value with text
    # fig.add_annotation(x=my_value, y=0.5, text="mean", showarrow=True, arrowhead=1)
    return fig
    # fig.show()


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
                thresholds = [3., 4.]
                bin_size = 0.1
            elif col_name == "Needle holder stitch velocity above threshold":
                thresholds = [20, 30]

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
                 show_hist:Optional[bool]=False,
                 ):
    fig = ff.create_distplot([dfst[dfst["Group"]=="student"][col_name].dropna(), dfst[dfst["Group"]=='expert'][col_name].dropna()],
                             group_labels=['student', 'expert'],
                             show_hist=show_hist,
                             show_rug=True,
                             curve_type='kde',
                             # histnorm='probability density'
                             # histnorm='probability'

                             )
    # find max value
    experts_max = dfst[dfst["Group"] == 'expert'][col_name].max()
    experts_max = max(experts_max, my_value)
    experts_max = max(experts_max, thresholds[1] + 0.5*(thresholds[1] - thresholds[0]))

    if thresholds and len(thresholds) > 0:
        pass
    if thresholds and len(thresholds) > 1:
        logger.debug(f"{thresholds=}")
        fig.add_vrect(
            x0=0, x1=thresholds[0],  # uprav si dle potřeby
            fillcolor="green",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )
        fig.add_vrect(
            x0=thresholds[0], x1=thresholds[1],  # uprav si dle potřeby
            fillcolor="yellow",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )
        logger.debug(f"{thresholds[0]=}, {experts_max=}")
        fig.add_vrect(
            x0=thresholds[1],
            x1 = experts_max,
            fillcolor="red",
            opacity=0.15,
            layer="below",  # vrstvení pod křivkami
            line_width=0,
        )

    if show_hist:
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


STITCH_DATA_FRAME = StitchDataFrame(
    relevant_column="mean_movement_annotation",
    filename_patch="*all_stitches_with_human_annotations*.xlsx",
)


import numpy as np

def set_overall_score(serverfile) -> float:
    """
    Set overall score from the serverfile.
    :param serverfile: UploadedFile object
    :return: overall score
    """
    loaded_results = load_results(serverfile)
    per_stitch_report = load_per_stitch_data(loaded_results, serverfile)

    scores = []

    for record in per_stitch_report:
        if "ai_movement_evaluation" in record and record["ai_movement_evaluation"] is not None:
            scores.append(record["ai_movement_evaluation"])


    scores = np.array(scores)
    if len(scores) > 0:
        score = np.mean(scores)
    else:
        score = None

    serverfile.score = score
    serverfile.save()





def load_per_stitch_data(loaded_results, serverfile):
    per_stitch_report = []
    if loaded_results is not None:

        for i in range(int(serverfile.stitch_count)):
            logger.debug(f"Stitch {i}")
            try:
                STITCH_DATA_FRAME.my_values_by_dict(loaded_results)
                graphs_html = STITCH_DATA_FRAME.get_figs_to_html(i)

                # find image
                needle_holder_compare_heatmaps_image = None
                looking_for = f"needle_holder_compare_heatmaps_stitch {int(i)}"
                logger.debug(f"{looking_for=}")
                for image in serverfile.bitmapimage_set.all():
                    logger.debug(f"  {image.bitmap_image.name}")
                    if looking_for in str(image.bitmap_image.name):
                        logger.debug(f"     Found {image.bitmap_image.name}")
                        needle_holder_compare_heatmaps_image = image
                        break

                per_stitch_report.append({
                    "stitch_id": i,
                    "advices": prepare_advices(loaded_results, i),
                    "ai_movement_evaluation": loaded_results.get(f"AI movement evaluation stitch {i} [%]", None),
                    'graphs_html': graphs_html,
                    "needle_holder_compare_heatmaps_image": needle_holder_compare_heatmaps_image,
                })

            except Exception as e:
                logger.error(f"Error in processing stitch {i}. {e}")
                logger.error(traceback.format_exc())
    return per_stitch_report


def load_results(serverfile) -> dict:
    fn_results = Path(serverfile.outputdir) / "results.json"
    results = {}
    loaded_results = None
    if fn_results.exists():
        with open(fn_results) as f:
            loaded_results = json.load(f)
    return loaded_results



def prepare_advices(results: dict, stitch_id:int) -> list:
    # logger.debug(f"{results=}")
    advices = []

    # set varibale fn to function which will return true if the value will be lower then some threshold
    fn = lambda x, threshold: x < threshold

    adv0 = "Try to keep your instruments closer to the incision to avoid unnecessary large movements."
    adv1 = "Try to move the instruments at a constant speed. Careless, rapid movements can result in unnecessary large movements. "

    rules_and_advices = [
        (f"Stitch {stitch_id} duration [s]", is_hi_than, 65.30, "The stitch duration is too long. ", "Try to make your movements more smooth and precise. "),
        # ),(# "Needle holder stitch area presence [%]" ,
        #     (f"Needle holder stitch {stitch_id} median area presence [%]", is_lo_than, 91.43, "The needle holder visibility in area around stitch is too low. " + adv0),
        (f"Needle holder stitch {stitch_id} median area presence [%]", is_lo_than, 70., "The needle holder visibility in area around stitch is too low. ", adv0),  # arbitrary value
        (f"Needle holder stitch {stitch_id} length [m]", is_hi_than, 2.67, "The needle holder trajectory length is too long. ", adv0),
        (f"Needle holder stitch {stitch_id} visibility [%]", is_lo_than, 84.25, "The needle holder visibility is too low. ", adv0),
        (f"Needle holder stitch {stitch_id} velocity above threshold" , is_hi_than, 20.18, "Too much sudden moves of the needle holder detected. ", adv1),
        (f"Forceps stitch {stitch_id} median area presence [%]" , is_hi_than, 86.23 , "The forceps visibility in area around stitch is too low. ", adv0)
    ]
    # "Knot duration [s]",
    # "Needle holder to forceps stitch below threshold [s]"

    advice_reason = {}
    for rule_and_advice in rules_and_advices:
        key, fn, threshold, reason, advice = rule_and_advice
        if key in results:
            if fn(results[key], threshold):
                if advice in advice_reason:
                    advice_reason[advice].append(reason)
                else:
                    advice_reason[advice] = [reason]
        else:
            logger.warning(f"Key '{key}' not found in results")

    # Put the advice on one line fallowed by the reasons.
    for advice in advice_reason:
        advices.append(advice + " " + " ".join(advice_reason[advice]))
    return advices

def is_lo_than(value, threshold):
    return value < threshold

def is_hi_than(value, threshold):
    return value > threshold

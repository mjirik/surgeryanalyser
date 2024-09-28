import pandas as pd
import numpy as np
import pickle
from loguru import logger
from pathlib import Path
try:
    import pigleg_evaluation_tools as pet
except ImportError:
    from . import pigleg_evaluation_tools as pet


PREDICTION_MODEL_PATH = Path("model_best_SVR.pkl")
PREDICTION_MODEL = None


class MovementEvaluation:
    def __init__(self, results:dict, meta:dict, mediafile_path:Path):
        self.results = results
        self.meta = meta
        self.mediafile_path = Path(mediafile_path)

        novy = {}
        novy.update(self.meta)
        novy.update(self.results)
        novy["filename"] = self.mediafile_path.name

        df_novy = pd.DataFrame(novy, index=[0])

        self.dfst = pet.new_dataframe_with_one_row_per_stitch(
            df_novy,
            keep_cols=[],
            # keep_cols=["filename", "annotation_annotation_annotation"]
        )



    def evaluate(self):
        self.dfst = movement_evaluation_prediction(self.dfst)
        predictions = list(self.dfst["prediction"])
        stitch_ids = list(self.dfst["stitch_id"])
        logger.debug(f'{predictions=}')
        logger.debug(f'{stitch_ids=}')

        additional_results = {}
        for stitch_id, prediction in zip(stitch_ids, predictions):
            # limit prediction to range  [0, 5]
            prediction = max(0, min(5, prediction))
            additional_results[f"AI movement evaluation {stitch_id}"] = prediction
        return additional_results



def movement_evaluation_prediction(dfst: pd.DataFrame) -> pd.DataFrame:
    """Evaluation of the movement of instrument tip."""
    global PREDICTION_MODEL

    if PREDICTION_MODEL is None:
        logger.debug("Loading prediction model")
        logger.debug(f"{PREDICTION_MODEL_PATH=}")

        with open(PREDICTION_MODEL_PATH, "rb") as f:
            PREDICTION_MODEL = pickle.load(f)
    model = PREDICTION_MODEL
    clf = model["model"]
    data_cols = model["data_cols"]
    sample_id_cols = model["sample_id_cols"]

    predicted_columns = model["predicted_columns"]
    logger.debug(dfst.shape)
    dfst_nna = dfst.dropna(subset=data_cols + sample_id_cols + predicted_columns
                           ).reset_index()
    dfst_nna = dfst_nna.copy()
    logger.debug(dfst_nna.shape)
    logger.debug(dfst_nna[data_cols].shape)
    clf.predict(dfst_nna[data_cols])
    dfst_nna["prediction"] = clf.predict(dfst_nna[data_cols])
    return dfst_nna

import json
from pathlib import Path

from typing import Dict, Set, Tuple, Callable

import numpy as np
import pandas as pd
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt


ann_path = Path("./data/coco/zero-shot/instances_val2017_all_2.json")
pre_path = Path("./output/try_prediction_result.json")


def load_prediction(
        prediction_path: Path = pre_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(prediction_path, "r") as f:
        predictions_json = json.load(f)
    # predictions_json = pd.read_json(prediction_path, orient='records')

    prediction_df = pd.DataFrame.from_records(predictions_json)
    # print('prediction_df', prediction_df)
    prediction_df[["xmin", "ymin", "xmax", "ymax"]] = prediction_df["bbox"].tolist()
    # print('prediction_df', prediction_df)

    prediction_df.reset_index(inplace=True)
    prediction_df.rename(
        columns={"index": "pred_id", "category_id": "label_id"}, inplace=True
    )
    prediction_df = prediction_df[
        ["pred_id", "image_id", "label_id", "xmin", "ymin", "xmax", "ymax", "score"]
    ]

    return prediction_df

def load_dataset(
    annotations_path: Path = ann_path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the COCO style json dataset and transform it into convenient DataFrames

    :return (images_df, targets_df):
        images_df: Columns "image_id" and "file_name"
        targets_df: Columns
            "target_id", "image_id", "xmin", "ymin", "xmax", "ymax", "label_id"
    """

    with open(annotations_path, "r") as f:
        targets_json = json.load(f)

    images_df = pd.DataFrame.from_records(targets_json["images"])
    images_df.rename(columns={"id": "image_id"}, inplace=True)
    images_df = images_df[["image_id", "file_name"]]

    targets_df = pd.DataFrame.from_records(targets_json["annotations"])
    targets_df[["xmin", "ymin", "w", "h"]] = targets_df["bbox"].tolist()
    targets_df["xmax"] = targets_df["xmin"] + targets_df["w"]
    targets_df["ymax"] = targets_df["ymin"] + targets_df["h"]
    targets_df.reset_index(inplace=True)
    targets_df.rename(
        columns={"index": "target_id", "category_id": "label_id"}, inplace=True
    )
    targets_df = targets_df[
        ["target_id", "image_id", "label_id", "xmin", "ymin", "xmax", "ymax"]
    ]

    return images_df, targets_df

TARGETS_DF_COLUMNS = [
    "target_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
]
PREDS_DF_COLUMNS = [
    "pred_id",
    "image_id",
    "label_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
]
ERRORS_DF_COLUMNS = ["pred_id", "target_id", "error_type"]

BACKGROUND_IOU_THRESHOLD = 0.1
FOREGROUND_IOU_THRESHOLD = 0.5


class ErrorType:
    OK = "correct"  # pred -> IoU > foreground; target_label == pred_label; highest score
    CLS = "classification"  # pred -> IoU > foreground; target_label != pred_label
    LOC = "localization"  # pred -> background < IoU < foreground; target_label == pred_label
    CLS_LOC = "cls & loc"  # pred -> background < IoU < foreground; target_label != pred_label
    DUP = "duplicate"  # pred -> background < IoU < foreground; target_label != pred_label
    BKG = "background"  # pred -> IoU > foreground; target_label == pred_label; no highest score
    MISS = "missed"  # target -> No pred with Iou > background


def classify_predictions_errors(
    targets_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_background: float = BACKGROUND_IOU_THRESHOLD,
    iou_foreground: float = FOREGROUND_IOU_THRESHOLD,
) -> pd.DataFrame:
    """Classify predictions

    We assume model is right as much as possible. Thus, in case of doubt
    (i.e matching two targets), a prediction will be first considered
    ErrorType.LOC before ErrorType.CLS.

    The error definition credit belongs to the following paper (refer to it for
    conceptual details):
        TIDE: A General Toolbox for Identifying Object Detection Errors
        https://arxiv.org/abs/2008.08115

    :param targets_df: DataFrame with all targets for all images with TARGETS_DF_COLUMNS.
    :param preds_df: DataFrame with all predictions for all images with PREDS_DF_COLUMNS.
    :param iou_background: Minimum IoU for a prediction not to be considered background.
    :param iou_foreground: Minimum IoU for a prediction to be considered foreground.
    :return errors_df: DataFrame with all error information with ERRORS_DF_COLUMNS
    """

    # Provide clarity on expectations and avoid confusing errors down the line
    assert (set(TARGETS_DF_COLUMNS) - set(targets_df.columns)) == set()
    assert (set(PREDS_DF_COLUMNS) - set(preds_df.columns)) == set()

    pred2error = dict()  # {pred_id: ErrorType}
    target2pred = (
        dict()
    )  # {target_id: pred_id}, require iou > iou_foreground & max score
    pred2target = dict()  # {pred_id: target_id}, require iou >= iou_background
    missed_targets = set()  # {target_id}

    # Higher scoring preds take precedence when multiple fulfill criteria
    preds_df = preds_df.sort_values(by="score", ascending=False)

    for image_id, im_preds_df in preds_df.groupby("image_id"):
        # Need to reset index to access dfs with same idx we access
        #   IoU matrix down the line
        im_targets_df = targets_df.query("image_id == @image_id").reset_index(
            drop=True
        )
        im_preds_df = im_preds_df.reset_index(drop=True)

        if im_targets_df.empty:
            pred2error = {**pred2error, **_process_empty_image(im_preds_df)}
        else:
            iou_matrix, iou_label_match_matrix = _compute_iou_matrices(
                im_targets_df, im_preds_df
            )

            # Iterate over all predictions. Higher scores first
            for pred_idx in range(len(im_preds_df)):
                match_found = _match_pred_to_target_with_same_label(
                    pred_idx,
                    pred2error,
                    pred2target,
                    target2pred,
                    iou_label_match_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )
                if match_found:
                    continue

                _match_pred_wrong_label_or_background(
                    pred_idx,
                    pred2error,
                    pred2target,
                    iou_matrix,
                    im_targets_df,
                    im_preds_df,
                    iou_background,
                    iou_foreground,
                )

    missed_targets = _find_missed_targets(targets_df, pred2target)
    errors_df = _format_errors_as_dataframe(
        pred2error, pred2target, missed_targets
    )
    return errors_df[list(ERRORS_DF_COLUMNS)]


def _process_empty_image(im_preds_df: pd.DataFrame) -> Dict[int, str]:
    """In an image without targets, all predictions represent a background error"""
    return {
        pred_id: ErrorType.BKG for pred_id in im_preds_df["pred_id"].unique()
    }


def _compute_iou_matrices(
    im_targets_df: pd.DataFrame, im_preds_df: pd.DataFrame
) -> Tuple[np.array, np.array]:
    """Compute IoU matrix between all targets and preds in the image

    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :return:
        iou_matrix: Matrix of size (n_targets, n_preds) with IoU between all
            targets & preds
        iou_label_match_matrix: Same as `iou_matrix` but 0 for all target-pred
            pair with different labels (i.e. IoU kept only if labels match).
    """
    # row indexes point to targets, column indexes to predictions
    iou_matrix = iou_matrix = torchvision.ops.box_iou(
        torch.from_numpy(
            im_targets_df[["xmin", "ymin", "xmax", "ymax"]].values
        ),
        torch.from_numpy(im_preds_df[["xmin", "ymin", "xmax", "ymax"]].values),
    ).numpy()

    # boolean matrix with True iff target and pred have the same label
    label_match_matrix = (
        im_targets_df["label_id"].values[:, None]
        == im_preds_df["label_id"].values[None, :]
    )
    # IoU matrix with 0 in all target-pred pairs that have different label
    iou_label_match_matrix = iou_matrix * label_match_matrix
    return iou_matrix, iou_label_match_matrix


def _match_pred_to_target_with_same_label(
    pred_idx: int,
    pred2error: Dict[int, str],
    pred2target: Dict[int, int],
    target2pred: Dict[int, int],
    iou_label_match_matrix: np.array,
    im_targets_df: pd.DataFrame,
    im_preds_df: pd.DataFrame,
    iou_background: float,
    iou_foreground: float,
) -> bool:
    """Try to match `pred_idx` to a target with the same label and identify error (if any)

    If there is a match `pred2error`, `pred2target` and (maybe) `target2pred`
    are modified in place.
    Possible error types found in this function:
        ErrorType.OK, ErrorType.DUP, ErrorType.LOC

    :param pred_idx: Index of prediction based on score (index 0 is maximum score for image).
    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with iou above background)
    :param target2pred: Dict mapping target_id to pred_id to pred considered correct (if any).
    :param iou_label_match_matrix: Matrix with size [n_targets, n_preds] with IoU between all preds
        and targets that share label (i.e. IoU = 0 if there is a label missmatch).
    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :param iou_background: Minimum IoU to consider a pred not background for target.
    :param iou_foreground: Minimum IoU to consider a pred foreground for a target.
    :return matched: Whether or not there was a match and we could identify the pred error.
    """
    # Find highest overlapping target for pred processed
    target_idx = np.argmax(iou_label_match_matrix[:, pred_idx])
    iou = np.max(iou_label_match_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    matched = False
    if iou >= iou_foreground:
        pred2target[pred_id] = target_id
        # Check if another prediction is already the match for target to
        #   identify duplicates
        if target2pred.get(target_id) is None:
            target2pred[target_id] = pred_id
            pred2error[pred_id] = ErrorType.OK
        else:
            pred2error[pred_id] = ErrorType.DUP
        matched = True

    elif iou_background <= iou < iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.LOC
        matched = True
    return matched


def _match_pred_wrong_label_or_background(
    pred_idx: int,
    pred2error: Dict[int, str],
    pred2target: Dict[int, int],
    iou_matrix: np.array,
    im_targets_df: pd.DataFrame,
    im_preds_df: pd.DataFrame,
    iou_background: float,
    iou_foreground: float,
) -> None:
    """Try to match `pred_idx` to a target (with different label) and identify error

    If there is a match `pred2error` and  (maybe) `pred2target` are modified in place.
    Possible error types found in this function:
        ErrorType.BKG, ErrorType.CLS, ErrorType.CLS_LOC

    :param pred_idx: Index of prediction based on score (index 0 is maximum score for image).
    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with iou above background)
    :param target2pred: Dict mapping target_id to pred_id to pred considered correct (if any).
    :param iou: Matrix with size [n_targets, n_preds] with IoU between all preds and targets.
    :param im_targets_df: DataFrame with targets for the image being processed.
    :param im_preds_df: DataFrame with preds for the image being processed.
    :param iou_background: Minimum IoU to consider a pred not background for target.
    :param iou_foreground: Minimum IoU to consider a pred foreground for a target.
    """
    # Find highest overlapping target for pred processed
    target_idx = np.argmax(iou_matrix[:, pred_idx])
    iou = np.max(iou_matrix[:, pred_idx])
    target_id = im_targets_df.at[target_idx, "target_id"]
    pred_id = im_preds_df.at[pred_idx, "pred_id"]

    if iou < iou_background:
        pred2error[pred_id] = ErrorType.BKG

    # preds with correct label do not get here. Thus, no need to check if label
    #   is wrong
    elif iou >= iou_foreground:
        pred2target[pred_id] = target_id
        pred2error[pred_id] = ErrorType.CLS
    else:
        # No match to target, as we cannot be sure model was remotely close to
        #   getting it right
        pred2error[pred_id] = ErrorType.CLS_LOC


def _find_missed_targets(
    im_targets_df: pd.DataFrame, pred2target: Dict[int, int]
) -> Set[int]:
    """Find targets in the processed image that were not matched by any prediction

    :param im_targets_df: DataFrame with targets for the image being processed.
    :param pred2target: Dict mapping pred_id to target_id (if match found with
        iou above background)
    :return missed_targets: Set of all the target ids that were missed

    """
    matched_targets = [t for t in pred2target.values() if t is not None]
    missed_targets = set(im_targets_df["target_id"]) - set(matched_targets)
    return missed_targets


def _format_errors_as_dataframe(
    pred2error: Dict[int, str],
    pred2target: Dict[int, int],
    missed_targets: Set[int],
) -> pd.DataFrame:
    """Use the variables used to classify errors to format them in a ready to use DataFrame

    :param pred2error: Dict mapping pred_id to error type.
    :param pred2target: Dict mapping pred_id to target_id (if match found with
        iou above background)
    :param missed_targets: Set of all the target ids that were missed
    :return: DataFrame with columns ERRORS_DF_COLUMNS
    """
    errors_df = pd.DataFrame.from_records(
        [
            {"pred_id": pred_id, "error_type": error}
            for pred_id, error in pred2error.items()
        ]
    )
    errors_df["target_id"] = None
    errors_df.set_index("pred_id", inplace=True)
    for pred_id, target_id in pred2target.items():
        errors_df.at[pred_id, "target_id"] = target_id

    missed_df = pd.DataFrame(
        {
            "pred_id": None,
            "error_type": ErrorType.MISS,
            "target_id": list(missed_targets),
        }
    )
    errors_df = pd.concat(
        [errors_df.reset_index(), missed_df], ignore_index=True
    ).astype(
        {"pred_id": float, "target_id": float, "error_type": pd.StringDtype()}
    )
    return errors_df

class MyMeanAveragePrecision:
    """Wrapper for the torchmetrics MeanAveragePrecision exposing API we need"""

    def __init__(self, foreground_threshold):
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.map = MeanAveragePrecision(
            iou_thresholds=[foreground_threshold]
        ).to(self.device)

    def __call__(self, targets_df, preds_df):
        targets, preds = self._format_inputs(targets_df, preds_df)
        self.map.update(preds=preds, target=targets)
        result = self.map.compute()["map"].item()
        self.map.reset()
        return result

    def _format_inputs(self, targets_df, preds_df):
        image_ids = set(targets_df["image_id"]) | set(preds_df["image_id"])
        targets, preds = [], []
        for image_id in image_ids:
            im_targets_df = targets_df.query("image_id == @image_id")
            im_preds_df = preds_df.query("image_id == @image_id")
            targets.append(
                {
                    "boxes": torch.as_tensor(
                        im_targets_df[["xmin", "ymin", "xmax", "ymax"]].values,
                        dtype=torch.float32,
                    ).to(self.device),
                    "labels": torch.as_tensor(
                        im_targets_df["label_id"].values, dtype=torch.int64
                    ).to(self.device),
                }
            )
            preds.append(
                {
                    "boxes": torch.as_tensor(
                        im_preds_df[["xmin", "ymin", "xmax", "ymax"]].values,
                        dtype=torch.float32,
                    ).to(self.device),
                    "labels": torch.as_tensor(
                        im_preds_df["label_id"].values, dtype=torch.int64
                    ).to(self.device),
                    "scores": torch.as_tensor(
                        im_preds_df["score"].values, dtype=torch.float32
                    ).to(self.device),
                }
            )
        return targets, preds

def calculate_error_impact(
    metric_name: str,
    metric: Callable,
    errors_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    preds_df: pd.DataFrame,
) -> Dict[str, float]:
    """Calculate the `metric` and the independant impact each error type has on it

    Impact is defined as the (metric_after_fixing - metric_before_fixing).
    Note that all error impacts and the metric will not add to 1. Nonetheless,
    the errors (and fixes) are defined in such a way that applying all fixes
    would end up with a perfect metric score.

    :param metric_name: Name of the metric to display for logging purposes.
    :param metric: Callable that will be called as metric(targets_df, preds_df)
        and returns a float.
    :param errors_df: DataFrame with error classification for all preds and targets
    :param targets_df: DataFrame with the targets.
    :param preds_df: DataFrame with the predictions.
    :return impact: Dictionary with one key for the metric without fixing and
        one for each error type.
    """

    ensure_consistency(errors_df, targets_df, preds_df)

    metric_values = {
        ErrorType.CLS: metric(*fix_cls_error(errors_df, targets_df, preds_df)),
        ErrorType.LOC: metric(*fix_loc_error(errors_df, targets_df, preds_df)),
        ErrorType.CLS_LOC: metric(
            *fix_cls_loc_error(errors_df, targets_df, preds_df)
        ),
        ErrorType.DUP: metric(*fix_dup_error(errors_df, targets_df, preds_df)),
        ErrorType.BKG: metric(*fix_bkg_error(errors_df, targets_df, preds_df)),
        ErrorType.MISS: metric(
            *fix_miss_error(errors_df, targets_df, preds_df)
        ),
    }

    # Compute the metric on the actual results
    baseline_metric = metric(targets_df, preds_df)
    # Calculate the difference (impact) in the metric when fixing each error
    impact = {
        error: (error_metric - baseline_metric)
        for error, error_metric in metric_values.items()
    }
    impact[metric_name] = baseline_metric
    return impact


def ensure_consistency(
    errors_df: pd.DataFrame, targets_df: pd.DataFrame, preds_df: pd.DataFrame
):
    """Make sure that all targets are preds are accounted for in errors"""
    target_ids = set(targets_df["target_id"])
    pred_ids = set(preds_df["pred_id"])

    error_target_ids = set(errors_df.query("target_id.notnull()")["target_id"])
    error_pred_ids = set(errors_df.query("pred_id.notnull()")["pred_id"])

    if not target_ids == error_target_ids:
        raise ValueError(
            f"Missing target IDs in error_df: {target_ids - error_target_ids}"
        )

    if not pred_ids == error_pred_ids:
        raise ValueError(
            f"Missing pred IDs in error_df: {pred_ids - error_pred_ids}"
        )


def fix_cls_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_correcting_and_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.CLS
    )


def fix_loc_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_correcting_and_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.LOC
    )


def fix_cls_loc_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.CLS_LOC
    )


def fix_bkg_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.BKG
    )


def fix_dup_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _fix_by_removing_preds(
        errors_df, targets_df, preds_df, ErrorType.DUP
    )


def fix_miss_error(
    errors_df, targets_df, preds_df
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fix missed targets by removing them

    Missed targets is the only type of errors that deals with targets rather
    than predictions

    :return: Fixed (`targets_df`, `errors_df`)
    """
    ensure_consistency(errors_df, targets_df, preds_df)

    targets_df = targets_df.merge(
        # Need to filter rest of errors or multi prediction per target makes
        #   target_df bigger
        # errors_df.query("error_type == @ErrorType.MISS"),
        errors_df.query("error_type in ['missed']"),
        on="target_id",
        how="left",
    ).query("error_type.isnull()")
    return targets_df[TARGETS_DF_COLUMNS], preds_df


def _fix_by_correcting_and_removing_preds(
    errors_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    error_type: ErrorType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correct predictions of `error_type` of unmatched target and remove the rest

    CLS and LOC errors are matched to targets. To assess their impact, we
    correct the highest scoring prediction for an unmatched target
    (no OK error for it).
        - For CLS, we set the label to the right one.
        - For LOC, we set the bounding box to match perfectly with the target's.

    The non-corrected predictions of `error_type` are removed from `preds_df`.

    The idea is to assess what happened if instead of missing a target due to an
    incorrect prediction, we would have had a correct one instead. The ones that
    are not highest-scoring for target would have been duplicates, so we remove
    them.

    :return: Fixed (`targets_df`, `errors_df`)
    """

    assert error_type in {
        ErrorType.CLS,
        ErrorType.LOC,
    }, f"error_type='{error_type}'"
    ensure_consistency(errors_df, targets_df, preds_df)

    cols_to_correct = {
        ErrorType.CLS: ["label_id"],
        ErrorType.LOC: ["xmin", "ymin", "xmax", "ymax"],
    }[error_type]

    # Add matched targets to relevant preds and sort so highest scoring is first.
    preds_df = (
        preds_df.merge(
            errors_df.query(
                "error_type in ['correct', 'classification', 'localization']"
                #"error_type in [@ErrorType.OK, @ErrorType.CLS, @ErrorType.LOC]"
            ),
            on="pred_id",
            how="left",
        )
        .merge(
            targets_df[["target_id"] + cols_to_correct],
            on="target_id",
            how="left",
            suffixes=("", "_target"),
        )
        .sort_values(by="score", ascending=False)
    )

    to_correct = preds_df["error_type"].eq(error_type)
    target_cols = [col + "_target" for col in cols_to_correct]
    preds_df.loc[to_correct, cols_to_correct] = preds_df.loc[
        to_correct, target_cols
    ].values

    to_drop = []
    for _, target_df in preds_df.groupby("target_id"):
        if target_df["error_type"].eq(ErrorType.OK).any():
            # If target has a correct prediction, drop all predictions of `error_type`
            to_drop += target_df.query("error_type == @error_type")[
                "pred_id"
            ].tolist()
        elif (
            target_df["error_type"].eq(error_type).any() and len(target_df) > 1
        ):
            # If target unmatched, drop all predictions of `error_type` that are
            #   not highest score
            to_keep = target_df["pred_id"].iloc[0]
            to_drop += target_df.query(
                "error_type == @error_type and pred_id != @to_keep"
            )["pred_id"].tolist()
    return (
        targets_df,
        preds_df.query("pred_id not in @to_drop")[PREDS_DF_COLUMNS],
    )


def _fix_by_removing_preds(
    errors_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    error_type: ErrorType,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fix the `error_type` by removing the predictions assigned to that error

    This is applicable to:
        - ErrorType.CLS_LOC and ErrorType.BKG because there is no target we
            could match it and be sure the model was "intending" to predict that.
        - ErrorType.DUP by definition.

    :return: Fixed (`targets_df`, `errors_df`)
    """

    assert error_type in {
        ErrorType.CLS_LOC,
        ErrorType.BKG,
        ErrorType.DUP,
    }, f"error_type='{error_type}'"
    ensure_consistency(errors_df, targets_df, preds_df)

    preds_df = preds_df.merge(errors_df, on="pred_id", how="left").query(
        "error_type != @error_type"
    )
    return targets_df, preds_df[PREDS_DF_COLUMNS]

images_df, targets_df = load_dataset()
preds_df = load_prediction()
errors_df = classify_predictions_errors(targets_df, preds_df)
errors_df["error_type"].value_counts()

impact = calculate_error_impact(
    "mAP@50",
    MyMeanAveragePrecision(foreground_threshold=FOREGROUND_IOU_THRESHOLD),
    errors_df,
    targets_df,
    preds_df
)

print(impact)

order = ["mAP@50", "classification", "localization", "cls & loc", "duplicate", "background", "missed"]
labels = ["mAP@50", "CLS", "LOC", "CLS & LOC", "DUP", "BKG", "MISS"]
x = range(len(impact))
y = [impact[o] for o in order]
plt.bar(x, y, color="gray", tick_label=labels, zorder=10)
plt.grid(alpha=0.5, zorder=1)
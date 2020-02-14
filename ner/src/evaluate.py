import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="JSON file of predictions.")
    parser.add_argument("--gold_standard", type=str, required=True,
                        help="JSON file of the gold standard annotations.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory in which to save the results.")
    return parser.parse_args()


def evaluate(predictions, gold_standard, outdir):
    stats = {"strict": {"precision": [],
                        "recall": [],
                        "f1": []},
             "lenient": {"precision": [],
                         "recall": [],
                         "f1": []}
             }

    all_labels = {"strict": {}, "lenient": {}}
    # Compute the evaluation metrics for the individual documents.
    for (doc_id, preds) in predictions.items():
        golds = gold_standard[doc_id]
        # Strict evaluation criteria
        strict_labels = list(add_labels_to_predictions_strict(preds, golds))
        all_labels["strict"][doc_id] = strict_labels
        prec, rec, f1 = compute_prec_rec_f1_from_labels(strict_labels)
        stats["strict"]["precision"].append(prec)
        stats["strict"]["recall"].append(rec)
        stats["strict"]["f1"].append(f1)
        del prec, rec, f1

        # Lenient evaluation criteria
        lenient_labels = list(add_labels_to_predictions_lenient(preds, golds))
        all_labels["lenient"][doc_id] = lenient_labels
        prec, rec, f1 = compute_prec_rec_f1_from_labels(lenient_labels)
        stats["lenient"]["precision"].append(prec)
        stats["lenient"]["recall"].append(rec)
        stats["lenient"]["f1"].append(f1)
        del prec, rec, f1

    # Compute the global metrics
    sprec, srec, sf1 = compute_prec_rec_f1_from_labels(all_labels["strict"])
    lprec, lrec, lf1 = compute_prec_rec_f1_from_labels(all_labels["lenient"])

    metrics_outfile = os.path.join(outdir, "metrics.txt")
    with open(metrics_outfile, 'w') as outF:
        outF.write("Global Results\n")
        outF.write("--------------\n")
        outF.write(f"{'': <10} {'Precision': <10} {'Recall': <10} {'F1': <10}\n")  # noqa
        outF.write(f"{'Strict': <10} {sprec: <10.2f} {srec: <10.2f} {sf1: <10.2f}\n")  # noqa
        outF.write(f"{'Lenient': <10} {lprec: <10.2f} {lrec: <10.2f} {lf1: <10.2f}\n")  # noqa

    stp, sfp, sfn = count_tp_fp_fn(all_labels["strict"])
    ltp, lfp, lfn = count_tp_fp_fn(all_labels["lenient"])
    pred_stats_outfile = os.path.join(outdir, "prediction_stats.txt")
    with open(pred_stats_outfile, 'w') as outF:
        outF.write("Predictions\n")
        outF.write("-----------\n")
        outF.write(f"{'': <10} {'TP': <10} {'FP': <10} {'FN': <10} {'Total': <10}\n")  # noqa
        outF.write(f"{'Strict': <10} {stp: <10} {sfp: <10} {sfn: <10} {stp+sfp: <10}\n")  # noqa
        outF.write(f"{'Lenient': <10} {ltp: <10} {lfp: <10} {lfn: <10} {ltp+lfp: <10}\n")  # noqa

    # Plot the distribution of metrics across documents.
    plot_eval_stats(stats["strict"], color="skyblue",
                    save_to=os.path.join(outdir, "strict.png"))
    plot_eval_stats(stats["lenient"], color="lightsalmon",
                    save_to=os.path.join(outdir, "lenient.png"))

    strict_outfile = os.path.join(outdir, "predictions_labeled_strict.json")
    with open(strict_outfile, 'w') as outF:
        json.dump(all_labels["strict"], outF)

    lenient_outfile = os.path.join(outdir, "predictions_labeled_lenient.json")
    with open(lenient_outfile, 'w') as outF:
        json.dump(all_labels["lenient"], outF)


def add_labels_to_predictions_strict(preds, golds, strict=True):
    """
    Returns predictions and missed gold annotations with
    a "label" key, which can take one of "TP", "FP", or "FN".
    """
    pred_spans = {(s["start"], s["end"]) for s in preds}
    gold_spans = {(s["start"], s["end"]) for s in golds}
    outspans = []
    for pred in preds:
        pred_span = (pred["start"], pred["end"])
        if pred_span in gold_spans:
            label = "TP"
        else:
            label = "FP"
        pred_copy = dict(pred)
        pred_copy["label"] = label
        outspans.append(pred_copy)

    for gold in golds:
        if (gold["start"], gold["end"]) not in pred_spans:
            gold_copy = dict(gold)
            gold_copy["label"] = "FN"
            outspans.append(gold_copy)

    return sorted(outspans, key=lambda s: s["start"])


def add_labels_to_predictions_lenient(preds, golds):
    pred_idxs = {i for p in preds for i in range(p["start"], p["end"])}
    gold_idxs = {i for g in golds for i in range(g["start"], g["end"])}
    outspans = []
    for pred in preds:
        if pred["start"] in gold_idxs or pred["end"] in gold_idxs:
            label = "TP"
        else:
            label = "FP"
        pred_copy = dict(pred)
        pred_copy["label"] = label
        outspans.append(pred_copy)

    for gold in golds:
        gold_range = set(range(gold["start"], gold["end"]))
        if gold_range.intersection(pred_idxs) == set():
            gold_copy = dict(gold)
            gold_copy["label"] = "FN"
            outspans.append(gold_copy)

    return sorted(outspans, key=lambda s: s["start"])


def count_tp_fp_fn(labeled):
    # This check allows us to compute the metrics
    # for a single example or the full data set.
    if isinstance(labeled, dict):
        labeled = [pred for doc_id in labeled.keys()
                   for pred in labeled[doc_id]]
    counts = Counter((pred["label"] for pred in labeled))
    return counts["TP"], counts["FP"], counts["FN"]


def compute_prec_rec_f1_from_labels(labeled):
    """
    Compute precision, recall, and F1.
    """
    tp, fp, fn = count_tp_fp_fn(labeled)

    try:
        prec = tp / (tp + fp)
    except ZeroDivisionError:
        prec = 0
    try:
        rec = tp / (tp + fn)
    except ZeroDivisionError:
        rec = 0
    try:
        f1 = (2 * prec * rec) / (prec + rec)
    except ZeroDivisionError:
        f1 = 0
    return prec, rec, f1


def plot_eval_stats(stats, color="skyblue", save_to=None):
    """
    Plots the distribution of metrics across documents.
    """
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.5))

    axs[0].hist(stats["precision"], color=color)
    axs[0].set_title("Precision")
    axs[0].set_xticks([0.0, 0.5, 1.0])
    axs[0].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    axs[1].hist(stats["recall"], color=color)
    axs[1].set_title("Recall")
    axs[1].set_xticks([0.0, 0.5, 1.0])
    axs[1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    axs[2].hist(stats["f1"], color=color)
    axs[2].set_title("F1")
    axs[2].set_xticks([0.0, 0.5, 1.0])
    axs[2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.tight_layout()

    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


if __name__ == "__main__":
    args = parse_args()
    predictions = json.load(open(args.predictions))
    gold_standard = json.load(open(args.gold_standard))
    evaluate(predictions, gold_standard, args.outdir)

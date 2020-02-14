import argparse
import json
import editdistance
import pickle
import numpy as np
from tqdm import tqdm


"""
This script computes the Levenshtein and Jaccard distances between the
raw matched and unmatched terms. For each term it finds the minimum
distance in the other set.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="JSON file containing matches.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Where to save the matching summary.")
    return parser.parse_args()


def main(infile, outfile):
    matches = json.load(open(infile))
    # We don't (as of now) compute distances between sets of normalized terms.
    matched, unmatched = get_matched_unmatched_raw_terms(matches)
    print("Computing Edit Distances")
    edits_match, edits_unmatch = min_distances(matched, unmatched,
                                               editdistance.eval)
    pickle.dump(edits_match, open(f"{outfile}.edit_dist_match.pickle", "wb"))
    pickle.dump(edits_unmatch, open(f"{outfile}.edit_dist_unmatch.pickle", "wb"))  # noqa
    print("Computing Jaccard Distances")
    jaccs_match, jaccs_unmatch = min_distances(matched, unmatched, jaccard)
    pickle.dump(jaccs_match, open(f"{outfile}.jaccard_dist_match.pickle", "wb"))  # noqa
    pickle.dump(jaccs_unmatch, open(f"{outfile}.jaccard_dist_unmatch.pickle", "wb"))  # noqa
    summarize(edits_match, outfile, write_mode='w',
              name="Edit Distance (matched -> unmatched)")
    summarize(edits_unmatch, outfile, write_mode='a',
              name="Edit Distance (unmatched -> matched)")
    summarize(jaccs_match, outfile, write_mode='a',
              name="Jaccard Distance (matched -> unmatched)")
    summarize(jaccs_unmatch, outfile, write_mode='a',
              name="Jaccard Distance (unmatched -> matched)")


def get_matched_unmatched_raw_terms(matches):
    matched_terms = set()
    unmatched_terms = set()
    for cui in matches:
        for aui in matches[cui]:
            match = matches[cui][aui]
            if len(match["umls_cuis"]) > 0:
                matched_terms.add(match["term"].lower())
            else:
                unmatched_terms.add(match["term"].lower())
    return matched_terms, unmatched_terms


def min_distances(matched, unmatched, distance_func):
    """
    Returns two lists of numbers:
      1. The minimum distance of each matched term to the unmatched terms.
      2. The minimum distance of each unmatched term to the matched terms.
    where distance is measured by the supplied distance function.
    """
    matched_distances = [np.inf] * len(matched)
    unmatched_distances = [np.inf] * len(unmatched)
    for (m, matched_term) in tqdm(list(enumerate(matched))):
        for (u, unmatched_term) in enumerate(unmatched):
            dist = distance_func(matched_term, unmatched_term)
            if dist < matched_distances[m]:
                matched_distances[m] = dist
            if dist < unmatched_distances[u]:
                unmatched_distances[m] = dist
    return matched_distances, unmatched_distances


def jaccard(term1, term2):
    st1 = set(term1)
    st2 = set(term2)
    num = len(st1.intersection(st2))
    denom = len(st1.union(st2))
    return 1 - (num / denom)


def summarize(distances, outfile, write_mode='w', name=""):
    q1, q2, q3 = np.percentile(distances, [25, 50, 75])
    minimum = min(distances)
    maximum = max(distances)
    with open(outfile, write_mode) as outF:
        outF.write(name + '\n')
        outF.write(f"min, max: {minimum:.2f}, {maximum:.2f}\n")
        outF.write(f"quartiles: {q1:.2f}, {q2:.2f}, {q3:.2f}\n\n")


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outfile)

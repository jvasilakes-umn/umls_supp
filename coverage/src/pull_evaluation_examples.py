import argparse
import csv
import json
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches_json", type=str, required=True,
                        help="JSON lines file containing matches.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Where to save the output as CSV.")
    return parser.parse_args()


def main(matches_file, outfile):
    matches = [json.loads(line.strip()) for line in open(matches_file)]
    matches = prepare_matches(matches)
    eval_matches = choose_evaluation_concepts(matches)
    with open(outfile, 'w') as outF:
        writer = csv.writer(outF)
        writer.writerows(format_as_csv(eval_matches))


def prepare_matches(matches):
    transformed = defaultdict(dict)
    for m in matches:
        cui = m["idisk_cui"]
        aui = m["idisk_aui"]
        if aui in transformed[cui]:
            if len(m["umls_cuis"]) == 0:
                continue
            if len(transformed[cui][aui]["umls_cuis"]) > 0:
                continue
        term_dict = {"term": m["term"], "umls_cuis": m["umls_cuis"],
                     "normed": m["normed"]}
        transformed[cui][aui] = term_dict
    return transformed


def choose_evaluation_concepts(matches):
    chosen_ones = {}
    num_terms = 0
    for cui in matches:
        num_unmatched = len([m for m in matches[cui].values()
                             if m["umls_cuis"] == []])
        proportion_unmatched = num_unmatched / len(matches[cui])
        if proportion_unmatched >= 0.75:
            matches_sorted = {aui: matches[cui][aui] for aui
                              in sorted(matches[cui].keys())}
            matches_unique = {}
            seen = set()
            for aui in matches_sorted:
                if matches[cui][aui]["term"] not in seen:
                    seen.add(matches[cui][aui]["term"])
                    matches_unique[aui] = matches[cui][aui]
            chosen_ones[cui] = matches_unique
            num_terms += len([m for m in matches_unique.values()
                              if m["umls_cuis"] == []])
            if num_terms in range(2875, 2950):
                print(num_terms)
                break
    return chosen_ones


def format_as_csv(matches):
    for cui in matches:
        for aui in matches[cui]:
            m = matches[cui][aui]
            row = [cui, aui,
                   m["term"],
                   ', '.join(m["umls_cuis"])]
            yield row


if __name__ == "__main__":
    args = parse_args()
    main(args.matches_json, args.outfile)

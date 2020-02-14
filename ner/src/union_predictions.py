import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions1", type=str, required=True,
                        help="The first labeled predictions file.")
    parser.add_argument("--predictions2", type=str, required=True,
                        help="The second labeled predictions file.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Where to write the union of the predictions.")
    return parser.parse_args()


def main(predfile1, predfile2, outfile):
    preds1 = json.load(open(predfile1))
    preds2 = json.load(open(predfile2))
    union_preds = union(preds1, preds2)
    with open(outfile, 'w') as outF:
        json.dump(union_preds, outF)


def union(preds1, preds2):
    union = {}
    all_cuis = set(preds1.keys()).union(set(preds2.keys()))
    for cui in all_cuis:
        if cui not in preds2.keys():
            union[cui] = preds1[cui]
        elif cui not in preds1.keys():
            union[cui] = preds2[cui]
        else:
            union[cui] = span_union(preds1[cui], preds2[cui])
    return union


def span_union(spans1, spans2):
    union = []
    seen_spans = set()
    all_spans = sorted(spans1 + spans2, key=lambda x: x["start"])
    for span in all_spans:
        char_span = (span["start"], span["end"])
        if char_span not in seen_spans:
            seen_spans.add(char_span)
            union.append(span)
    return union


if __name__ == "__main__":
    args = parse_args()
    main(args.predictions1, args.predictions2, args.outfile)

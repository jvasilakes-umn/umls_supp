import os
import sys
import argparse
import json
from tqdm import tqdm

from quickumls import QuickUMLS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=str, required=True,
                        help="Documents to run QuickUMLS over.")
    parser.add_argument("--quickumls_install_dir", type=str, required=True,
                        help="Path to QuickUMLS installation.")
    parser.add_argument("--quickumls_conf", type=str, required=True,
                        help="Path to quickumls JSON configuration file.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="""Where to save the predictions.""")
    args = parser.parse_args()
    return args


def run_quickumls(docs, matcher):
    ann_by_text_id = {}
    for (text_id, text) in tqdm(docs.items()):
        try:
            matches = matcher.match(text, best_match=True)
            best_matches = []
            for phrase in matches:
                best = sorted(phrase, key=lambda x: x["similarity"])[-1]
                best = format_annotation(best)
                best_matches.append(best)
            best_matches = sorted(best_matches, key=lambda x: x["start"])
            ann_by_text_id[text_id] = best_matches
        except KeyboardInterrupt:
            print("Saving what has been annotated so far.")
            print("  Please wait a moment.")
            return ann_by_text_id
    return ann_by_text_id


def format_annotation(ann):
    outann = {"start": ann["start"],
              "end": ann["end"],
              "matched_text": ann["ngram"],
              "semtypes": list(ann["semtypes"]),
              "cui": ann["cui"]}
    return outann


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.outfile):
        print("Output file already exists. Please delete it and rerun.")
        print("Aborting.")
        sys.exit(1)
    docs = json.load(open(args.documents, 'r'))
    conf = json.load(open(args.quickumls_conf))
    print("========================")
    print("QuickUMLS configuration:")
    print("QuickUMLS installation: {args.quickumls_install_dir}")
    print(json.dumps(conf, indent=2))
    print("========================")
    matcher = QuickUMLS(args.quickumls_install_dir, **conf)
    anns = run_quickumls(docs, matcher)
    json.dump(anns, open(args.outfile, 'w'))

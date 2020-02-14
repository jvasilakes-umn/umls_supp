import sys
import os
import argparse
import json
from collections import defaultdict
from nltk.tokenize import sent_tokenize

# Import the BratANN class from ira_seq.py at the following path
sys.path.insert(0, "/Users/vasil024/Projects/Code/IRA/brat-irr/brat/sequence/")
from ira_seq import BratANN  # noqa


"""
Convert the brat annotations into a common JSON format.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anndirs", type=str, nargs='+', required=True,
                        help="""List of directories containing the
                                .ann and .txt files.""")
    parser.add_argument("--outdir", type=str, default=None,
                        help="""Directory in which to save
                                the converted annotations.""")
    return parser.parse_args()


def main(anndirs, outdir):
    abstracts_json = {}
    annotations_json = {}
    total_stats = {"total_anns": 0,
                   "unique_terms": 0}
    stats_by_dir = {}
    terms_and_examples = defaultdict(set)
    for anndir in anndirs:
        seen_terms = set()
        stats = {"total_anns": 0,
                 "avg_anns_per_file": 0,
                 "unique_terms": 0}
        basenames = [os.path.splitext(fname)[0]
                     for fname in os.listdir(anndir)
                     if not fname.startswith('.')]
        basepaths = [os.path.join(anndir, bname) for bname in basenames]

        for (i, basepath) in enumerate(basepaths):
            abstract_id = os.path.basename(basepath)
            ann_path = basepath + ".ann"
            txt_path = basepath + ".txt"

            abstract = open(txt_path, 'r').read().strip()
            abstract_sents = sent_tokenize(abstract)
            anns = [BratANN.from_string(line.strip())
                    for line in open(ann_path)]
            # Skip abstracts with no annotations
            if len(anns) == 0:
                continue

            json_abs = {f"{abstract_id}": abstract}
            json_anns = convert_brat_to_json(abstract_id, anns)
            abstracts_json.update(json_abs)
            annotations_json.update(json_anns)

            for ann in anns:
                term = ann.text.lower()
                start_pos = ann.start
                sent = get_sentence_by_char_index(abstract_sents, start_pos)
                terms_and_examples[term].add(sent)
            seen_terms.update(set([a.text.lower() for a in anns]))

            stats["total_anns"] += len(anns)
            prev_avg = stats["avg_anns_per_file"]
            new_avg = prev_avg + ((1/(i+1)) * (len(anns) - prev_avg))
            stats["avg_anns_per_file"] = new_avg

        stats["unique_terms"] = len(seen_terms)
        stats_by_dir[anndir] = stats
        # all_seen_terms.update(seen_terms)

    total_stats["total_anns"] = sum([stats_by_dir[d]["total_anns"]
                                     for d in stats_by_dir])
    # total_stats["unique_terms"] = len(all_seen_terms)
    total_stats["unique_terms"] = len(terms_and_examples.keys())

    abstracts_outfile = os.path.join(outdir, "abstracts.json")
    annotations_outfile = os.path.join(outdir, "gold_standard.json")
    stats_outfile = os.path.join(outdir, "gold_standard_stats.log")
    terms_outfile = os.path.join(outdir, "gold_standard_terms.jsonl")

    with open(abstracts_outfile, 'w') as outF:
        json.dump(abstracts_json, outF)

    with open(annotations_outfile, 'w') as outF:
        json.dump(annotations_json, outF)

    with open(stats_outfile, 'w') as outF:
        for dirname in stats_by_dir.keys():
            outF.write(dirname + '\n')
            outF.write(str(stats_by_dir[dirname]) + '\n\n')

    with open(terms_outfile, 'w') as outF:
        # for term in sorted(all_seen_terms):
        for (term, examples) in terms_and_examples.items():
            data = {"string": term, "examples": list(examples),
                    "cui": "", "flagged": False}
            json.dump(data, outF)
            outF.write('\n')


def convert_brat_to_json(abstract_id, brat_anns):
    spans = []
    for ann in brat_anns:
        span = {"start": ann.start, "end": ann.end,
                "matched_text": ann.text, "semtypes": [],
                "cui": "NONE"}
        spans.append(span)
    return {f"{abstract_id}": spans}


def get_sentence_by_char_index(sentences, index):
    assert index >= 0
    ch_idx = 0
    for sent in sentences:
        ch_idx += len(sent)
        if ch_idx >= index:
            return sent
    return None


if __name__ == "__main__":
    args = parse_args()
    main(args.anndirs, outdir=args.outdir)

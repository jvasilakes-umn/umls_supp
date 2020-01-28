import os
import argparse
import csv
import pickle

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

CWD = os.path.dirname(os.path.abspath(__file__))
PICKLE_DIR = os.path.join(CWD, "../data/external")

KEEP_SEMTYPES = ["T116", "T123", "T103", "T196", "T125", "T129",
                 "T197", "T114", "T109", "T121", "T127", "T007",
                 "T013", "T004", "T002", "T168"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuis", type=str, nargs='+',
                        help="Comma separated CUIs to investigate")
    parser.add_argument("--mrconso", type=str, required=True,
                        help="""Path to MRCONSO.RRF or the previously
                                pickled version""")
    parser.add_argument("--mrsty", type=str, required=True,
                        help="""Path to MRSTY.RRF or the previously
                                pickled version""")
    parser.add_argument("--mrrel", type=str, required=True,
                        help="""Path to MRREL.RRF or the previously
                                pickled version""")
    parser.add_argument("--outdir", type=str, required=True,
                        help="""Where to save the terms.""")
    parser.add_argument("--stopwords", type=str, default=None,
                        help="""Path to file containing stopwords to filter
                                from the resulting term list.""")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Run tests.")
    parser.add_argument("--reverse", action="store_true", default=False,
                        help="""Ascend rather than descend the tree.""")
    return parser.parse_args()


def main(cuis, mrconso_path, mrsty_path, mrrel_path, outdir,
         stopwords_file=None, reverse=False):
    if stopwords_file is not None:
        print(f"Filtering stopwords from {stopwords_file}.")
        stopwords = [l.strip() for l in open(stopwords_file)]
    else:
        stopwords = []
    mrconso = index_mrconso(mrconso_path, stopwords=stopwords)
    mrsty = index_mrsty(mrsty_path)
    mrrel = index_mrrel(mrrel_path)
    all_related = set()
    for cui in cuis:
        related = get_related_cuis(cui, mrrel, mrsty, reverse=reverse)
        all_related.update(related)
    print("Num related CUIs: ", len(related))

    mrconso_outfile = os.path.join(outdir, "MRCONSO.RRF")
    with open(mrconso_outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter='|')
        for rcui in all_related:
            writer.writerows(mrconso[rcui])

    mrsty_outfile = os.path.join(outdir, "MRSTY.RRF")
    with open(mrsty_outfile, 'w') as outF:
        writer = csv.writer(outF, delimiter='|')
        for rcui in all_related:
            writer.writerows(mrsty[rcui])


def index_mrconso(mrconso_path, stopwords=[]):
    # headers = ['cui', 'lat', 'ts', 'lui', 'stt', 'sui',
    #            'ispref', 'aui', 'saui', 'scui', 'sdui',
    #            'sab', 'tty', 'code', 'str', 'srl', 'suppress', 'cvf']
    picklefile = os.path.join(PICKLE_DIR, "mrconso.pickle")
    if os.path.exists(picklefile):
        print("Loading pickled MRCONSO.")
        mrconso = pickle.load(open(picklefile, 'rb'))
    elif mrconso_path.endswith("pickle"):
        print("Loading pickled MRCONSO.")
        mrconso = pickle.load(open(mrconso_path, 'rb'))
    else:
        mrconso = defaultdict(list)
        with open(mrconso_path, 'r', encoding="utf-8") as rrf_file:
            reader = csv.reader(rrf_file, delimiter='|')
            print("Indexing MRCONSO")
            for row in tqdm(reader):
                if row[14].lower() in stopwords:
                    continue
                cui = str(row[0])
                mrconso[cui].append(row)
        print(f"Saving indexed MRCONSO to {picklefile}")
        pickle.dump(mrconso, open(picklefile, 'wb'))
    return mrconso


def index_mrsty(mrsty_path):
    picklefile = os.path.join(PICKLE_DIR, "mrsty.pickle")
    if os.path.exists(picklefile):
        print("Loading pickled MRSTY.")
        mrsty = pickle.load(open(picklefile, 'rb'))
    elif mrsty_path.endswith("pickle"):
        print("Loading pickled MRSTY.")
        mrsty = pickle.load(open(mrsty_path, 'rb'))
    else:
        mrsty = defaultdict(list)
        with open(mrsty_path, 'r', encoding="utf-8") as rrf_file:
            reader = csv.reader(rrf_file, delimiter='|')
            print("Indexing MRSTY")
            for row in tqdm(reader):
                cui = str(row[0])
                # FYI: semtype = row[1]
                mrsty[cui].append(row)
        print(f"Saving indexed MRSTY to {picklefile}")
        pickle.dump(mrsty, open(picklefile, "wb"))
    return mrsty


def index_mrrel(mrrel_path):
    picklefile = os.path.join(PICKLE_DIR, "mrrel.pickle")
    if os.path.exists(picklefile):
        print("Loading pickled MRREL.")
        mrrel = pickle.load(open(picklefile, 'rb'))
    elif mrrel_path.endswith("pickle"):
        print("Loading pickled MRREL.")
        mrrel = pickle.load(open(mrrel_path, 'rb'))
    else:
        mrrel = defaultdict(set)
        with open(mrrel_path, 'r', encoding="utf-8") as rrf_file:
            reader = csv.reader(rrf_file, delimiter='|')
            print("Indexing MRREL")
            for row in tqdm(reader):
                obj, subj = row[0], row[4]
                if subj == obj:
                    continue
                rel = row[3]
                # CHD: child, RN: narrower
                if rel == "CHD" or rel == "RN":
                    mrrel[obj].add(subj)
                # PAR: parent, RB: broader
                elif rel == "PAR" or rel == "RB":
                    mrrel[subj].add(obj)
                else:
                    continue
        print(f"Saving indexed MRREL to {picklefile}")
        pickle.dump(mrrel, open(picklefile, 'wb'))
    return mrrel


def get_related_cuis(cui, mrrel, mrsty, visited=set(), reverse=False):
    # Ascend the hierarchy rather than descend
    if reverse is True:
        new_mrrel = defaultdict(set)
        for (key, vals) in mrrel.items():
            for val in vals:
                new_mrrel[val].add(key)
        mrrel = new_mrrel
        reverse = False
    for rcui in mrrel[cui]:
        semtypes = {row[1] for row in mrsty[rcui]}
        semtype_is_valid = len(semtypes.intersection(KEEP_SEMTYPES)) > 0
        if rcui not in visited and semtype_is_valid is True:
            visited.add(rcui)
            get_related_cuis(rcui, mrrel, mrsty,
                             visited=visited, reverse=reverse)
    return visited


def get_related_path(start_cui, end_cui, mrrel, visited=set(), reverse=False):
    # Ascend the hierarchy rather than descend
    if reverse is True:
        new_mrrel = defaultdict(set)
        for (key, vals) in mrrel.items():
            for val in vals:
                new_mrrel[val].add(key)
        mrrel = new_mrrel
        reverse = False
    if start_cui == end_cui:
        return [end_cui]
    for rcui in mrrel[start_cui]:
        if rcui in visited:
            continue
        visited.add(rcui)
        related = get_related_path(rcui, end_cui, mrrel, visited=visited)
        if end_cui in related:
            return [start_cui] + related
    return []


def test():
    mrsty = {c: "T116" for c in "abcdety"}
    mrrel = defaultdict(set)
    mrrel['a'] = set(['b', 'c'])
    mrrel['b'] = set(['c', 'd', 'e'])
    mrrel['c'] = set(['t'])
    mrrel['d'] = set(['y'])
    assert get_related_cuis('a', mrrel, mrsty) == {'y', 'c', 'e', 'd', 't', 'b'}  # noqa

    mrrel_copy = deepcopy(mrrel)
    assert get_related_cuis('a', mrrel_copy, mrsty,
                            visited=set(), reverse=True) == set()
    assert get_related_cuis('t', mrrel_copy, mrsty,
                            visited=set(), reverse=True) == {'a', 'c', 'b'}
    assert get_related_cuis('y', mrrel_copy, mrsty,
                            visited=set(), reverse=True) == {'a', 'b', 'd'}

    # Add some cycles
    mrrel['d'] = set(['a', 'b'])
    mrrel['y'] = set(['b'])
    assert get_related_cuis('a', mrrel, mrsty) == {'y', 'c', 'e', 'd', 't', 'b'}  # noqa
    print("passed")


if __name__ == "__main__":
    args = parse_args()
    if args.test is True:
        test()
    else:
        cuis = list(csv.reader(args.cuis, skipinitialspace=True))[0]
        main(cuis, args.mrconso, args.mrsty, args.mrrel, args.outdir,
             stopwords_file=args.stopwords, reverse=args.reverse)

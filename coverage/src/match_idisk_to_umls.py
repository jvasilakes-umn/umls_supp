import os
import argparse
import json
import subprocess
import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from tempfile import NamedTemporaryFile

import idlib


KEEP_SEMTYPES = ["T116", "T123", "T103", "T196", "T125", "T129",
                 "T197", "T114", "T109", "T121", "T127", "T007",
                 "T013", "T004", "T002", "T168"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--umls_mrconso", type=str, required=True,
                        help="""Path to UMLS MRCONSO file.""")
    parser.add_argument("--umls_mrsty", type=str, required=True,
                        help="""Path to UMLS MRSTY file.""")
    parser.add_argument("--idisk_version_dir", type=str, required=True,
                        help="""Path to iDISK version to load.""")
    parser.add_argument("--luinorm_bin", type=str, required=True,
                        help="Path to luiNorm binary.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="""Where to save the links.""")
    return parser.parse_args()


def index_mrsty(mrsty_path):
    mrsty = pd.read_csv(mrsty_path, sep='|', header=None,
                        usecols=[0, 1])
    mrsty.columns = ["cui", "sty"]

    cui2sty = defaultdict(set)
    for row in tqdm(mrsty.itertuples()):
        cui = str(row.cui)
        sty = str(row.sty)
        cui2sty[cui].add(sty)
    return cui2sty


def index_mrconso(mrconso_path, mrsty_lookup):
    mrconso = pd.read_csv(mrconso_path, sep='|', header=None,
                          usecols=[0, 14])
    mrconso.columns = ["cui", "str"]

    str2cui = defaultdict(set)
    for row in tqdm(mrconso.itertuples()):
        string = str(row.str).lower()
        cui = str(row.cui)
        valid_semtypes = mrsty_lookup[cui].intersection(KEEP_SEMTYPES)
        if len(valid_semtypes) > 0:
            str2cui[string].add(cui)
    return str2cui


def norm_atoms(concepts, luinorm_bin):
    query_terms = set()
    for concept in concepts:
        for atom in concept.get_atoms():
            query_terms.add(atom.term.lower())

    with NamedTemporaryFile(delete=False) as f:
        infname = f.name
        f.write('\n'.join(query_terms).encode())

    outfname = ".luiNorm.out"
    call = f"{luinorm_bin} -i:{infname} -o:{outfname}"
    print(f"Norm call: {call}")
    process = subprocess.Popen(call, shell=True, stdout=subprocess.PIPE)
    process.wait()
    # {original_term: normed_term}
    normed = dict([l.split('|') for l in open(outfname)])
    normed = {key: val.strip('\n') for (key, val) in normed.items()}

    # Update the atom terms with their normed versions.
    for concept in concepts:
        for atom in concept.get_atoms():
            try:
                atom.term = normed[atom.term.lower()]
            except KeyError:
                pass

    os.unlink(infname)
    os.unlink(outfname)
    return concepts


def match_idisk_to_umls(idisk_kb, mrconso_lookup,
                        norm=False, luinorm_bin=None):
    sdsi = [c for c in idisk_kb if c.concept_type == "SDSI"]
    if norm is True:
        if luinorm_bin is None:
            raise ValueError("If norm=True, luinorm_bin must not be None.")
        sdsi = norm_atoms(sdsi, luinorm_bin)

    matches = defaultdict(dict)
    for concept in tqdm(sdsi):
        idisk_cui = concept.ui
        for atom in concept.get_atoms():
            string = atom.term.lower()
            cuis = list(mrconso_lookup[string])
            data = {"term": atom.term, "umls_cuis": cuis, "normed": norm}
            matches[idisk_cui][atom.ui] = data
    return matches


if __name__ == "__main__":
    args = parse_args()

    print("LOADING IDISK")
    idisk = idlib.load_kb(args.idisk_version_dir)
    print("LOADING UMLS")
    mrsty_lookup = index_mrsty(args.umls_mrsty)
    mrconso_lookup = index_mrconso(args.umls_mrconso, mrsty_lookup)
    print("DONE")

    print("MATCHING TERMINOLOGIES")
    print("  Resticting matches to UMLS semantic types:")
    print(f"   {KEEP_SEMTYPES}\n")
    exact_matches = match_idisk_to_umls(idisk, mrconso_lookup, norm=False)
    norm_matches = match_idisk_to_umls(idisk, mrconso_lookup, norm=True,
                                       luinorm_bin=args.luinorm_bin)
    # Merge the matches together. If an exact match was found, keep it.
    # Otherwise, keep the normalized match, whether it was found or not.
    out_matches = defaultdict(dict)
    for cui in exact_matches.keys():
        for aui in exact_matches[cui].keys():
            if len(exact_matches[cui][aui]["umls_cuis"]) > 0:
                out_matches[cui][aui] = exact_matches[cui][aui]
            else:
                out_matches[cui][aui] = norm_matches[cui][aui]
    print("DONE")

    print(f"SAVING TO {args.outfile}")
    with open(args.outfile, 'w') as outF:
        json.dump(out_matches, outF)

    # summarize_matches(all_matches)

import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, required=True,
                        help="JSON file containing matches.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Directory in which to save the summary files.")
    return parser.parse_args()


def main(infile, outdir):
    matches = json.load(open(infile))
    summary_str = coverage_summary(matches)
    summary_outfile = os.path.join(outdir, "coverage_summary.txt")
    with open(summary_outfile, 'w') as outF:
        outF.write(summary_str)

    plot_matching_stats(matches, outdir)


def coverage_summary(matches):
    outstr = ""
    num_atoms = len([aui for cui in matches.keys()
                     for aui in matches[cui].keys()])
    all_matched = [m for cui in matches.keys() for m in matches[cui].values()
                   if len(m["umls_cuis"]) > 0]
    exact_matched = [m for m in all_matched if m["normed"] is False]
    norm_matched = [m for m in all_matched if m["normed"] is True]
    prop_all_match = len(all_matched) / num_atoms
    prop_exact_match = len(exact_matched) / num_atoms
    prop_norm_match = len(norm_matched) / num_atoms

    outstr += f"Number of Atoms: {num_atoms}\n"
    outstr += f"  Total matches: {len(all_matched)} ({prop_all_match:.2f})\n"
    outstr += f"  Exact matches: {len(exact_matched)} ({prop_exact_match:.2f})\n"  # noqa
    outstr += f"  Normalized matches: {len(norm_matched)} ({prop_norm_match:.2f})\n"  # noqa

    # The number of unique terms.
    num_terms = len({m["term"].lower() for cui in matches.keys()
                     for m in matches[cui].values()})
    terms_matched = {m["term"].lower() for m in all_matched}
    terms_exact = {m["term"].lower() for m in exact_matched}
    terms_norm = {m["term"].lower() for m in norm_matched}
    prop_all_match = len(terms_matched) / num_terms
    prop_exact_match = len(terms_exact) / num_terms
    prop_norm_match = len(terms_norm) / num_terms

    outstr += f"\nNumber of unique terms: {num_terms}\n"
    outstr += f"  Unique terms matched: {len(terms_matched)} ({prop_all_match:.2f})\n"  # noqa
    outstr += f"  Exact matches: {len(terms_exact)} ({prop_exact_match:.2f})\n"
    outstr += f"  Normalized matches: {len(terms_norm)} ({prop_norm_match:.2f})\n"  # noqa
    return outstr


def plot_matching_stats(matches, outdir):
    outfile = os.path.join(outdir, "atoms_per_concept.png")
    plot_atoms_per_concept(matches, outfile)
    plot_match_stats(matches, outdir)


def plot_atoms_per_concept(matches, outfile):
    atom_counts = [len(matches[cui]) for cui in matches.keys()]
    plt.figure()
    plt.hist(atom_counts)
    plt.xlabel("Number of Atoms")
    plt.ylabel("Number of Concepts")
    plt.savefig(outfile)


def plot_matches_per_concept(matches, outfile):
    matched_proportions = []
    for cui in matches.keys():
        atoms = list(matches[cui].values())
        num_matched = len([m for m in atoms if len(m["umls_cuis"]) > 0])
        matched_proportions.append(num_matched / len(atoms))

    bins = [f"{start:.2f}-{start+0.09:.2f}"
            for start in np.linspace(0.0, 1.0, num=11)]
    bins[-1] = "1.00"
    bins[0] = "0.01-0.09"
    bins.insert(0, "0.00")
    binned_matches = {binn: 0 for binn in bins}
    for proportion in matched_proportions:
        if proportion == 0.0:
            binn_idx = 0
        else:
            # We multiply by 10 to avoid rounding errors
            binn_idx = int((proportion * 10) // 1) + 1
        binn = bins[binn_idx]
        binned_matches[binn] += 1

    x_positions = range(len(bins))
    heights = sorted(binned_matches.values(), reverse=True)
    labels = bins[::-1]
    plt.figure()
    plt.bar(x_positions, heights)
    plt.xticks(x_positions, labels, rotation=45)
    plt.title("Number of concepts by proportion of atoms matched")
    plt.xlabel("Proportion of atoms matched")
    plt.ylabel("Number of concepts")
    plt.tight_layout()
    plt.savefig(outfile)


def plot_match_stats(matches, outdir):
    matched_proportions = {}
    num_atoms_per_concept = {}
    num_cuis_per_concept = {}
    for cui in matches.keys():
        atoms = list(matches[cui].values())
        num_matched = len([m for m in atoms if len(m["umls_cuis"]) > 0])
        matched_proportions[cui] = (num_matched / len(atoms))
        uniq_concepts = {cui for m in atoms for cui in m["umls_cuis"]}
        num_atoms_per_concept[cui] = len(atoms)
        num_cuis_per_concept[cui] = len(uniq_concepts)

    bins = [f"{start:.2f}-{start+0.09:.2f}"
            for start in np.linspace(0.0, 1.0, num=11)]
    bins[-1] = "1.00"
    bins[0] = "0.01-0.09"
    bins.insert(0, "0.00")
    proportion_bins = {binn: 0 for binn in bins}
    num_atoms_bins = {binn: [] for binn in bins}
    num_cuis_bins = {binn: [] for binn in bins}
    for (cui, proportion) in matched_proportions.items():
        if proportion == 0.0:
            binn_idx = 0
        else:
            # We multiply by 10 to avoid rounding errors
            binn_idx = int((proportion * 10) // 1) + 1
        binn = bins[binn_idx]
        proportion_bins[binn] += 1
        num_atoms_bins[binn].append(num_atoms_per_concept[cui])
        num_cuis_bins[binn].append(num_cuis_per_concept[cui])

    x_positions = range(len(bins))
    labels = bins[::-1]
    prop_heights = sorted(proportion_bins.values(), reverse=True)
    plt.figure()
    plt.bar(x_positions, prop_heights)
    plt.xticks(x_positions, labels, rotation=45)
    plt.title("Number of concepts by proportion of atoms matched")
    plt.xlabel("Proportion of atoms matched")
    plt.ylabel("Number of concepts")
    plt.tight_layout()
    outfile = os.path.join(outdir, "matches_per_concept.png")
    plt.savefig(outfile)

    sorted_num_atoms_bins = [(binn, num_atoms_bins[binn]) for binn in labels]
    mean_num_atoms_heights = [np.mean(vals) for (binn, vals)
                              in sorted_num_atoms_bins]
    plt.figure()
    plt.bar(x_positions, mean_num_atoms_heights)
    plt.xticks(x_positions, labels, rotation=45)
    plt.title("Mean number of atoms per concept by\nproportion of atoms matched")  # noqa
    plt.xlabel("Proportion of atoms matched")
    plt.ylabel("Mean number of atoms")
    plt.tight_layout()
    outfile = os.path.join(outdir, "mean_atoms_per_match_proportion.png")
    plt.savefig(outfile)

    sorted_num_cuis_bins = [(binn, num_cuis_bins[binn]) for binn in labels]
    mean_num_cuis_heights = [np.mean(vals) for (binn, vals)
                             in sorted_num_cuis_bins]
    plt.figure()
    plt.bar(x_positions, mean_num_cuis_heights)
    plt.xticks(x_positions, labels, rotation=45)
    plt.title("Mean number of unique CUIs per concept by\nproportion of atoms matched")  # noqa
    plt.xlabel("Proportion of atoms matched")
    plt.ylabel("Mean number of unique CUIs")
    plt.tight_layout()
    outfile = os.path.join(outdir, "mean_cuis_per_match_proportion.png")
    plt.savefig(outfile)


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outdir)

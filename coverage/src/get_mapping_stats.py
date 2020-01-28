import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from idlib.entity_linking import MetaMapDriver


FIGDIR = "figures"
if not os.path.exists(FIGDIR):
    os.makedirs(FIGDIR)

KEEP_SEMTYPES = ["aapp", "bacs", "chem", "elii", "horm", "imft",
                 "inch", "nnon", "orch", "phsu", "vita", "bact",
                 "fish", "fngs", "plnt", "food"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_in", type=str, required=True,
                        help="""MetaMap input.""")
    parser.add_argument("--query_out", type=str, required=True,
                        help="""Raw MetaMap JSON output.""")
    parser.add_argument("--metamap_bin", type=str, required=True,
                        help="""Path to MetaMap bin directory.""")
    parser.add_argument("--outfile", type=str, required=True,
                        help="""Where to save the summary as Markdown.""")
    return parser.parse_args()


def main(query_in, query_out, mm_bin, outfile):
    mm_in = [l.strip() for l in open(query_in)]
    mm_out = json.load(open(query_out))
    metamap_driver = MetaMapDriver(mm_bin, min_score=1000,
                                   term_processing=True, data_year="2018AA",
                                   keep_semtypes=KEEP_SEMTYPES)
    number_atoms_per_concept(mm_in)
    return
    table1 = count_matches_by_score(mm_in, mm_out, metamap_driver)
    get_concept_score_distribution(mm_in, mm_out, metamap_driver)
    table2s = {}
    table2s[1000] = count_exact_matches_by_concept(mm_in, mm_out,
                                                   metamap_driver,
                                                   score_range=(1000, 1001))
    table2s[800] = count_exact_matches_by_concept(mm_in, mm_out,
                                                  metamap_driver,
                                                  score_range=(800, 900))
    make_markdown_file(table1, table2s, outfile=outfile)


def number_atoms_per_concept(mm_in):
    num_atoms = defaultdict(int)
    for row in mm_in:
        qid = row.split('|')[0]
        concept_id, atom_idx = qid.split('-')
        num_atoms[concept_id] += 1
    plt.hist(num_atoms.values())
    plt.xlabel("Number of atoms")
    plt.ylabel("Number of concepts")
    plt.show()


def count_matches_by_score(mm_in, mm_out, metamap_driver):
    mappings = metamap_driver._convert_output_to_candidate_links(mm_out)

    failed_queries = []
    mapping_scores = {}
    for query_id in mappings.keys():
        if not mappings[query_id]:
            failed_queries.append(query_id)
            continue
        for (phrase, candidates) in mappings[query_id].items():
            scores = [abs(int(cand.linking_score)) for cand in candidates]
            mapping_scores[query_id] = max(scores)

    bins = [f"{start}-{start+99}" for start in range(0, 1000, 100)] + ["1000"]
    bins[0] = "1-99"
    bins.insert(0, "0")
    binned_queries = {binn: 0 for binn in bins}
    for (query_id, score) in mapping_scores.items():
        if score == 0:
            binn_idx = 0
        else:
            binn_idx = (score // 100) + 1
        binn = bins[binn_idx]
        binned_queries[binn] += 1
    binned_queries['0'] = len(failed_queries)
    assert sum(binned_queries.values()) == len(mappings.keys())

    pos = list(range(len(bins)))
    heights = list(binned_queries.values())[::-1]
    labels = bins[::-1]
    plt.figure()
    plt.bar(pos, heights)
    plt.xticks(pos, labels, rotation=45)
    plt.title("Number of atoms mapped per MetaMap score range")
    plt.xlabel("MetaMap score range")
    plt.ylabel("Number of atoms")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/matches_by_score.png")
    return binned_queries


def get_concept_score_distribution(mm_in, mm_out, metamap_driver):
    mappings = metamap_driver._convert_output_to_candidate_links(mm_out)

    concept_ids = set([qid.split('-')[0] for qid in mappings.keys()])
    scores_per_concept = {cid: [] for cid in concept_ids}

    for query_id in mappings.keys():
        concept_id = query_id.split('-')[0]
        for (phrase, candidates) in mappings[query_id].items():
            mean_score = np.mean([abs(int(cand.linking_score))
                                  for cand in candidates])
            scores_per_concept[concept_id].append(mean_score)
    mean_score_per_concept = {concept_id: np.mean(scores)
                              for (concept_id, scores)
                              in scores_per_concept.items()}
    print(f"Mean score == 1000")
    print(len(set(mean_score_per_concept.keys())))
    print(len([cid for cid, val in mean_score_per_concept.items()
               if val == 1000]))
    print()
    plt.figure()
    plt.hist(mean_score_per_concept.values())
    plt.title("Mean mapping score across atoms per concept")
    plt.xlabel("Mean score")
    plt.ylabel("Number of concepts")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/mean_score_per_concept.png")


def count_exact_matches_by_concept(mm_in, mm_out, metamap_driver,
                                   score_range=(1000, 1001)):
    full_score_range = range(*score_range)
    score_range = score_range[0]
    mappings = metamap_driver._convert_output_to_candidate_links(mm_out)

    concept_ids = set([qid.split('-')[0] for qid in mappings.keys()])
    num_atoms_per_concept = {cid: 0 for cid in concept_ids}
    umls_cuis_per_concept = {cid: set() for cid in concept_ids}
    exact_matches_per_concept = {cid: [] for cid in concept_ids}
    for query_id in mappings.keys():
        concept_id = query_id.split('-')[0]
        num_atoms_per_concept[concept_id] += 1
        if not mappings[query_id]:
            continue
        for (phrase, candidates) in mappings[query_id].items():
            cuis = [cand.candidate_id for cand in candidates
                    if abs(int(cand.linking_score)) in full_score_range]
            if len(cuis) > 0:
                exact_matches_per_concept[concept_id].append(1)
                umls_cuis_per_concept[concept_id].update(cuis)
            else:
                exact_matches_per_concept[concept_id].append(0)

    normalized_exact_matches = {cui: np.mean(val) for (cui, val)
                                in exact_matches_per_concept.items()}

    bins = [f"{start:.2f}-{start+0.09:.2f}"
            for start in np.linspace(0.0, 1.0, num=11)]
    bins[-1] = "1.00"
    bins[0] = "0.01-0.09"
    bins.insert(0, "0.00")
    binned_queries = {binn: 0 for binn in bins}
    num_atoms_per_bin = {binn: [] for binn in bins}
    num_cuis_per_bin = {binn: [] for binn in bins}
    for (concept_id, num) in normalized_exact_matches.items():
        # We multiply by 10 to avoid rounding errors.
        if num == 0.0:
            binn_idx = 0
        else:
            binn_idx = int((num * 10) // 1) + 1
        binn = bins[binn_idx]
        binned_queries[binn] += 1
        num_atoms_per_bin[binn].append(num_atoms_per_concept[concept_id])
        num_cuis_per_bin[binn].append(len(umls_cuis_per_concept[concept_id]))
        if binn == "1.00":
            print(concept_id)
            input()
    print(len(concept_ids))
    print(sum(binned_queries.values()))
    assert sum(binned_queries.values()) == len(concept_ids)

    pos = range(len(bins))
    heights = list(binned_queries.values())[::-1]
    labels = bins[::-1]
    plt.figure()
    plt.bar(pos, heights)
    plt.xticks(pos, labels, rotation=45)
    plt.title(f"Number of concepts by proportion of atoms mapped\nwith score {score_range}")  # noqa
    plt.xlabel(f"Proportion of atoms with score {score_range}")
    plt.ylabel("Number of concepts")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/matches_by_concept_{score_range}.png")

    mean_num_atoms_per_bin = {binn: np.mean(val) for (binn, val)
                              in num_atoms_per_bin.items()}
    labels = list(mean_num_atoms_per_bin.keys())[::-1]
    heights = list(mean_num_atoms_per_bin.values())[::-1]
    pos = range(len(labels))
    plt.figure()
    plt.bar(pos, heights)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.title(f"Mean number of atoms per concept by\nproportion of atoms with mapping score {score_range}")  # noqa
    plt.xlabel(f"Proportion of atoms with score {score_range}")
    plt.ylabel("Mean number of atoms")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/mean_atoms_per_concept_{score_range}.png")

    fig, axes = plt.subplots(3, 4, figsize=(10, 10))
    binvals = list(num_atoms_per_bin.items())
    idx = 0
    for i in range(3):
        for j in range(4):
            binn, vals = binvals[idx]
            ax = axes[i, j]
            ax.hist(vals)
            ax.set_title(f"{binn} mapped")
            idx += 1
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(8)
    plt.savefig(f"{FIGDIR}/num_atoms_all_mapped_{score_range}.png")

    mean_num_cuis_per_bin = {binn: np.mean(val) for (binn, val)
                             in num_cuis_per_bin.items()}
    labels = list(mean_num_cuis_per_bin.keys())[::-1]
    heights = list(mean_num_cuis_per_bin.values())[::-1]
    pos = range(len(labels))
    plt.figure()
    plt.bar(pos, heights)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.title(f"Mean number of unique CUIs per concept by\nproportion of atoms with mapping score {score_range}.")  # noqa
    plt.xlabel(f"Proportion of atoms with score {score_range}")
    plt.ylabel("Mean number of unique CUIs")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/mean_cuis_per_concept_{score_range}.png")

    return binned_queries


def make_markdown_file(table1, table2s, outfile):

    with open(outfile, 'w') as outF:
        outF.write("# Number of Atoms by Mapping Score\n\n")
        outF.write(f"### Total number of atoms: {sum(table1.values())}\n")
        fig = f"![Figure 1]({FIGDIR}/matches_by_score.png)"
        outF.write(fig)
        outF.write('\n\n')

        colnames = [str(v) for v in table1.keys()][::-1]
        header = '| Mapping Score |' + '|'.join(colnames) + '|\n'
        header = header + '|' + '|'.join([" --- "] * (len(table1) + 1)) + '|\n'
        outF.write(header)
        rowvals = [str(v) for v in table1.values()][::-1]
        row = '| Num Atoms |' + '|'.join(rowvals) + '|\n'  # noqa
        outF.write(row)
        outF.write('\n\n')

        outF.write(f"# Summary of concepts\n\n")
        outF.write(f"### Total number of concepts: {sum(table2s[1000].values())}\n\n")  # noqa

        outF.write(f"## Number of concepts per mean score\n\n")
        fig = f"![Figure 1]({FIGDIR}/mean_score_per_concept.png)"
        outF.write(fig)
        outF.write("\n\n")

        for score in [1000, 800]:
            table = table2s[score]
            outF.write(f"## Number of Concepts by Proportion of Atoms with Score {score}\n\n")  # noqa
            fig = f"![Figure 1]({FIGDIR}/matches_by_concept_{score}.png)"
            outF.write(fig)
            outF.write('\n\n')

            colnames = list(table.keys())[::-1]
            header = '| % Mapped Atoms |' + '|'.join(colnames) + '|\n'
            header = header + '|' + '|'.join([" --- "] * (len(table) + 1)) + '|\n'  # noqa
            outF.write(header)
            rowvals = [str(v) for v in table.values()][::-1]
            row = '| Num Concepts |' + '|'.join(rowvals) + '|\n'  # noqa
            outF.write(row)
            outF.write('\n')

            fig = f"![Figure 3]({FIGDIR}/mean_atoms_per_concept_{score}.png)"
            outF.write(fig)
            outF.write('\n\n')

            fig = f"![Figure 4]({FIGDIR}/mean_cuis_per_concept_{score}.png)"
            outF.write(fig)
            outF.write('\n\n')


if __name__ == "__main__":
    args = parse_args()
    main(args.query_in, args.query_out,
         args.metamap_bin, args.outfile)

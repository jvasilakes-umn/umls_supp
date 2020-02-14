import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str, required=True,
                        help="CSV file of annotations.")
    parser.add_argument("--mrsty_file", type=str, required=True,
                        help="Path to UMLS MRSTY.RRF")
    parser.add_argument("--semgroups_file", type=str, required=True,
                        help="Path to UMLS semantic groups file.")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Where to save the summary.")
    return parser.parse_args()


def main(annotations_file, mrsty_file, semgroups_file, outfile):
    anns = pd.read_csv(annotations_file)
    anns.loc[:, "INGREDIENT TYPE"] = anns["INGREDIENT TYPE"].str.replace("LIVING BEING", "Living Beings")
    ambiguous = anns[pd.notnull(anns["IGNORE"])]
    annotated = anns[pd.notnull(anns["MISSINGNESS TYPE"])]

    sty2group = get_sty2group(mrsty_file, semgroups_file)
    annotated_grouped = annotated.merge(sty2group, on="UMLS CUI", how="left")
    null_idxs = pd.isnull(annotated_grouped["SemType"])
    annotated_grouped.loc[null_idxs, "SemGroup"] = annotated_grouped.loc[null_idxs, "INGREDIENT TYPE"]
    summary = summarize(ambiguous, annotated_grouped)
    with open(outfile, 'w') as outF:
        outF.write(summary)


def get_sty2group(mrsty_file, semgroups_file):
    mrsty = pd.read_csv(mrsty_file, sep='|', header=None, usecols=[0, 1])
    mrsty.columns = ["UMLS CUI", "SemType"]
    semgroups = pd.read_csv(semgroups_file, sep='|', header=None,
                            usecols=[1, 2])
    semgroups.columns = ["SemGroup", "SemType"]
    sty2group = mrsty.merge(semgroups, on="SemType").sort_values(by="UMLS CUI")
    sty2group.drop_duplicates(inplace=True)
    return sty2group


def summarize(unannotated, annotated):
    summary = "'Objects' and 'Anatomy' both get mapped to 'Living Beings'\n\n"
    total_unannotated = unannotated["IDISK AUI"].unique().shape[0]
    total_annotated = annotated["IDISK AUI"].unique().shape[0]
    total = total_unannotated + total_annotated
    summary += f"Total annotated atoms: {total}\n"
    summary += f"Number unannotated: {total_unannotated}\n\n"

    summary += f"Number annotated: {total_annotated}\n"
    counts = annotated.drop_duplicates(subset="IDISK AUI")["MISSINGNESS TYPE"].value_counts()  # noqa
    for (miss_type, count) in counts.items():
        summary += f"  {miss_type}: {count} ({count / total_annotated:.2f})\n"
    summary += '\n'

    for miss_type in annotated["MISSINGNESS TYPE"].unique():
        subset = annotated[annotated["MISSINGNESS TYPE"] == miss_type]
        total_miss = subset.drop_duplicates(subset="IDISK AUI").shape[0]
        summary += f"{miss_type}: {total_miss}\n"

        group_counts = subset.groupby("IDISK AUI")["SemGroup"].apply(set).value_counts()  # noqa
        for (group, count) in group_counts.items():
            summary += f"  {group}: {count} ({count / total_miss:.2f})\n"
    return summary


if __name__ == "__main__":
    args = parse_args()
    main(args.annotations_file, args.mrsty_file, args.semgroups_file,
         args.outfile)

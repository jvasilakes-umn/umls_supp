import os
import argparse
import datetime
import numpy as np
from glob import glob
from typing import List, Set, Tuple


"""
Author: Jake Vasilakes (vasil024@umn.edu)

Computes character-level Cohen's kappa and percentage
agreement for a set of brat annotated files from two
annotators for a sequence labeling task (e.g. NER).
"""


class BratANN(object):
    """
    A brat annotation.

    >>> ann = "T1\tent 1 4\tcat"
    >>> b1 = BratANN("T3", "ent", 1, 4, "cat")
    >>> b2 = BratANN.from_string(ann)
    >>> b1 == b2
    True
    >>> b3 = BratANN("T3", "ent", 1, 5, "cat ")
    >>> b1 == b3
    False
    """
    def __init__(self, num: str, label: str, start: int, end: int, text: str):
        self.num = num
        self.label = label
        self.start = int(start)
        self.end = int(end)
        self.text = text

    @classmethod
    def from_string(cls, string: str):
        (n, l, s, e, t) = string.split(maxsplit=4)
        return cls(n, l, int(s), int(e), t)

    def __str__(self) -> str:
        return f"{self.num}\t{self.label} {self.start} {self.end}\t{self.text}"  # noqa

    def __repr__(self) -> str:
        return f"<ira.BratANN '{self.num}, {self.label}, {self.start}, {self.end}, {self.text}'>"  # noqa

    def __eq__(self, other) -> bool:
        """
        Overrides the default implementation
        Two BratANNs are considering equal iff they have the same label,
        offset, and text.
        Equality does not consider the annotation number, e.g. T1
        """
        if isinstance(other, BratANN):
            return all([self.label == other.label,
                        self.start == other.start,
                        self.end == other.end,
                        self.text == other.text])
        else:
            return False


def parse_args():
    def usage():
        return """ira.py
            [--help, Show this help message and exit]
            [--test, Test the ira function]
            [--docdir, Directory containing the documents that were annotated.
                       If not specified, looks in indir1.]
            --indir1, Directory containing first annotators annotations
            --indir2, Directory containing second annotators annotations
            --annotation_conf, The brat annotation.conf that was used
                               for this annotation task
            --disagreements, Whether to suppress, print, or log files
                             in which annotators disagree. Possible values
                             are "suppress", "print", "log". Default is
                             "suppress". If "log", writes file names to
                             "disagreements.log" in the current working
                             directory.
            """

    desc = """Computes Cohen's kappa at the token
              level for a sequence labeling task."""
    parser = argparse.ArgumentParser(description=desc, usage=usage())
    parser.add_argument("--test", action="store_true", default=False,
                        help="""Test the ira function.""")
    args, remainder = parser.parse_known_args()
    if args.test is True:
        return args

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--indir1", type=str, required=True)
    parser.add_argument("--indir2", type=str, required=True)
    parser.add_argument("--annotation_conf", type=str, required=True)
    parser.add_argument("--docdir", type=str, required=False, default=None)
    parser.add_argument("--disagreements", type=str,
                        required=False,
                        default="suppress",
                        choices=["suppress", "print", "log"])
    args = parser.parse_args(remainder)
    args.test = False
    return args


def main(indir1: str, indir2: str, ann_conf: str,
         docdir: str = None, disagreements: str = "suppress"):
    """
    param indir{1,2}: Input directories containing the first and second
                           annotators .ann files, respectively.
    param ann_conf: Path to the annotation.conf file.
    param docdir: Directory containing the .txt files which were annotated.
                    If None, uses indir1.
    param disagreements: How disagreements are logged. Possible values are
                         "suppress", "print" and "log". If "suppress",
                         do nothing. If "print", prints files that disagree
                         to the console. If "log", files that disagree
                         will be written to "disagreements.log" in the current
                         working directory.
    """
    # Read in the documents.
    if docdir is not None:
        doc_fnames = glob(f"{docdir}/*.txt")
    else:
        doc_fnames = glob(f"{indir1}/*.txt")
    docs = read_docs(doc_fnames)
    # Read in the annotations.
    basenames = [os.path.splitext(os.path.basename(fn))[0]
                 for fn in doc_fnames]
    ann_fnames1 = [os.path.join(indir1, f"{bn}.ann") for bn in basenames]
    ann_fnames2 = [os.path.join(indir2, f"{bn}.ann") for bn in basenames]
    anns1 = read_anns(ann_fnames1)
    anns2 = read_anns(ann_fnames2)
    if not len(docs) == len(anns1) == len(anns2):
        raise ValueError("Different numbers of documents and annotations.")
    # Read the entity labels.
    labels = read_labels(ann_conf)
    # Compute inter rater agreement.
    kappa, agreement, disagree_idxs = ira(docs, anns1, anns2, labels)
    summary(kappa, "Cohen's Kappa")
    summary(agreement, "Percentage Agreement")
    # Do something with disagreements.
    if disagreements == "print":
        print("=== Disagreements ===")
        for (idx, p_o) in disagree_idxs:
            bn = os.path.basename(doc_fnames[idx])
            print(f"{bn}: Agreement={p_o:.3f}")
    if disagreements == "log":
        with open("disagreements.log", 'w') as outF:
            outF.write(str(datetime.datetime.now() + '\n'))
            for (idx, p_o) in disagree_idxs:
                bn = os.path.basename(doc_fnames[idx])
                outF.write(f"{bn}: Agreement={p_o:.3f}\n")


def read_docs(fnames: List[str]) -> List[str]:
    """
    Reads in the documents.

    param fnames: List of paths to .txt files to read.
    returns: List of input documents.
    """
    all_docs = []
    for docfile in fnames:
        doc = open(docfile, 'r').read()
        all_docs.append(doc)
    return all_docs


def read_anns(fnames: List[str]) -> List[List[BratANN]]:
    """
    Reads all .ann files and converts their
    annotations to BratANN objects.

    param fnames: List of paths to .ann files to read.
    returns: List of annotations.
    """
    all_anns = []
    for annfile in fnames:
        anns = [BratANN.from_string(a.strip()) for a in open(annfile, 'r')]
        all_anns.append(anns)
    return all_anns


def read_labels(ann_conf: str) -> Set[str]:
    """
    Reads the entity labels from annotation.conf.

    param ann_conf: Path to annotation.conf
    returns: set of entity labels.
    """
    labels = set()
    with open(ann_conf, 'r') as infile:
        copy = False
        for line in infile:
            # Skip blank lines and comments.
            if not line.strip() or line.strip().startswith('#'):
                continue
            if line.strip() == "[entities]":
                copy = True
            elif line.strip() == "[relations]":
                copy = False
            elif copy is True:
                labels.add(line.strip())
    return labels


def ira(docs: List[str],
        anns1: List[List[BratANN]],
        anns2: List[List[BratANN]],
        labels: Set[str]) -> Tuple[np.array, np.array, List[Tuple[int, float]]]:  # noqa
    """
    Computes Cohen's kappa and percentage agreement between two annotators.

    param docs: List of documents, output of read_docs().
    param anns1: List of first annotators annotations, output of read_anns().
    param anns2: List of second annotators annotations, output of read_anns().
    param labels: Set of labels annotated, output of read_labels().
    returns: Kappa and percentage agreement for each document.
    """
    n_docs = len(docs)
    p_os = np.zeros(n_docs)
    kappas = np.zeros(n_docs)
    disagree_idxs_po = []
    for i in range(n_docs):
        denom = len(docs[i])
        v1 = label_vector(docs[i], anns1[i], labels)
        v2 = label_vector(docs[i], anns2[i], labels)
        # Observed agreement: How often the two annotators actually agreed.
        #   Equivalent to accuracy.
        p_o = np.sum(v1 == v2) / denom
        if p_o != 1.0:
            disagree_idxs_po.append((i, p_o))
        # Expected agreement: How often the two annotators are expected to
        #   agree. For number of items N, labels k, and the number of times
        #   rater j predicted label k, n_j_k:
        #   p_e = (1/N^2) * sum_k (n_1_k * n_2_k)
        p_e = (1/denom**2) * np.sum([np.sum(v1 == k) * np.sum(v2 == k)
                                     for k in range(len(labels)+1)])
        if p_e == 1:
            k = 0.0
        else:
            k = (p_o - p_e) / (1 - p_e)
        p_os[i] = p_o
        kappas[i] = k
    return (kappas, p_os, disagree_idxs_po)


def label_vector(doc: List[str],
                 anns: List[List[BratANN]],
                 labels: Set[str]) -> np.array:
    """
    Converts the document into an integer vector. The value
    of each element corresponds to the entity type of the
    annotation at that character position, with 0 indicating
    no annotation. So an annotation task with 3 annotation types
    would have a vector of 0s, 1s, 2s, and 3s.

    param doc: Document that was annotated.
    param anns: Annotations for each document.
    param labels: Set of entity labels for this task.
    returns: Vector of character level annotations.
    """
    v = np.zeros(len(doc))  # For each character
    for (i, lab) in enumerate(labels):
        i += 1  # 0 is reserved for no label
        idxs = [np.arange(a.start, a.end) for a in anns if a.label == lab]
        idxs = [j for mask in idxs for j in mask]
        v[idxs] = i
    return v


def summary(results: np.array, varname: str = None):
    """
    Prints summary statistics for the supplied results.

    param results: Numeric array of results (e.g. kappas).
    param varname: (Optional) Name of the variable being summarized.
    """
    if varname is not None:
        print(varname)
    if len(results) == 1:
        print(f"{results[0]:.3f}")
    else:
        rmean = np.mean(results)
        rmax = np.max(results)
        rmin = np.min(results)
        rstd = np.std(results)
        print(f"""Mean: {rmean:.3f} +/-{rstd:.3f}\nRange: ({rmin:.3f}, {rmax:.3f})""")  # noqa


def test():
    """
    A small example to test ira().
    """
    docs = ["The cats sat on the mat"]
    ann_strs1 = ["T1\tent 4 8\tcats",
                 "T2\tent 9 12\tsat",
                 "T3\tent 20 23\tmat"]
    anns1 = [[BratANN.from_string(s) for s in ann_strs1]]
    ann_strs2 = ["T1\tent 4 7\tcat", "T2\tent 20 23 mat"]
    anns2 = [[BratANN.from_string(s) for s in ann_strs2]]
    labels = ["ent"]
    kappas, agreements, disagreements = ira(docs, anns1, anns2, labels)
    assert(np.isclose(kappas[0], 0.629, atol=1e-03))
    assert(np.isclose(agreements[0], 0.826, atol=1e-03))
    print("All tests passed.")


if __name__ == "__main__":
    args = parse_args()
    if args.test is True:
        import doctest
        doctest.testmod()
        test()
    else:
        main(args.indir1, args.indir2, args.annotation_conf,
             docdir=args.docdir, disagreements=args.disagreements)

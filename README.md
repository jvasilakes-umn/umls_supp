# Extending the UMLS with Dietary Supplement Terms


## Set up

If you have cloned this repo from Github with the intention of reproducing the results in the
paper, you'll first need to clean up the existing results with 

```
make -f Makefile.example clean_all
```

First, ensure you have a working Python 3 installation and a virtual environment manager such
as `conda` or `virtualenv`. You'll also need the following tools:

 * [LVG](https://lsg3.nlm.nih.gov/LexSysGroup/Projects/lvg/current/web/download.html): The "lite" version is fine, as these experiments just use the `luiNorm` binary.
 * [UMLS Metathesaurus data files](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html): You do not need the full release, as these experiments only use `MRCONSO.RRF`, `MRSTY.RRF`, and `MRREL.RRF`.
 * [UMLS Semantic Groups file](https://metamap.nlm.nih.gov/Docs/SemGroups_2018.txt)
 * [iDISK 1.0.1](https://github.com/jvasilakes/idisk/releases/tag/v1.0.1): You don't need the source code, just the release `.zip` file. Unzip this somewhere to obtain the required iDISK data files.

Once you've downloaded these, edit the relevant variables in the `PROJECT CONFIGURATION` section of `Makefile.example` to point to their respective locations. Check that these file paths are correct to avoid wasting time later:

```
make check_files
```

If doesn't output anything, then all is well and you can install all the dependencies with

```
make requirements
```


## Experiments

There are two sets of experiments. The first matches the iDISK dietary supplement ingredient
terminology to the UMLS using exact string matching. The second runs a basic dictionary-based
NER system over a set of PubMed abstracts. To run the full experiment pipeline and reproduce
the results from the paper run

```
make experiments
```

This will take around 30 minutes.

Details on each set of experiments are below.


### Coverage Experiments

We attempt to match all dietary supplement ingredient terms
from the iDISK release to the UMLS using exact string matching
and normalized string matching (via luiNorm).

We perform a qualitative analysis of the unmatched terms.

The following command will perform the matching and generate
summary statistics and plots in the `coverage/results/` directory.

```
make coverage_run_all
```


### Named Entity Recognition Experiments

To demonstrate the practical contribution of the iDISK dietary supplement
ingredient terminology, we compare the performance of three simple dictionary-based
NER systems built using one of three term lists:

 1. UMLS\_DS: Dietary supplement concepts extracted from the UMLS. These were obtained by computing
    the transitive closure of the UMLS parent-child hierarchy starting from the
    "Dietary Supplements" (C0242295) and "Vitamin" (C0042890) concepts.

 2. iDISK: The dietary supplement ingredient term list lifted directly from iDISK.

 3. Combined: The union of the above two term lists.

The NER predictions were evaluated on 398 PubMed abstracts annotated for mentions of dietary supplement ingredients.
We report precision, recall, and F1 score. These metrics can be found in the `ner/results/` directory.

```
make ner_prepare_all
make ner_run_all
```


## Some extra information

The experiments above limit the UMLS Concepts used according to their semantic type. The semantic types we keep are below.

```
Amino Acid, Peptide, or Protein|aapp|T116|Chemicals & Drugs|CHEM
Biologically Active Substance|bacs|T123|Chemicals & Drugs|CHEM
Chemical|chem|T103|Chemicals & Drugs|CHEM
Element, Ion, or Isotope|elii|T196|Chemicals & Drugs|CHEM
Hormone|horm|T125|Chemicals & Drugs|CHEM
Immunologic Factor|imft|T129|Chemicals & Drugs|CHEM
Inorganic Chemical|inch|T197|Chemicals & Drugs|CHEM
Nucleic Acid, Nucleoside, or Nucleotide|nnon|T114|Chemicals & Drugs|CHEM
Organic Chemical|orch|T109|Chemicals & Drugs|CHEM
Pharmacologic Substance|phsu|T121|Chemicals & Drugs|CHEM
Vitamin|vita|T127|Chemicals & Drugs|CHEM
Bacterium|bact|T007|Living Beings|LIVB
Fish|fish|T013|Living Beings|LIVB
Fungus|fngs|T004|Living Beings|LIVB
Plant|plnt|T002|Living Beings|LIVB
Food|food|T168|Objects|OBJC
```

The exact versions of the software and data files used are listed below:

```
python==3.7.6
umls_metathesaurus==2019AB
idisk==1.0.1
idlib==0.0.1
matplotlib==3.1.2
numpy==1.18.1
quickumls==1.3.0.post4
quickumls-simstring==1.1.5.post1
spacy==2.2.3
editdistance==0.5.3
```

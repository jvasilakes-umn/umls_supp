This directory is meant to contain the UMLS-style RRF files and the QuickUMLS
installations generated from them. These were not included in this repository due to
their size, but can be regenerated via the following `make` recipes.

```
make ner_prepare_term_lists
make ner_prepare_models
```

These commands assume you have followed the instructions in the main README by downloading
and installing the UMLS Metathesaurus, the iDISK release, and the various software requirements.

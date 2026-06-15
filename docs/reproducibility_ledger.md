# Reproducibility Ledger

Every server artifact must record:

- git commit and dirty status;
- model checkpoint/API id and access date;
- dataset manifest and split hash;
- prompt/trace extractor config hash;
- seed list;
- baseline source and commit;
- output hashes and row counts;
- evaluator command;
- ARIS evidence tier.

No paper result may bypass this ledger.


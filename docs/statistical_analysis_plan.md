# Statistical Analysis Plan

## Main Test

Use paired bootstrap over examples for intervention deltas. Report mean, 95
percent confidence interval, p-value, and paired effect size.

## Multiple Comparisons

Apply Holm correction for detection, intervention, transfer, and utility claims.

## Seeds And Replicates

Paper-facing results require at least 20 seeds or bootstrap replicates. Pilot
runs below this threshold are `diagnostic` or `pilot`, never `paper_result`.

## Failure Handling

All failed runs must remain in the provenance ledger. Excluding a failed run
requires a documented non-method cause and rerun under the same seed policy.


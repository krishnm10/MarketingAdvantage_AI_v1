# Dedup UI Sample Pack

These files are designed to help you validate the ingestion deduplication UI.

## Suggested Ingest Order

1. `01_unique_baseline.txt`
2. `02_l1_exact_repeat.txt`
3. `03_l1_one_unique_many_dups.txt`
4. `04_seed_policy.txt`
5. `05_l2_cross_file.txt`
6. `06_seed_marketing.txt`
7. `07_mixed_l1_l2.txt`
8. `08_seed_semantic.txt`
9. `09_l3_candidate.txt`
10. `10_sales_rows.csv`
11. `11_policy_repeat.txt`
12. `12_business_mix.txt`

## Expected Outcome Summary

- `01_unique_baseline.txt`
  Clean file with mostly unique content.
- `02_l1_exact_repeat.txt`
  Same-file exact repeats should light up L1.
- `03_l1_one_unique_many_dups.txt`
  Good for testing one survivor plus many duplicates.
- `04_seed_policy.txt` then `05_l2_cross_file.txt`
  Use these together to test L2 cross-file behavior.
- `06_seed_marketing.txt` then `07_mixed_l1_l2.txt`
  Use these together to test mixed duplicate lanes.
- `08_seed_semantic.txt` then `09_l3_candidate.txt`
  Use these together to try L3 semantic-review behavior.
- `10_sales_rows.csv`
  Row-based duplicate testing.
- `11_policy_repeat.txt`
  Paragraph repeat testing.
- `12_business_mix.txt`
  Easy business-readable mixed unique/duplicate file.

## UI Checks

For each ingested file, verify:

1. File name is shown correctly.
2. `Total Chunks`, `Unique`, `Duplicates`, and `Dedup Ratio` look reasonable.
3. Only active lanes are emphasized.
4. Duplicate rows are highlighted by lane color.
5. `Dedup Match Review` shows:
   - ingestion file name
   - matched against
   - matching semantics
   - duplicate value
6. `Unique Values Table` shows surviving rows clearly.

## Important Note

L3 depends on what the current backend returns. If `09_l3_candidate.txt` does not show L3, that means the backend did not expose that semantic-review case for this dataset.


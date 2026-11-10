# Qwen3.8-27B RULER scores by context length

Each table reports scores on a 0–100 scale for 13 tasks, with 50 examples
per task at each length and 3,250 examples per method across all five
lengths. The final row is the unweighted mean of the 13 task scores
at that length. We compute means before rounding to two decimal places.

The first six methods use the cell scores from the
[RULER audit CSV](ruler27b_task_length.csv). KVzip uses the archived
[27B RULER task and length cells](kvzip_reads.json). The
[RULER protocol](ruler27b_summary.json) and
[budget protocol](../budget_protocol.md) describe scoring and budget calibration.

## RULER 8K

| Category | Task | Full Attention | PrismKV | Quest | SnapKV | PyramidKV | H2O | KVzip |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrieval | `niah_single_1` | 100.00 | 100.00 | 8.00 | 48.00 | 14.00 | 0.00 | 100.00 |
| Retrieval | `niah_single_2` | 100.00 | 100.00 | 98.00 | 82.00 | 26.00 | 2.00 | 0.00 |
| Retrieval | `niah_single_3` | 100.00 | 100.00 | 80.00 | 2.00 | 0.00 | 2.00 | 0.00 |
| Retrieval | `niah_multikey_1` | 100.00 | 100.00 | 98.00 | 76.00 | 30.00 | 4.00 | 0.00 |
| Retrieval | `niah_multikey_2` | 100.00 | 100.00 | 96.00 | 100.00 | 92.00 | 4.00 | 10.00 |
| Retrieval | `niah_multikey_3` | 100.00 | 100.00 | 72.00 | 0.00 | 0.00 | 2.00 | 0.00 |
| Retrieval | `niah_multivalue` | 100.00 | 100.00 | 100.00 | 29.00 | 6.00 | 2.00 | 0.00 |
| Retrieval | `niah_multiquery` | 100.00 | 100.00 | 99.50 | 19.50 | 0.50 | 2.00 | 0.00 |
| Variable tracking | `vt` | 100.00 | 100.00 | 97.20 | 59.20 | 11.60 | 66.40 | 100.00 |
| Aggregation | `cwe` | 100.00 | 100.00 | 99.60 | 91.40 | 25.20 | 98.80 | 41.20 |
| Aggregation | `fwe` | 100.00 | 99.33 | 98.67 | 85.33 | 26.67 | 99.33 | 84.00 |
| Question answering | `qa_1` | 88.00 | 88.00 | 86.00 | 86.00 | 86.00 | 54.00 | 64.00 |
| Question answering | `qa_2` | 72.00 | 72.00 | 72.00 | 72.00 | 68.00 | 52.00 | 56.00 |
| | **Macro average** | 96.92 | 96.87 | 85.00 | 57.73 | 29.69 | 29.89 | 35.02 |

## RULER 16K

| Category | Task | Full Attention | PrismKV | Quest | SnapKV | PyramidKV | H2O | KVzip |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrieval | `niah_single_1` | 100.00 | 100.00 | 38.00 | 100.00 | 66.00 | 28.00 | 100.00 |
| Retrieval | `niah_single_2` | 100.00 | 100.00 | 98.00 | 100.00 | 84.00 | 2.00 | 0.00 |
| Retrieval | `niah_single_3` | 100.00 | 100.00 | 80.00 | 4.00 | 0.00 | 4.00 | 0.00 |
| Retrieval | `niah_multikey_1` | 100.00 | 100.00 | 100.00 | 84.00 | 54.00 | 4.00 | 0.00 |
| Retrieval | `niah_multikey_2` | 100.00 | 100.00 | 98.00 | 100.00 | 100.00 | 12.00 | 8.00 |
| Retrieval | `niah_multikey_3` | 100.00 | 100.00 | 66.00 | 24.00 | 0.00 | 2.00 | 0.00 |
| Retrieval | `niah_multivalue` | 100.00 | 100.00 | 100.00 | 54.50 | 16.50 | 0.00 | 0.00 |
| Retrieval | `niah_multiquery` | 100.00 | 100.00 | 100.00 | 71.50 | 2.50 | 1.50 | 0.00 |
| Variable tracking | `vt` | 100.00 | 100.00 | 97.60 | 87.60 | 14.40 | 87.20 | 100.00 |
| Aggregation | `cwe` | 100.00 | 100.00 | 99.60 | 99.40 | 43.60 | 99.20 | 43.60 |
| Aggregation | `fwe` | 100.00 | 99.33 | 100.00 | 94.00 | 58.00 | 100.00 | 81.33 |
| Question answering | `qa_1` | 86.00 | 86.00 | 84.00 | 86.00 | 86.00 | 56.00 | 70.00 |
| Question answering | `qa_2` | 74.00 | 74.00 | 74.00 | 74.00 | 74.00 | 56.00 | 60.00 |
| | **Macro average** | 96.92 | 96.87 | 87.32 | 75.31 | 46.08 | 34.76 | 35.61 |

## RULER 32K

| Category | Task | Full Attention | PrismKV | Quest | SnapKV | PyramidKV | H2O | KVzip |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrieval | `niah_single_1` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 70.00 | 100.00 |
| Retrieval | `niah_single_2` | 100.00 | 100.00 | 100.00 | 100.00 | 84.00 | 4.00 | 0.00 |
| Retrieval | `niah_single_3` | 100.00 | 100.00 | 92.00 | 58.00 | 4.00 | 4.00 | 0.00 |
| Retrieval | `niah_multikey_1` | 100.00 | 100.00 | 100.00 | 100.00 | 76.00 | 6.00 | 0.00 |
| Retrieval | `niah_multikey_2` | 100.00 | 98.00 | 100.00 | 100.00 | 100.00 | 2.00 | 6.00 |
| Retrieval | `niah_multikey_3` | 100.00 | 100.00 | 82.00 | 54.00 | 0.00 | 8.00 | 0.00 |
| Retrieval | `niah_multivalue` | 100.00 | 100.00 | 100.00 | 82.00 | 29.50 | 1.00 | 1.00 |
| Retrieval | `niah_multiquery` | 100.00 | 100.00 | 100.00 | 100.00 | 34.50 | 2.50 | 0.00 |
| Variable tracking | `vt` | 100.00 | 100.00 | 98.00 | 98.00 | 30.80 | 92.40 | 100.00 |
| Aggregation | `cwe` | 100.00 | 100.00 | 99.60 | 99.20 | 59.40 | 98.00 | 65.40 |
| Aggregation | `fwe` | 100.00 | 100.00 | 100.00 | 97.33 | 51.33 | 100.00 | 72.00 |
| Question answering | `qa_1` | 82.00 | 82.00 | 80.00 | 82.00 | 82.00 | 54.00 | 64.00 |
| Question answering | `qa_2` | 74.00 | 74.00 | 74.00 | 74.00 | 72.00 | 62.00 | 62.00 |
| | **Macro average** | 96.62 | 96.46 | 94.28 | 88.04 | 55.66 | 38.76 | 36.18 |

## RULER 64K

| Category | Task | Full Attention | PrismKV | Quest | SnapKV | PyramidKV | H2O | KVzip |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrieval | `niah_single_1` | 100.00 | 100.00 | 98.00 | 100.00 | 100.00 | 88.00 | 100.00 |
| Retrieval | `niah_single_2` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 2.00 | 0.00 |
| Retrieval | `niah_single_3` | 100.00 | 100.00 | 88.00 | 98.00 | 4.00 | 6.00 | 0.00 |
| Retrieval | `niah_multikey_1` | 100.00 | 100.00 | 100.00 | 100.00 | 84.00 | 6.00 | 0.00 |
| Retrieval | `niah_multikey_2` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 2.00 | 6.00 |
| Retrieval | `niah_multikey_3` | 100.00 | 98.00 | 86.00 | 84.00 | 12.00 | 0.00 | 0.00 |
| Retrieval | `niah_multivalue` | 100.00 | 100.00 | 100.00 | 100.00 | 48.00 | 3.50 | 0.00 |
| Retrieval | `niah_multiquery` | 100.00 | 99.50 | 100.00 | 100.00 | 82.50 | 4.00 | 0.00 |
| Variable tracking | `vt` | 99.60 | 99.60 | 99.60 | 99.60 | 61.20 | 99.20 | 99.60 |
| Aggregation | `cwe` | 99.80 | 100.00 | 99.80 | 99.80 | 89.80 | 100.00 | 72.60 |
| Aggregation | `fwe` | 99.33 | 98.67 | 99.33 | 98.00 | 78.00 | 100.00 | 80.00 |
| Question answering | `qa_1` | 88.00 | 88.00 | 88.00 | 88.00 | 88.00 | 58.00 | 72.00 |
| Question answering | `qa_2` | 72.00 | 74.00 | 72.00 | 72.00 | 72.00 | 60.00 | 62.00 |
| | **Macro average** | 96.83 | 96.75 | 94.67 | 95.34 | 70.73 | 40.67 | 37.86 |

## RULER 128K

| Category | Task | Full Attention | PrismKV | Quest | SnapKV | PyramidKV | H2O | KVzip |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Retrieval | `niah_single_1` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 94.00 | 100.00 |
| Retrieval | `niah_single_2` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 6.00 | 0.00 |
| Retrieval | `niah_single_3` | 100.00 | 100.00 | 94.00 | 100.00 | 16.00 | 6.00 | 0.00 |
| Retrieval | `niah_multikey_1` | 100.00 | 100.00 | 100.00 | 100.00 | 98.00 | 8.00 | 0.00 |
| Retrieval | `niah_multikey_2` | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 2.00 | 4.00 |
| Retrieval | `niah_multikey_3` | 100.00 | 100.00 | 94.00 | 100.00 | 60.00 | 2.00 | 0.00 |
| Retrieval | `niah_multivalue` | 100.00 | 100.00 | 100.00 | 100.00 | 89.00 | 2.00 | 1.00 |
| Retrieval | `niah_multiquery` | 100.00 | 100.00 | 100.00 | 100.00 | 97.50 | 5.00 | 0.00 |
| Variable tracking | `vt` | 100.00 | 100.00 | 100.00 | 100.00 | 88.00 | 100.00 | 100.00 |
| Aggregation | `cwe` | 100.00 | 100.00 | 99.80 | 100.00 | 90.60 | 99.60 | 81.40 |
| Aggregation | `fwe` | 100.00 | 99.33 | 99.33 | 98.67 | 92.67 | 100.00 | 80.67 |
| Question answering | `qa_1` | 86.00 | 86.00 | 86.00 | 86.00 | 86.00 | 54.00 | 64.00 |
| Question answering | `qa_2` | 68.00 | 68.00 | 68.00 | 68.00 | 68.00 | 62.00 | 62.00 |
| | **Macro average** | 96.46 | 96.41 | 95.47 | 96.36 | 83.52 | 41.58 | 37.93 |

# AMI Train Split — Smallest & Largest Meetings

Reference size statistics across all **136** meetings in `./shared/datasets/amicorpus/train`, ranked by **number of segment lines** (ties broken by GT spoken word count).

- **Segments (Lines):** Number of diarized utterances in `parsed_diarized_gt.txt` (row-aligned with `custom_transcript_gt_segments.txt` and `beam_results_2.json`).
- **Spoken Words:** Whitespace-delimited words after stripping the `(START - END)[Speaker X]: ` prefix.
- **Total Words (w/ prefix):** Whitespace-delimited words for the entire file, including timestamps and speaker tags.
- **ASR (`custom_transcript_gt_segments`):** Corresponding word counts from Whisper beam-1 transcripts.

---

## Top 15 Smallest Meetings (by Segment Lines)

| Rank | Meeting | Segments (Lines) | GT Spoken Words | GT Total Words (w/ prefix) | GT Bytes | ASR Spoken Words | ASR Total Words (w/ prefix) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `ES2005a` | 92 | 747 | 1,119 | 6,652 | 752 | 1,124 |
| 2 | `IS1004a` | 135 | 1,537 | 2,081 | 11,816 | 1,377 | 1,920 |
| 3 | `ES2012a` | 143 | 2,781 | 3,357 | 18,541 | 2,325 | 2,901 |
| 4 | `ES2003a` | 160 | 2,027 | 2,671 | 15,447 | 1,773 | 2,417 |
| 5 | `ES2013a` | 177 | 1,542 | 2,254 | 13,293 | 2,092 | 2,803 |
| 6 | `TS3010a` | 203 | 1,100 | 1,916 | 11,375 | 1,415 | 2,231 |
| 7 | `ES2010a` | 203 | 1,458 | 2,274 | 13,855 | 1,515 | 2,330 |
| 8 | `IS1001a` | 215 | 1,755 | 2,619 | 15,300 | 1,730 | 2,594 |
| 9 | `IS1005a` | 224 | 1,642 | 2,542 | 14,921 | 1,605 | 2,505 |
| 10 | `ES2012d` | 232 | 2,587 | 3,519 | 19,744 | 2,505 | 3,436 |
| 11 | `ES2008a` | 239 | 2,495 | 3,455 | 20,326 | 2,427 | 3,385 |
| 12 | `ES2014a` | 242 | 2,197 | 3,169 | 18,684 | 2,072 | 3,044 |
| 13 | `IS1003a` | 257 | 1,477 | 2,509 | 15,232 | 1,601 | 2,631 |
| 14 | `IS1007a` | 258 | 1,936 | 2,972 | 17,662 | 2,287 | 3,322 |
| 15 | `ES2006a` | 267 | 2,731 | 3,803 | 22,348 | 2,584 | 3,656 |

---

## Top 15 Largest Meetings (by Segment Lines)

| Rank | Meeting | Segments (Lines) | GT Spoken Words | GT Total Words (w/ prefix) | GT Bytes | ASR Spoken Words | ASR Total Words (w/ prefix) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `EN2009d` | 2,056 | 18,042 | 26,270 | 158,567 | 20,714 | 28,941 |
| 2 | `TS3006d` | 1,440 | 8,217 | 13,981 | 86,486 | 8,932 | 14,696 |
| 3 | `IN1013` | 1,405 | 10,963 | 16,587 | 101,349 | 12,986 | 18,609 |
| 4 | `TS3005d` | 1,355 | 9,038 | 14,462 | 87,312 | 9,688 | 15,108 |
| 5 | `EN2001a` | 1,335 | 16,006 | 21,350 | 125,299 | 16,728 | 22,068 |
| 6 | `EN2006a` | 1,330 | 9,021 | 14,345 | 89,019 | 10,833 | 16,157 |
| 7 | `EN2006b` | 1,306 | 6,833 | 12,061 | 77,279 | 7,791 | 13,019 |
| 8 | `IN1016` | 1,304 | 11,673 | 16,893 | 100,722 | 12,039 | 17,259 |
| 9 | `EN2005a` | 1,260 | 15,246 | 20,290 | 120,599 | 15,628 | 20,672 |
| 10 | `EN2004a` | 1,191 | 9,889 | 14,657 | 88,837 | 10,601 | 15,365 |
| 11 | `IN1014` | 1,138 | 10,616 | 15,172 | 88,366 | 11,156 | 15,711 |
| 12 | `TS3007d` | 1,133 | 7,684 | 12,220 | 73,966 | 8,177 | 12,713 |
| 13 | `TS3006c` | 1,123 | 7,367 | 11,863 | 72,458 | 7,946 | 12,442 |
| 14 | `IN1007` | 1,077 | 8,029 | 12,341 | 74,251 | 9,342 | 13,654 |
| 15 | `IN1001` | 1,074 | 9,381 | 13,681 | 80,682 | 9,119 | 13,418 |

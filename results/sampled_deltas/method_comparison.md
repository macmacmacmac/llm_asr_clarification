# Teacher-metric comparison (train pilot meetings)

Each method ranks all lines (mistranscription detector = ALL, so ranking is importance only); the top N are
replaced with GT by `clarifications.pipelinev4`; the quiz is answered + graded by Qwen3-32B-FP8 greedy
(`quiz.bulk_vllm_offline_quiz`, the same setup as the official ASR/GT scores). 10 questions per meeting.

## All 4 meetings (no sampled): ES2005d, IN1012, IS1007c, TS3005d

ASR transcript: 33/40; GT transcript: 37/40.

### Gain vs ASR (points)

| method            |   10 |   20 |   30 |   40 |   50 |
|:------------------|-----:|-----:|-----:|-----:|-----:|
| cosine (previous) |   -3 |    2 |    1 |    2 |    3 |
| logp 8B           |    2 |    2 |    3 |    2 |    2 |
| logp 32B          |    3 |    1 |    3 |    4 |    4 |
| random s47        |   -2 |   -1 |   -1 |   -1 |   -2 |
| random s1         |   -3 |    0 |    1 |    0 |    0 |
| random s2         |   -3 |    1 |   -2 |   -1 |   -2 |

### Questions fixed that the GT transcript also gets right / questions broken

| method            | 10    | 20    | 30    | 40    | 50    |
|:------------------|:------|:------|:------|:------|:------|
| cosine (previous) | 1 / 4 | 1 / 0 | 1 / 1 | 2 / 1 | 5 / 2 |
| logp 8B           | 2 / 0 | 3 / 1 | 3 / 0 | 3 / 1 | 3 / 1 |
| logp 32B          | 3 / 0 | 3 / 2 | 4 / 2 | 4 / 0 | 4 / 0 |
| random s47        | 0 / 2 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 2 |
| random s1         | 0 / 3 | 2 / 2 | 2 / 1 | 1 / 1 | 1 / 1 |
| random s2         | 0 / 3 | 2 / 1 | 2 / 4 | 1 / 2 | 1 / 3 |

## 3 common meetings (all methods): ES2005d, IN1012, TS3005d

ASR transcript: 24/30; GT transcript: 27/30.

### Gain vs ASR (points)

| method                      |   10 |   20 |   30 |   40 |   50 |
|:----------------------------|-----:|-----:|-----:|-----:|-----:|
| cosine (previous)           |   -3 |    1 |    0 |    1 |    2 |
| sampled 8B (expected score) |   -1 |   -2 |   -1 |   -1 |   -1 |
| logp 8B                     |    1 |    1 |    2 |    1 |    1 |
| logp 32B                    |    2 |    0 |    2 |    3 |    3 |
| random s47                  |   -1 |   -1 |   -1 |   -1 |   -2 |
| random s1                   |   -3 |   -1 |    0 |    0 |    0 |
| random s2                   |   -2 |    0 |   -2 |    0 |   -1 |

### Questions fixed that the GT transcript also gets right / questions broken

| method                      | 10    | 20    | 30    | 40    | 50    |
|:----------------------------|:------|:------|:------|:------|:------|
| cosine (previous)           | 0 / 3 | 0 / 0 | 0 / 1 | 1 / 1 | 4 / 2 |
| sampled 8B (expected score) | 1 / 2 | 0 / 2 | 0 / 1 | 0 / 1 | 2 / 3 |
| logp 8B                     | 1 / 0 | 2 / 1 | 2 / 0 | 2 / 1 | 2 / 1 |
| logp 32B                    | 2 / 0 | 2 / 2 | 3 / 2 | 3 / 0 | 3 / 0 |
| random s47                  | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 2 |
| random s1                   | 0 / 3 | 1 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| random s2                   | 0 / 2 | 1 / 1 | 1 / 3 | 1 / 1 | 1 / 2 |

## Per meeting

| meeting   | method                      |   N |   score |   fixed |   fixed (GT-fixable) |   broken | fixed qs   | broken qs   |
|:----------|:----------------------------|----:|--------:|--------:|---------------------:|---------:|:-----------|:------------|
| ES2005d   | cosine (previous)           |  10 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | cosine (previous)           |  20 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | cosine (previous)           |  30 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | cosine (previous)           |  40 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | cosine (previous)           |  50 |      10 |       3 |                    3 |        0 | [2, 3, 6]  | []          |
| ES2005d   | sampled 8B (expected score) |  10 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | sampled 8B (expected score) |  20 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | sampled 8B (expected score) |  30 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | sampled 8B (expected score) |  40 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | sampled 8B (expected score) |  50 |       9 |       2 |                    2 |        0 | [3, 6]     | []          |
| ES2005d   | logp 8B                     |  10 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | logp 8B                     |  20 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | logp 8B                     |  30 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | logp 8B                     |  40 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | logp 8B                     |  50 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | logp 32B                    |  10 |       9 |       2 |                    2 |        0 | [3, 6]     | []          |
| ES2005d   | logp 32B                    |  20 |       9 |       2 |                    2 |        0 | [2, 6]     | []          |
| ES2005d   | logp 32B                    |  30 |      10 |       3 |                    3 |        0 | [2, 3, 6]  | []          |
| ES2005d   | logp 32B                    |  40 |      10 |       3 |                    3 |        0 | [2, 3, 6]  | []          |
| ES2005d   | logp 32B                    |  50 |      10 |       3 |                    3 |        0 | [2, 3, 6]  | []          |
| ES2005d   | random s47                  |  10 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s47                  |  20 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s47                  |  30 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s47                  |  40 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s47                  |  50 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s1                   |  10 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s1                   |  20 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | random s1                   |  30 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | random s1                   |  40 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | random s1                   |  50 |       8 |       1 |                    1 |        0 | [3]        | []          |
| ES2005d   | random s2                   |  10 |       7 |       0 |                    0 |        0 | []         | []          |
| ES2005d   | random s2                   |  20 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | random s2                   |  30 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | random s2                   |  40 |       8 |       1 |                    1 |        0 | [2]        | []          |
| ES2005d   | random s2                   |  50 |       8 |       1 |                    1 |        0 | [2]        | []          |
| IN1012    | cosine (previous)           |  10 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | cosine (previous)           |  20 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | cosine (previous)           |  30 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | cosine (previous)           |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | cosine (previous)           |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | sampled 8B (expected score) |  10 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | sampled 8B (expected score) |  20 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | sampled 8B (expected score) |  30 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | sampled 8B (expected score) |  40 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | sampled 8B (expected score) |  50 |       7 |       0 |                    0 |        2 | []         | [2, 9]      |
| IN1012    | logp 8B                     |  10 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 8B                     |  20 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 8B                     |  30 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 8B                     |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 8B                     |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 32B                    |  10 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 32B                    |  20 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IN1012    | logp 32B                    |  30 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IN1012    | logp 32B                    |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | logp 32B                    |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | random s47                  |  10 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IN1012    | random s47                  |  20 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s47                  |  30 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | random s47                  |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | random s47                  |  50 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s1                   |  10 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s1                   |  20 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IN1012    | random s1                   |  30 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | random s1                   |  40 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s1                   |  50 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s2                   |  10 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s2                   |  20 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s2                   |  30 |       8 |       0 |                    0 |        1 | []         | [9]         |
| IN1012    | random s2                   |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IN1012    | random s2                   |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | cosine (previous)           |  10 |       9 |       1 |                    1 |        1 | [2]        | [3]         |
| IS1007c   | cosine (previous)           |  20 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | cosine (previous)           |  30 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | cosine (previous)           |  40 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | cosine (previous)           |  50 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 8B                     |  10 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 8B                     |  20 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 8B                     |  30 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 8B                     |  40 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 8B                     |  50 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 32B                    |  10 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 32B                    |  20 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 32B                    |  30 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 32B                    |  40 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | logp 32B                    |  50 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | random s47                  |  10 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IS1007c   | random s47                  |  20 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s47                  |  30 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s47                  |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s47                  |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s1                   |  10 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s1                   |  20 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | random s1                   |  30 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | random s1                   |  40 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s1                   |  50 |       9 |       0 |                    0 |        0 | []         | []          |
| IS1007c   | random s2                   |  10 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IS1007c   | random s2                   |  20 |      10 |       1 |                    1 |        0 | [2]        | []          |
| IS1007c   | random s2                   |  30 |       9 |       1 |                    1 |        1 | [2]        | [3]         |
| IS1007c   | random s2                   |  40 |       8 |       0 |                    0 |        1 | []         | [3]         |
| IS1007c   | random s2                   |  50 |       8 |       0 |                    0 |        1 | []         | [3]         |
| TS3005d   | cosine (previous)           |  10 |       6 |       0 |                    0 |        2 | []         | [3, 9]      |
| TS3005d   | cosine (previous)           |  20 |       9 |       1 |                    0 |        0 | [5]        | []          |
| TS3005d   | cosine (previous)           |  30 |       8 |       1 |                    0 |        1 | [5]        | [3]         |
| TS3005d   | cosine (previous)           |  40 |       8 |       1 |                    0 |        1 | [5]        | [4]         |
| TS3005d   | cosine (previous)           |  50 |       7 |       1 |                    1 |        2 | [1]        | [3, 4]      |
| TS3005d   | sampled 8B (expected score) |  10 |       7 |       0 |                    0 |        1 | []         | [3]         |
| TS3005d   | sampled 8B (expected score) |  20 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | sampled 8B (expected score) |  30 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | sampled 8B (expected score) |  40 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | sampled 8B (expected score) |  50 |       7 |       0 |                    0 |        1 | []         | [4]         |
| TS3005d   | logp 8B                     |  10 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | logp 8B                     |  20 |       8 |       1 |                    1 |        1 | [1]        | [9]         |
| TS3005d   | logp 8B                     |  30 |       9 |       1 |                    1 |        0 | [1]        | []          |
| TS3005d   | logp 8B                     |  40 |       8 |       1 |                    1 |        1 | [1]        | [4]         |
| TS3005d   | logp 8B                     |  50 |       8 |       1 |                    1 |        1 | [1]        | [9]         |
| TS3005d   | logp 32B                    |  10 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | logp 32B                    |  20 |       7 |       0 |                    0 |        1 | []         | [3]         |
| TS3005d   | logp 32B                    |  30 |       8 |       1 |                    0 |        1 | [5]        | [4]         |
| TS3005d   | logp 32B                    |  40 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | logp 32B                    |  50 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s47                  |  10 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s47                  |  20 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s47                  |  30 |       7 |       0 |                    0 |        1 | []         | [6]         |
| TS3005d   | random s47                  |  40 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s47                  |  50 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s1                   |  10 |       6 |       0 |                    0 |        2 | []         | [3, 9]      |
| TS3005d   | random s1                   |  20 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s1                   |  30 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s1                   |  40 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s1                   |  50 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s2                   |  10 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s2                   |  20 |       8 |       0 |                    0 |        0 | []         | []          |
| TS3005d   | random s2                   |  30 |       6 |       0 |                    0 |        2 | []         | [4, 9]      |
| TS3005d   | random s2                   |  40 |       7 |       0 |                    0 |        1 | []         | [9]         |
| TS3005d   | random s2                   |  50 |       6 |       0 |                    0 |        2 | []         | [4, 9]      |

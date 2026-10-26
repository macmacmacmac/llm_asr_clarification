# Validation: average quiz accuracy (18 meetings, 180 questions)

Qwen3-32B-FP8 answers + grades greedily; mistranscription detector ALL, so lines are ranked by importance only.
ALL + ALL ranks all lines equally (random ties), i.e. a random-K baseline.
The first three rows are one fresh run (logp32b_val_comparison.sh). "(old)" rows and the ASR/GT transcript
rows were stored by earlier runs whose model/prompt is unknown, so treat them only as rough references.

```
                    10     20     30     40     50
ALL + COSINE     0.856  0.889  0.889  0.900  0.889
ALL + LOGP32B    0.889  0.906  0.917  0.900  0.911
ALL + ALL        0.850  0.850  0.861  0.844  0.844
COSINE (old)     0.772  0.822  0.844  0.878  0.839
ALL + ALL (old)  0.717  0.722  0.711  0.744  0.739
ASR transcript   0.850
GT transcript    0.939
```

Fixable questions (ASR wrong, GT right) fixed / questions ASR got right that broke (21 fixable):

```
fixed/broken        10     20     30     40     50
ALL + COSINE     10/10   14/7   12/5   13/5   14/8
ALL + LOGP32B     12/7   13/5   14/4   13/6   14/5
ALL + ALL          7/8   9/10    9/8   8/10   8/10
COSINE (old)     14/31  15/23  15/19  17/15  17/20
ALL + ALL (old)   9/36   8/35   9/38  10/32   9/33
```

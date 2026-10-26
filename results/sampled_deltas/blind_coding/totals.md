# Blind coding totals: pilot meetings ES2005d, IN1012, TS3005d

Each metric's top 20 per meeting was coded by a separate fresh agent that saw only the quiz and shuffled (question, ASR text, GT text) rows. 12 coders in total, 60 lines per metric.

## Totals over the 3 meetings

| metric                           |   lines |   mistranscribed |   important Yes |   important Partly |   mistranscribed & important Yes |   mistranscribed & important Yes/Partly |   top 10: mistranscribed & important Yes |
|:---------------------------------|--------:|-----------------:|----------------:|-------------------:|---------------------------------:|----------------------------------------:|-----------------------------------------:|
| Sampled 8B: expected total score |      60 |               49 |               2 |                  4 |                                2 |                                       6 |                                        1 |
| Sampled 8B: max over questions   |      60 |               40 |               4 |                  2 |                                3 |                                       5 |                                        3 |
| Logp 8B: max over questions      |      60 |               48 |              19 |                  9 |                               16 |                                      24 |                                       13 |
| Logp 32B: max over questions     |      60 |               51 |              28 |                 19 |                               23 |                                      39 |                                       20 |
| Cosine (GT importance detector)  |      60 |               40 |              33 |                 25 |                               22 |                                      40 |                                       15 |

## Mistranscribed & important (Yes), per meeting

| metric                           |   ES2005d |   IN1012 |   TS3005d |   total |
|:---------------------------------|----------:|---------:|----------:|--------:|
| Sampled 8B: expected total score |         0 |        2 |         0 |       2 |
| Sampled 8B: max over questions   |         1 |        2 |         0 |       3 |
| Logp 8B: max over questions      |         4 |        9 |         3 |      16 |
| Logp 32B: max over questions     |         7 |        8 |         8 |      23 |
| Cosine (GT importance detector)  |         4 |       11 |         7 |      22 |

## Mistranscribed & important (Yes or Partly), per meeting

| metric                           |   ES2005d |   IN1012 |   TS3005d |   total |
|:---------------------------------|----------:|---------:|----------:|--------:|
| Sampled 8B: expected total score |         2 |        2 |         2 |       6 |
| Sampled 8B: max over questions   |         3 |        2 |         0 |       5 |
| Logp 8B: max over questions      |         4 |       15 |         5 |      24 |
| Logp 32B: max over questions     |        12 |       16 |        11 |      39 |
| Cosine (GT importance detector)  |        10 |       18 |        12 |      40 |

## Distinct questions covered by mistranscribed & important (Yes) lines (out of 10 per meeting)

| metric                           |   ES2005d |   IN1012 |   TS3005d |   total |
|:---------------------------------|----------:|---------:|----------:|--------:|
| Sampled 8B: expected total score |         0 |        2 |         0 |       2 |
| Sampled 8B: max over questions   |         1 |        2 |         0 |       3 |
| Logp 8B: max over questions      |         3 |        5 |         3 |      11 |
| Logp 32B: max over questions     |         5 |        4 |         5 |      14 |
| Cosine (GT importance detector)  |         1 |        6 |         4 |      11 |

## Of those, distinct questions the metric's own baseline got wrong on the ASR transcript

Baseline: Qwen3-8B (sampled p < 0.5; 12 wrong questions) for S8B-*/LP8B; official Qwen3-32B quiz (6 wrong questions) for LP32B and COS.

| metric                           |   0 |
|:---------------------------------|----:|
| Sampled 8B: expected total score |   2 |
| Sampled 8B: max over questions   |   3 |
| Logp 8B: max over questions      |   5 |
| Logp 32B: max over questions     |   3 |
| Cosine (GT importance detector)  |   2 |

## Distinct questions covered by mistranscribed & important (Yes or Partly) lines (out of 10 per meeting)

| metric                           |   ES2005d |   IN1012 |   TS3005d |   total |
|:---------------------------------|----------:|---------:|----------:|--------:|
| Sampled 8B: expected total score |         2 |        2 |         1 |       5 |
| Sampled 8B: max over questions   |         3 |        2 |         0 |       5 |
| Logp 8B: max over questions      |         3 |        6 |         4 |      13 |
| Logp 32B: max over questions     |         7 |        6 |         7 |      20 |
| Cosine (GT importance detector)  |         2 |        6 |         5 |      13 |

## Coder agreement on (meeting, line, question) pairs coded more than once: 58 pairs

mistranscribed: 52/58; important: 55/58

## Slack table

```
           Mis   Imp   Mis+Imp    Qs   Qs′
S8B-exp     49     2         2     2     2
S8B-max     40     4         3     3     3
LP8B        48    19        16    11     5
LP32B       51    28        23    14     3
COS         40    33        22    11     2
```

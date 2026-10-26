# Blind coding results: TS3005d top-20 lines per teacher metric

Each metric's top 20 was coded by a separate fresh agent that saw only the quiz and shuffled (question, ASR text, GT text) rows, with no metric names, ranks, line numbers or scores. Counts are out of 20 lines per metric.

| metric                           |   mistranscribed |   important: Yes |   important: Partly |   mistranscribed AND important Yes |   mistranscribed AND important Yes/Partly |   in top 10: mistranscribed AND important Yes |
|:---------------------------------|-----------------:|-----------------:|--------------------:|-----------------------------------:|------------------------------------------:|----------------------------------------------:|
| Sampled 8B: expected total score |               16 |                0 |                   2 |                                  0 |                                         2 |                                             0 |
| Sampled 8B: max over questions   |               13 |                1 |                   0 |                                  0 |                                         0 |                                             0 |
| Logp 8B: max over questions      |               14 |                5 |                   2 |                                  3 |                                         5 |                                             3 |
| Logp 32B: max over questions     |               18 |                9 |                   4 |                                  8 |                                        11 |                                             7 |

## Agreement on (line, question) pairs coded by more than one coder (14 pairs)

mistranscribed agreement: 13/14; important agreement: 13/14

|   line | q   | coders   | mistranscribed   | important             | mis agree   | imp agree   |
|-------:|:----|:---------|:-----------------|:----------------------|:------------|:------------|
|      6 | q0  | AD       | No / No          | Yes / Yes             | True        | True        |
|     36 | q1  | ACD      | No / Yes / Yes   | Partly / Partly / Yes | False       | False       |
|    141 | q1  | AC       | Yes / Yes        | Partly / Partly       | True        | True        |
|    212 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |
|    313 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |
|    483 | q7  | AD       | Yes / Yes        | Partly / Partly       | True        | True        |
|    650 | q6  | AD       | Yes / Yes        | Yes / Yes             | True        | True        |
|    683 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |
|    729 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |
|    778 | q9  | BC       | No / No          | No / No               | True        | True        |
|   1035 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |
|   1067 | q8  | AD       | Yes / Yes        | Yes / Yes             | True        | True        |
|   1109 | q9  | BC       | No / No          | No / No               | True        | True        |
|   1236 | q9  | BC       | Yes / Yes        | No / No               | True        | True        |

## All codings (unblinded)

| metric                           | coder   | row   |   rank |   line | q   | mistranscribed   | important   | note                                                                    |
|:---------------------------------|:--------|:------|-------:|-------:|:----|:-----------------|:------------|:------------------------------------------------------------------------|
| Logp 32B: max over questions     | A       | A11   |      1 |    757 | q5  | Yes              | Yes         | ASR invents content and says 'use' where GT says 'scrap'                |
| Logp 32B: max over questions     | A       | A07   |      2 |     88 | q2  | Yes              | Yes         | 'date text' for teletext; loses 'should still be used'                  |
| Logp 32B: max over questions     | A       | A02   |      3 |      6 | q0  | No               | Yes         | Same meaning; states the folder was getting busy                        |
| Logp 32B: max over questions     | A       | A04   |      4 |    831 | q5  | Yes              | Yes         | 'top the set speaker' garbles 'drop the sample speaker'                 |
| Logp 32B: max over questions     | A       | A12   |      5 |    978 | q6  | Yes              | Yes         | '12-year-old, 15' for 'twelve Euro fifty'                               |
| Logp 32B: max over questions     | A       | A10   |      6 |    141 | q1  | Yes              | Partly      | ASR drops 'the evaluation criteria'                                     |
| Logp 32B: max over questions     | A       | A13   |      7 |   1147 | q8  | Yes              | Yes         | ASR drops the key phrase 'past each other'                              |
| Logp 32B: max over questions     | A       | A03   |      8 |    650 | q6  | Yes              | Yes         | Drops 'Euro'; 'ready for quick' garbles 'very quick'                    |
| Logp 32B: max over questions     | A       | A15   |      9 |   1076 | q9  | Yes              | Yes         | ASR drops 'together' and 'separate'                                     |
| Logp 32B: max over questions     | A       | A01   |     10 |     20 | q1  | Yes              | Partly      | ASR line is unrelated; GT mentions the evaluation still left            |
| Logp 32B: max over questions     | A       | A05   |     11 |   1067 | q8  | Yes              | Yes         | 'I also had some' reverses 'I didn't have'                              |
| Logp 32B: max over questions     | A       | A17   |     12 |     35 | q1  | Yes              | No          | Completely different utterance                                          |
| Logp 32B: max over questions     | A       | A08   |     13 |    483 | q7  | Yes              | Partly      | 'We're not going to' and 'salt' distort; GT gives decision, no reason   |
| Logp 32B: max over questions     | A       | A20   |     14 |     51 | q2  | Yes              | No          | 'This is' for 'View'                                                    |
| Logp 32B: max over questions     | A       | A16   |     15 |     36 | q1  | No               | Partly      | 'Eiffeluation' is still recognizable as 'evaluation'                    |
| Logp 32B: max over questions     | A       | A06   |     16 |    203 | q7  | Yes              | No          | ASR invents a long passage; GT is a vague fragment                      |
| Logp 32B: max over questions     | A       | A19   |     17 |     33 | q1  | Yes              | No          | Question instead of a statement; wrong words                            |
| Logp 32B: max over questions     | A       | A14   |     18 |     19 | q1  | Yes              | No          | ASR invents words where GT is only a backchannel                        |
| Logp 32B: max over questions     | A       | A18   |     19 |     23 | q1  | Yes              | No          | 'great' for 'main'                                                      |
| Logp 32B: max over questions     | A       | A09   |     20 |     17 | q1  | Yes              | No          | 'forever this' for 'wherever it is'; about a document, not the question |
| Logp 8B: max over questions      | D       | D20   |      1 |      6 | q0  | No               | Yes         | Same meaning; states folders separated because busy                     |
| Logp 8B: max over questions      | D       | D01   |      2 |    650 | q6  | Yes              | Yes         | Drops 'Euro'; 'ready for quick' garbles 'very very quick'               |
| Logp 8B: max over questions      | D       | D02   |      3 |   1067 | q8  | Yes              | Yes         | ASR loses 'information I didn't have'; GT supports criticism            |
| Logp 8B: max over questions      | D       | D04   |      4 |     36 | q1  | Yes              | Yes         | 'Eiffeluation' garbles evaluation; GT mentions evaluation criteria      |
| Logp 8B: max over questions      | D       | D05   |      5 |     28 | q4  | No               | No          | Only 'hopefully' dropped; unrelated to shape                            |
| Logp 8B: max over questions      | D       | D03   |      6 |    656 | q6  | Yes              | Partly      | Entirely different utterance; GT cost figure, not limit                 |
| Logp 8B: max over questions      | D       | D17   |      7 |    346 | q4  | Yes              | No          | 'speak your connection' vs speech recognition                           |
| Logp 8B: max over questions      | D       | D12   |      8 |    483 | q7  | Yes              | Partly      | Negation and 'salt' invented; GT states decision, not reason            |
| Logp 8B: max over questions      | D       | D15   |      9 |    927 | q2  | Yes              | No          | Backchannel misheard; no content                                        |
| Logp 8B: max over questions      | D       | D06   |     10 |     78 | q4  | Yes              | No          | Invented 'flat-side'; about scroll menu, not shape                      |
| Logp 8B: max over questions      | D       | D08   |     11 |    426 | q2  | No               | No          | Minor wording; unrelated to teletext                                    |
| Logp 8B: max over questions      | D       | D13   |     12 |    757 | q9  | Yes              | No          | Invented scroll content; 'use' vs 'scrap'; q5 not q9                    |
| Logp 8B: max over questions      | D       | D16   |     13 |     67 | q4  | Yes              | No          | ASR invents extra content; not about shape                              |
| Logp 8B: max over questions      | D       | D11   |     14 |     62 | q4  | Yes              | No          | 'Just' vs 'Yeah'; trivial backchannel                                   |
| Logp 8B: max over questions      | D       | D19   |     15 |    868 | q3  | No               | No          | Fragment, same meaning; no content                                      |
| Logp 8B: max over questions      | D       | D14   |     16 |   1288 | q1  | Yes              | No          | Completely different utterance                                          |
| Logp 8B: max over questions      | D       | D10   |     17 |     61 | q0  | Yes              | No          | 'growth' invented; unrelated to folders                                 |
| Logp 8B: max over questions      | D       | D18   |     18 |     53 | q4  | Yes              | No          | 'photo' vs 'prototype'; about colour                                    |
| Logp 8B: max over questions      | D       | D09   |     19 |    331 | q3  | No               | Yes         | Same meaning; options not always on screen                              |
| Logp 8B: max over questions      | D       | D07   |     20 |      0 | q4  | No               | No          | Identical backchannel                                                   |
| Sampled 8B: expected total score | C       | C16   |      1 |    154 | q9  | Yes              | No          | Invented thanks for 'Yeah'                                              |
| Sampled 8B: expected total score | C       | C17   |      2 |    683 | q9  | Yes              | No          | 'got to' obligation vs 'got Excel'; minor                               |
| Sampled 8B: expected total score | C       | C18   |      3 |      9 | q1  | Yes              | No          | Different utterance                                                     |
| Sampled 8B: expected total score | C       | C11   |      4 |   1236 | q9  | Yes              | No          | Drops 'worked'                                                          |
| Sampled 8B: expected total score | C       | C09   |      5 |     36 | q1  | Yes              | Partly      | 'Eiffeluation' garbles key term; asks about evaluation criteria         |
| Sampled 8B: expected total score | C       | C06   |      6 |    931 | q9  | No               | No          | Only tag 'huh' missing                                                  |
| Sampled 8B: expected total score | C       | C19   |      7 |    469 | q9  | No               | No          | Identical                                                               |
| Sampled 8B: expected total score | C       | C15   |      8 |    212 | q9  | Yes              | No          | Loses elder people won't like it                                        |
| Sampled 8B: expected total score | C       | C05   |      9 |    778 | q9  | No               | No          | Same meaning, filler only                                               |
| Sampled 8B: expected total score | C       | C13   |     10 |    325 | q1  | Yes              | No          | Invented 'L.O.L.' for 'Yeah'                                            |
| Sampled 8B: expected total score | C       | C01   |     11 |    729 | q9  | Yes              | No          | Different word entirely; unrelated to process improvement               |
| Sampled 8B: expected total score | C       | C04   |     12 |    313 | q9  | Yes              | No          | Invented sentence for 'Yep'                                             |
| Sampled 8B: expected total score | C       | C02   |     13 |   1121 | q1  | Yes              | No          | ASR invents pen/computer content; unrelated to q1                       |
| Sampled 8B: expected total score | C       | C20   |     14 |    141 | q1  | Yes              | Partly      | ASR drops 'evaluation criteria' mention                                 |
| Sampled 8B: expected total score | C       | C08   |     15 |    569 | q9  | Yes              | No          | ASR adds uncertainty 'I don't know if'                                  |
| Sampled 8B: expected total score | C       | C14   |     16 |   1109 | q9  | No               | No          | Both simple acknowledgments                                             |
| Sampled 8B: expected total score | C       | C10   |     17 |   1253 | q1  | Yes              | No          | Invented thanks for 'Huh'                                               |
| Sampled 8B: expected total score | C       | C03   |     18 |     58 | q1  | Yes              | No          | 'question is so far' loses 'any questions'; logo talk                   |
| Sampled 8B: expected total score | C       | C07   |     19 |     51 | q1  | Yes              | No          | Different words                                                         |
| Sampled 8B: expected total score | C       | C12   |     20 |   1035 | q9  | Yes              | No          | 'race' vs 'erase'; question vs statement                                |
| Sampled 8B: max over questions   | B       | B04   |      1 |    295 | q5  | Yes              | No          | 'story' vs 'three'; unrelated to sample speaker                         |
| Sampled 8B: max over questions   | B       | B07   |      2 |    120 | q3  | No               | Yes         | Gives the reason: on top it is always visible                           |
| Sampled 8B: max over questions   | B       | B06   |      3 |     58 | q3  | Yes              | No          | Logo wording and 'any questions' garbled                                |
| Sampled 8B: max over questions   | B       | B02   |      4 |   1014 | q3  | No               | No          | Identical backchannel                                                   |
| Sampled 8B: max over questions   | B       | B18   |      5 |     38 | q3  | No               | No          | Yeah vs Yep, same meaning                                               |
| Sampled 8B: max over questions   | B       | B15   |      6 |   1126 | q3  | Yes              | No          | Meaning about ease of designing lost                                    |
| Sampled 8B: max over questions   | B       | B17   |      7 |    490 | q3  | No               | No          | Identical                                                               |
| Sampled 8B: max over questions   | B       | B12   |      8 |    842 | q3  | Yes              | No          | Hallucinated repetition vs 'Point twelve'                               |
| Sampled 8B: max over questions   | B       | B11   |      9 |      6 | q3  | No               | No          | About folders (q0), not menu placement                                  |
| Sampled 8B: max over questions   | B       | B20   |     10 |    683 | q9  | Yes              | No          | 'excel to count' vs 'Excel to count'                                    |
| Sampled 8B: max over questions   | B       | B08   |     11 |   1236 | q9  | Yes              | No          | Drops 'worked'                                                          |
| Sampled 8B: max over questions   | B       | B19   |     12 |    212 | q9  | Yes              | No          | Loses 'elder people won't like it'; relates q7, not q9                  |
| Sampled 8B: max over questions   | B       | B13   |     13 |    313 | q9  | Yes              | No          | Invented sentence vs 'Yep'                                              |
| Sampled 8B: max over questions   | B       | B05   |     14 |    778 | q9  | No               | No          | Only filler and article differ                                          |
| Sampled 8B: max over questions   | B       | B09   |     15 |    729 | q9  | Yes              | No          | 'Dorothy' vs 'Thirteen'                                                 |
| Sampled 8B: max over questions   | B       | B10   |     16 |    339 | q9  | Yes              | No          | 'menu' vs 'manual'                                                      |
| Sampled 8B: max over questions   | B       | B03   |     17 |   1035 | q9  | Yes              | No          | 'can we race' vs 'you can erase'                                        |
| Sampled 8B: max over questions   | B       | B01   |     18 |   1109 | q9  | No               | No          | Both are backchannel agreement; extra 'all right' adds nothing          |
| Sampled 8B: max over questions   | B       | B16   |     19 |    815 | q9  | Yes              | No          | Hallucinated content vs filler 'Uh'                                     |
| Sampled 8B: max over questions   | B       | B14   |     20 |    130 | q9  | Yes              | No          | 'So' vs 'Looks good'                                                    |

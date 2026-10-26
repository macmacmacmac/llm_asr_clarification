# Blind coding results: ES2005d top-20 lines per teacher metric

Each metric's top 20 was coded by a separate fresh agent that saw only the quiz and shuffled (question, ASR text, GT text) rows, with no metric names, ranks, line numbers or scores. Counts are out of 20 lines per metric.

| metric                           |   mistranscribed |   important: Yes |   important: Partly |   mistranscribed AND important Yes |   mistranscribed AND important Yes/Partly |   in top 10: mistranscribed AND important Yes |
|:---------------------------------|-----------------:|-----------------:|--------------------:|-----------------------------------:|------------------------------------------:|----------------------------------------------:|
| Sampled 8B: expected total score |               14 |                0 |                   2 |                                  0 |                                         2 |                                             0 |
| Sampled 8B: max over questions   |               13 |                1 |                   2 |                                  1 |                                         3 |                                             1 |
| Logp 8B: max over questions      |               15 |                5 |                   0 |                                  4 |                                         4 |                                             4 |
| Logp 32B: max over questions     |               15 |               10 |                   6 |                                  7 |                                        12 |                                             5 |

## Agreement on (line, question) pairs coded by more than one coder (12 pairs)

mistranscribed agreement: 11/12; important agreement: 12/12

|   line | q   | coders   | mistranscribed   | important                | mis agree   | imp agree   |
|-------:|:----|:---------|:-----------------|:-------------------------|:------------|:------------|
|     28 | q3  | AB       | Yes / Yes        | No / No                  | True        | True        |
|     46 | q3  | ABD      | Yes / Yes / Yes  | Partly / Partly / Partly | True        | True        |
|     56 | q2  | CD       | Yes / Yes        | Yes / Yes                | True        | True        |
|     73 | q4  | AC       | Yes / No         | No / No                  | False       | True        |
|     74 | q4  | AC       | Yes / Yes        | No / No                  | True        | True        |
|     75 | q4  | AB       | Yes / Yes        | Partly / Partly          | True        | True        |
|    106 | q6  | AD       | Yes / Yes        | Yes / Yes                | True        | True        |
|    109 | q6  | CD       | Yes / Yes        | Yes / Yes                | True        | True        |
|    144 | q0  | CD       | No / No          | Yes / Yes                | True        | True        |
|    231 | q1  | CD       | Yes / Yes        | Yes / Yes                | True        | True        |
|    232 | q1  | CD       | Yes / Yes        | Yes / Yes                | True        | True        |
|    412 | q4  | AB       | No / No          | No / No                  | True        | True        |

## All codings (unblinded)

| metric                           | coder   | row   |   rank |   line | q   | mistranscribed   | important   | note                                                                        |
|:---------------------------------|:--------|:------|-------:|-------:|:----|:-----------------|:------------|:----------------------------------------------------------------------------|
| Logp 32B: max over questions     | D       | D19   |      1 |     85 | q5  | Yes              | Yes         | 'bothered' vs 'bashed'                                                      |
| Logp 32B: max over questions     | D       | D01   |      2 |    109 | q6  | Yes              | Yes         | ASR garbled; GT gives 'out of the way' reason                               |
| Logp 32B: max over questions     | D       | D18   |      3 |    144 | q0  | No               | Yes         | Save two Euros by flattening, preserved                                     |
| Logp 32B: max over questions     | D       | D03   |      4 |     46 | q3  | Yes              | Partly      | Different utterance; GT shows tomato wasn't discussed                       |
| Logp 32B: max over questions     | D       | D12   |      5 |    106 | q6  | Yes              | Yes         | 'uphill' vs 'obvious' place                                                 |
| Logp 32B: max over questions     | D       | D13   |      6 |     47 | q3  | Yes              | Partly      | GT says tomato not recalled; ASR differs                                    |
| Logp 32B: max over questions     | D       | D04   |      7 |    232 | q1  | Yes              | Yes         | 'put 20 money' loses 'no point spending money'                              |
| Logp 32B: max over questions     | D       | D16   |      8 |    231 | q1  | Yes              | Yes         | ASR loses 'trying to save money'                                            |
| Logp 32B: max over questions     | D       | D17   |      9 |     45 | q3  | Yes              | Partly      | 'the other one' vs 'Tomato?'                                                |
| Logp 32B: max over questions     | D       | D14   |     10 |     42 | q3  | Yes              | No          | Different utterance; unrelated                                              |
| Logp 32B: max over questions     | D       | D02   |     11 |     48 | q3  | No               | Yes         | Red material left over, matches gold answer                                 |
| Logp 32B: max over questions     | D       | D15   |     12 |     53 | q4  | No               | Yes         | Yellow is the slogan, preserved                                             |
| Logp 32B: max over questions     | D       | D11   |     13 |     56 | q2  | Yes              | Yes         | 'speak tracking notion' vs speech recognition                               |
| Logp 32B: max over questions     | D       | D07   |     14 |    134 | q0  | No               | Partly      | Cost context, not the flattening reason                                     |
| Logp 32B: max over questions     | D       | D08   |     15 |    238 | q0  | Yes              | Partly      | '12 years' vs 'twelve Euros'; budget context                                |
| Logp 32B: max over questions     | D       | D05   |     16 |      3 | q3  | No               | No          | Only a false start differs; unrelated to tomato                             |
| Logp 32B: max over questions     | D       | D06   |     17 |    468 | q8  | Yes              | Yes         | ASR says '2.10' instead of final score two                                  |
| Logp 32B: max over questions     | D       | D10   |     18 |    342 | q3  | Yes              | Partly      | 'tumulties' vs tomato; mentions tomato only                                 |
| Logp 32B: max over questions     | D       | D09   |     19 |    424 | q7  | Yes              | No          | ASR invented fruit content; GT is filler                                    |
| Logp 32B: max over questions     | D       | D20   |     20 |    129 | q1  | Yes              | No          | ASR invents content; GT generic issues prompt                               |
| Logp 8B: max over questions      | C       | C05   |      1 |    144 | q0  | No               | Yes         | Minor wording change; GT states flattening saves two Euros                  |
| Logp 8B: max over questions      | C       | C14   |      2 |    231 | q1  | Yes              | Yes         | ASR loses 'we're trying to save money'                                      |
| Logp 8B: max over questions      | C       | C16   |      3 |    232 | q1  | Yes              | Yes         | 'no point in spending money' garbled into 'put 20 money'                    |
| Logp 8B: max over questions      | C       | C19   |      4 |     56 | q2  | Yes              | Yes         | 'speech recognition' became 'speak tracking notion'                         |
| Logp 8B: max over questions      | C       | C20   |      5 |    109 | q6  | Yes              | Yes         | 'out of the way' lost; that is part of the gold answer                      |
| Logp 8B: max over questions      | C       | C04   |      6 |     13 | q6  | No               | No          | Only fillers differ; no content                                             |
| Logp 8B: max over questions      | C       | C10   |      7 |     73 | q4  | No               | No          | Same words; does not explain yellow element                                 |
| Logp 8B: max over questions      | C       | C03   |      8 |    305 | q7  | Yes              | No          | ASR invents content; GT just asks if Brian finished                         |
| Logp 8B: max over questions      | C       | C06   |      9 |     74 | q4  | Yes              | No          | Different backchannel; unrelated to yellow slogan                           |
| Logp 8B: max over questions      | C       | C18   |     10 |    538 | q9  | Yes              | No          | Numbers garbled; scores, not the launch conclusion                          |
| Logp 8B: max over questions      | C       | C08   |     11 |    419 | q2  | Yes              | No          | Invented 'requirements'; GT unrelated to speech recognition                 |
| Logp 8B: max over questions      | C       | C07   |     12 |     12 | q9  | Yes              | No          | Hallucinated phrase; GT is just 'Hmm'                                       |
| Logp 8B: max over questions      | C       | C11   |     13 |    111 | q9  | Yes              | No          | 'banana-based one' lost; not about launch conclusion                        |
| Logp 8B: max over questions      | C       | C15   |     14 |    556 | q7  | No               | No          | Yep vs yeah; backchannel                                                    |
| Logp 8B: max over questions      | C       | C17   |     15 |    233 | q7  | Yes              | No          | ASR repeats earlier content; GT is just 'So yeah'                           |
| Logp 8B: max over questions      | C       | C02   |     16 |    241 | q6  | Yes              | No          | Different utterance; backchannel unrelated to question                      |
| Logp 8B: max over questions      | C       | C09   |     17 |    248 | q7  | Yes              | No          | Question turned into statement; creativity criterion, not fruit-shape sales |
| Logp 8B: max over questions      | C       | C13   |     18 |    266 | q8  | No               | No          | Repetition only; bare agreement                                             |
| Logp 8B: max over questions      | C       | C12   |     19 |    229 | q2  | Yes              | No          | Question garbled; finances discussion, not interaction method               |
| Logp 8B: max over questions      | C       | C01   |     20 |    437 | q6  | Yes              | No          | 'chips' replaces 'shape'; about fruit shapes, not button placement          |
| Sampled 8B: expected total score | B       | B13   |      1 |     46 | q3  | Yes              | Partly      | GT 'What tomato?' shows tomato unexpected                                   |
| Sampled 8B: expected total score | B       | B16   |      2 |     75 | q4  | Yes              | Partly      | sand vs stand; yellow element is charging stand                             |
| Sampled 8B: expected total score | B       | B10   |      3 |    240 | q2  | Yes              | No          | Statement vs question, subject changed                                      |
| Sampled 8B: expected total score | B       | B17   |      4 |    412 | q4  | No               | No          | Punctuation only; evaluation instructions                                   |
| Sampled 8B: expected total score | B       | B19   |      5 |    207 | q2  | Yes              | No          | books vs box; unrelated to speech                                           |
| Sampled 8B: expected total score | B       | B20   |      6 |    289 | q2  | No               | No          | Extra fillers only                                                          |
| Sampled 8B: expected total score | B       | B07   |      7 |    369 | q2  | Yes              | No          | Completely different utterance; unrelated                                   |
| Sampled 8B: expected total score | B       | B02   |      8 |     88 | q2  | Yes              | No          | 'some hands' vs 'small hands'; clause dropped                               |
| Sampled 8B: expected total score | B       | B05   |      9 |    268 | q2  | No               | No          | info vs input, essentially same meaning                                     |
| Sampled 8B: expected total score | B       | B04   |     10 |    387 | q2  | Yes              | No          | ASR garbled; GT about operating TV, not speech                              |
| Sampled 8B: expected total score | B       | B06   |     11 |     24 | q2  | No               | No          | Interjection only                                                           |
| Sampled 8B: expected total score | B       | B14   |     12 |     28 | q3  | Yes              | No          | Different words; unrelated to tomato reason                                 |
| Sampled 8B: expected total score | B       | B08   |     13 |    281 | q2  | Yes              | No          | Backchannel turned into invented words                                      |
| Sampled 8B: expected total score | B       | B12   |     14 |    475 | q1  | Yes              | No          | Garbled; GT asks what to write                                              |
| Sampled 8B: expected total score | B       | B03   |     15 |    249 | q2  | No               | No          | Only tense differs; generic praise                                          |
| Sampled 8B: expected total score | B       | B15   |     16 |    150 | q2  | Yes              | No          | Backchannel turned into invented words                                      |
| Sampled 8B: expected total score | B       | B01   |     17 |     15 | q2  | Yes              | No          | 'Not right' reverses 'Alright'; backchannel                                 |
| Sampled 8B: expected total score | B       | B11   |     18 |    318 | q2  | No               | No          | Apology either way                                                          |
| Sampled 8B: expected total score | B       | B18   |     19 |    511 | q2  | Yes              | No          | Different utterance; yellow not about speech                                |
| Sampled 8B: expected total score | B       | B09   |     20 |    465 | q1  | Yes              | No          | '1.0.2' vs 'one or two'; unrelated to q1                                    |
| Sampled 8B: max over questions   | A       | A13   |      1 |    106 | q6  | Yes              | Yes         | 'obvious' became 'uphill'; GT gives the gold reason                         |
| Sampled 8B: max over questions   | A       | A08   |      2 |     73 | q4  | Yes              | No          | 'stand' became name 'Stan'                                                  |
| Sampled 8B: max over questions   | A       | A02   |      3 |     74 | q4  | Yes              | No          | Different utterance; backchannel unrelated to yellow element                |
| Sampled 8B: max over questions   | A       | A12   |      4 |     75 | q4  | Yes              | Partly      | 'stand' became 'sand'; yellow element mentioned, not slogan                 |
| Sampled 8B: max over questions   | A       | A04   |      5 |    279 | q4  | Yes              | No          | 'couldn't reach' became 'could have reached'; meaning flipped               |
| Sampled 8B: max over questions   | A       | A01   |      6 |     61 | q4  | No               | No          | Same backchannel, filler difference only                                    |
| Sampled 8B: max over questions   | A       | A15   |      7 |     68 | q4  | No               | No          | Repetition dropped; same meaning                                            |
| Sampled 8B: max over questions   | A       | A17   |      8 |     46 | q3  | Yes              | Partly      | 'What tomato?' lost; GT raises tomato, no reason                            |
| Sampled 8B: max over questions   | A       | A05   |      9 |    412 | q4  | No               | No          | Punctuation/apostrophe only                                                 |
| Sampled 8B: max over questions   | A       | A09   |     10 |     15 | q4  | Yes              | No          | 'Alright' became 'Not right', negates meaning                               |
| Sampled 8B: max over questions   | A       | A18   |     11 |    332 | q4  | Yes              | No          | 'forgot' and 'latex' garbled                                                |
| Sampled 8B: max over questions   | A       | A07   |     12 |    392 | q4  | No               | No          | Number formatting only                                                      |
| Sampled 8B: max over questions   | A       | A20   |     13 |    188 | q4  | Yes              | No          | 'maybe...stuck in' became 'you're...stick'                                  |
| Sampled 8B: max over questions   | A       | A06   |     14 |    171 | q4  | Yes              | No          | 'shape' became 'sheep'; garbled first clause                                |
| Sampled 8B: max over questions   | A       | A14   |     15 |     37 | q4  | No               | No          | Backchannel swap, same meaning                                              |
| Sampled 8B: max over questions   | A       | A16   |     16 |    422 | q3  | Yes              | No          | Invented 'yellow orange' ending; minor. Relevant to q7                      |
| Sampled 8B: max over questions   | A       | A03   |     17 |    294 | q4  | No               | No          | Filler/false start only                                                     |
| Sampled 8B: max over questions   | A       | A10   |     18 |     58 | q4  | No               | No          | Minor filler difference only                                                |
| Sampled 8B: max over questions   | A       | A11   |     19 |     28 | q3  | Yes              | No          | Different words; unrelated to tomato                                        |
| Sampled 8B: max over questions   | A       | A19   |     20 |    431 | q3  | Yes              | No          | 'rather' became 'lab and fruit is'; q7-related                              |

# Blind coding results: IN1012 top-20 lines per teacher metric

Each metric's top 20 was coded by a separate fresh agent that saw only the quiz and shuffled (question, ASR text, GT text) rows, with no metric names, ranks, line numbers or scores. Counts are out of 20 lines per metric.

| metric                           |   mistranscribed |   important: Yes |   important: Partly |   mistranscribed AND important Yes |   mistranscribed AND important Yes/Partly |   in top 10: mistranscribed AND important Yes |
|:---------------------------------|-----------------:|-----------------:|--------------------:|-----------------------------------:|------------------------------------------:|----------------------------------------------:|
| Sampled 8B: expected total score |               19 |                2 |                   0 |                                  2 |                                         2 |                                             1 |
| Sampled 8B: max over questions   |               14 |                2 |                   0 |                                  2 |                                         2 |                                             2 |
| Logp 8B: max over questions      |               19 |                9 |                   7 |                                  9 |                                        15 |                                             6 |
| Logp 32B: max over questions     |               18 |                9 |                   9 |                                  8 |                                        16 |                                             8 |

## Agreement on (line, question) pairs coded by more than one coder (22 pairs)

mistranscribed agreement: 19/22; important agreement: 20/22

|   line | q   | coders   | mistranscribed        | important             | mis agree   | imp agree   |
|-------:|:----|:---------|:----------------------|:----------------------|:------------|:------------|
|      2 | q0  | BD       | Yes / Yes             | No / No               | True        | True        |
|     36 | q0  | ABC      | Yes / Yes / Yes       | Yes / Yes / Yes       | True        | True        |
|     44 | q1  | AD       | No / Yes              | Yes / Yes             | False       | True        |
|     53 | q2  | AD       | Yes / Yes             | No / No               | True        | True        |
|     57 | q2  | AD       | Yes / Yes             | Yes / Yes             | True        | True        |
|     68 | q2  | AD       | Yes / Yes             | Partly / Yes          | True        | False       |
|    207 | q1  | BC       | Yes / Yes             | No / No               | True        | True        |
|    235 | q4  | AD       | Yes / Yes             | Partly / Yes          | True        | False       |
|    289 | q6  | AD       | Yes / Yes             | Partly / Partly       | True        | True        |
|    332 | q7  | AD       | Yes / Yes             | Yes / Yes             | True        | True        |
|    354 | q1  | BC       | Yes / Yes             | No / No               | True        | True        |
|    358 | q1  | BC       | No / Yes              | No / No               | False       | True        |
|    462 | q1  | BC       | No / Yes              | No / No               | False       | True        |
|    775 | q6  | BC       | Yes / Yes             | No / No               | True        | True        |
|    917 | q9  | ABCD     | Yes / Yes / Yes / Yes | Yes / Yes / Yes / Yes | True        | True        |
|    918 | q9  | AD       | Yes / Yes             | Yes / Yes             | True        | True        |
|    926 | q9  | AD       | Yes / Yes             | Partly / Partly       | True        | True        |
|    927 | q9  | AD       | Yes / Yes             | Partly / Partly       | True        | True        |
|    928 | q9  | AD       | Yes / Yes             | Yes / Yes             | True        | True        |
|    930 | q9  | AD       | Yes / Yes             | Partly / Partly       | True        | True        |
|    938 | q9  | AD       | Yes / Yes             | Partly / Partly       | True        | True        |
|    942 | q9  | AD       | No / No               | Partly / Partly       | True        | True        |

## All codings (unblinded)

| metric                           | coder   | row   |   rank |   line | q   | mistranscribed   | important   | note                                                                  |
|:---------------------------------|:--------|:------|-------:|-------:|:----|:-----------------|:------------|:----------------------------------------------------------------------|
| Logp 32B: max over questions     | A       | A09   |      1 |    332 | q7  | Yes              | Yes         | ASR drops 'Nyquist', which is key to the answer                       |
| Logp 32B: max over questions     | A       | A06   |      2 |    928 | q9  | Yes              | Yes         | Lost 'Linux machines' and 'diphone based'                             |
| Logp 32B: max over questions     | A       | A02   |      3 |    917 | q9  | Yes              | Yes         | 'iPhone' replaces 'diphones'; GT states diphone basis                 |
| Logp 32B: max over questions     | A       | A20   |      4 |     15 | q0  | Yes              | Yes         | 'multi-processing' for 'multiple session at the same time'            |
| Logp 32B: max over questions     | A       | A05   |      5 |    918 | q9  | Yes              | Yes         | 'for your phones' replaces 'Diphone'                                  |
| Logp 32B: max over questions     | A       | A18   |      6 |    340 | q7  | Yes              | Yes         | Garbled 'Nyquist frequency'; GT has the core constraint               |
| Logp 32B: max over questions     | A       | A19   |      7 |     36 | q0  | Yes              | Yes         | ASR misses the whole line; GT names multiple sessions                 |
| Logp 32B: max over questions     | A       | A13   |      8 |    927 | q9  | Yes              | Partly      | ASR loses 'diphone'; GT asks which approach is running                |
| Logp 32B: max over questions     | A       | A17   |      9 |    235 | q4  | Yes              | Partly      | 'T-FTC' for 'if you see'; 'mixtures' lost                             |
| Logp 32B: max over questions     | A       | A10   |     10 |     57 | q2  | Yes              | Yes         | 'year' for 'ear'; GT names the invited talk                           |
| Logp 32B: max over questions     | A       | A03   |     11 |     68 | q2  | Yes              | Partly      | 'year' for 'ear'; GT identifies ear-implant inventor                  |
| Logp 32B: max over questions     | A       | A15   |     12 |    926 | q9  | Yes              | Partly      | ASR drops 'diphone' and changes the ending                            |
| Logp 32B: max over questions     | A       | A01   |     13 |    930 | q9  | Yes              | Partly      | ASR entirely different; GT mentions diphone samples                   |
| Logp 32B: max over questions     | A       | A07   |     14 |     44 | q1  | No               | Yes         | 'say' vs 'see' minor; GT states the gold answer                       |
| Logp 32B: max over questions     | A       | A14   |     15 |    942 | q9  | No               | Partly      | 'die phones' is a phonetic misspelling; GT contrasts with non-diphone |
| Logp 32B: max over questions     | A       | A04   |     16 |    289 | q6  | Yes              | Partly      | Garbled 'assume'; GT gives 12.5 ms, not gold sizes                    |
| Logp 32B: max over questions     | A       | A16   |     17 |    293 | q6  | Yes              | Partly      | Different words; GT gives a window size, not the gold one             |
| Logp 32B: max over questions     | A       | A08   |     18 |     53 | q2  | Yes              | No          | ASR is a different utterance; GT about panel discussions              |
| Logp 32B: max over questions     | A       | A11   |     19 |    938 | q9  | Yes              | Partly      | 'iPhone' for diphone, invented content; GT mentions diphone synthesis |
| Logp 32B: max over questions     | A       | A12   |     20 |    887 | q0  | Yes              | No          | 'happy' vs 'annoying'; not about conference sessions                  |
| Logp 8B: max over questions      | D       | D08   |      1 |    928 | q9  | Yes              | Yes         | Garbled; GT says Festival on Linux is diphone-based                   |
| Logp 8B: max over questions      | D       | D20   |      2 |    332 | q7  | Yes              | Yes         | Dropped 'Nyquist'; GT states gold answer                              |
| Logp 8B: max over questions      | D       | D17   |      3 |     57 | q2  | Yes              | Yes         | 'year' vs 'ear'; GT names invited speaker implant talk                |
| Logp 8B: max over questions      | D       | D01   |      4 |     53 | q2  | Yes              | No          | Entirely different utterance; GT is about panel discussions           |
| Logp 8B: max over questions      | D       | D14   |      5 |     44 | q1  | Yes              | Yes         | 'say' vs 'see' posters; GT states gold answer                         |
| Logp 8B: max over questions      | D       | D12   |      6 |    927 | q9  | Yes              | Partly      | Lost 'diphone'; GT about which synthesis approach runs                |
| Logp 8B: max over questions      | D       | D13   |      7 |    235 | q4  | Yes              | Yes         | Garbled 'T-FTC', lost 'mixtures'; GT: many models, mixtures           |
| Logp 8B: max over questions      | D       | D09   |      8 |    918 | q9  | Yes              | Yes         | 'your phones' vs 'Diphone'; confirms diphone                          |
| Logp 8B: max over questions      | D       | D16   |      9 |    289 | q6  | Yes              | Partly      | Garbled 'definitely assume'; GT gives 12.5 ms minimum                 |
| Logp 8B: max over questions      | D       | D15   |     10 |    926 | q9  | Yes              | Partly      | Dropped 'diphone', added content; Festival voices context             |
| Logp 8B: max over questions      | D       | D05   |     11 |    930 | q9  | Yes              | Partly      | Different content; GT mentions diphone samples                        |
| Logp 8B: max over questions      | D       | D07   |     12 |    917 | q9  | Yes              | Yes         | 'iPhone' vs 'diphones'; GT states diphone-based                       |
| Logp 8B: max over questions      | D       | D02   |     13 |      4 | q0  | Yes              | No          | 'stuck' vs 'start'; unrelated to sessions                             |
| Logp 8B: max over questions      | D       | D19   |     14 |      2 | q0  | Yes              | No          | Different content; unrelated to simultaneous sessions                 |
| Logp 8B: max over questions      | D       | D18   |     15 |    942 | q9  | No               | Partly      | 'die phones' homophone of diphones; contrasts non-diphone             |
| Logp 8B: max over questions      | D       | D06   |     16 |     68 | q2  | Yes              | Yes         | 'year' vs 'ear'; GT names ear implant inventor                        |
| Logp 8B: max over questions      | D       | D10   |     17 |    924 | q9  | Yes              | Partly      | Totally different; GT: Festival on Linux, voices                      |
| Logp 8B: max over questions      | D       | D03   |     18 |    590 | q6  | Yes              | No          | Different content; about NIST results, not window sizes               |
| Logp 8B: max over questions      | D       | D11   |     19 |    938 | q9  | Yes              | Partly      | 'iPhone' for diphone, 'dual' for default; voices context              |
| Logp 8B: max over questions      | D       | D04   |     20 |    330 | q7  | Yes              | Yes         | 'safety/nitric' vs 'shift/Nyquist'; GT links shift to Nyquist         |
| Sampled 8B: expected total score | C       | C10   |      1 |     36 | q0  | Yes              | Yes         | GT states multiple sessions annoying; ASR hallucinated thanks         |
| Sampled 8B: expected total score | C       | C03   |      2 |    207 | q1  | Yes              | No          | Diagonal covariance/decorrelation lost; unrelated to posters          |
| Sampled 8B: expected total score | C       | C04   |      3 |   1014 | q1  | Yes              | No          | CSLU garbled, invented content; not about posters                     |
| Sampled 8B: expected total score | C       | C13   |      4 |    949 | q1  | Yes              | No          | Most content lost; unrelated                                          |
| Sampled 8B: expected total score | C       | C17   |      5 |    720 | q1  | Yes              | No          | MFCCs lost, invented content; unrelated                               |
| Sampled 8B: expected total score | C       | C01   |      6 |    354 | q1  | Yes              | No          | 'loot ratio' vs maximum likelihood; unrelated to posters              |
| Sampled 8B: expected total score | C       | C08   |      7 |    936 | q1  | Yes              | No          | Much invented content; GT about Festival voices                       |
| Sampled 8B: expected total score | C       | C19   |      8 |    462 | q1  | Yes              | No          | 'And' vs backchannel; trivial                                         |
| Sampled 8B: expected total score | C       | C11   |      9 |    775 | q6  | Yes              | No          | Completely different utterance; Nokia/PLP unrelated to windows        |
| Sampled 8B: expected total score | C       | C15   |     10 |    770 | q1  | Yes              | No          | 'Daimler' lost; unrelated                                             |
| Sampled 8B: expected total score | C       | C06   |     11 |    520 | q1  | Yes              | No          | 'hotel', 'room in group' vs reading group                             |
| Sampled 8B: expected total score | C       | C07   |     12 |    541 | q1  | Yes              | No          | Invented words for a filler                                           |
| Sampled 8B: expected total score | C       | C09   |     13 |    358 | q1  | Yes              | No          | 'you' hallucinated for backchannel                                    |
| Sampled 8B: expected total score | C       | C12   |     14 |    780 | q1  | Yes              | No          | Invented content after 'Cohen'; unrelated                             |
| Sampled 8B: expected total score | C       | C05   |     15 |    573 | q1  | No               | No          | Identical backchannel                                                 |
| Sampled 8B: expected total score | C       | C02   |     16 |    917 | q9  | Yes              | Yes         | 'iPhone' vs diphones; GT gives Festival answer                        |
| Sampled 8B: expected total score | C       | C14   |     17 |    591 | q1  | Yes              | No          | Large invented continuation; unrelated                                |
| Sampled 8B: expected total score | C       | C20   |     18 |   1028 | q1  | Yes              | No          | Completely different utterance; Simon King lost                       |
| Sampled 8B: expected total score | C       | C18   |     19 |    837 | q1  | Yes              | No          | Different wording; Viterbi garbled; unrelated                         |
| Sampled 8B: expected total score | C       | C16   |     20 |    521 | q1  | Yes              | No          | Invented sentence for 'Mm'                                            |
| Sampled 8B: max over questions   | B       | B11   |      1 |    917 | q9  | Yes              | Yes         | 'iPhone' vs 'diphones'; GT states answer                              |
| Sampled 8B: max over questions   | B       | B20   |      2 |     36 | q0  | Yes              | Yes         | ASR loses 'annoying thing is multiple sessions'                       |
| Sampled 8B: max over questions   | B       | B12   |      3 |    775 | q6  | Yes              | No          | Different utterance; about Nokia tandem PLPs                          |
| Sampled 8B: max over questions   | B       | B17   |      4 |    531 | q6  | Yes              | No          | ASR adds speech-coding content; GT about optimisation criteria        |
| Sampled 8B: max over questions   | B       | B05   |      5 |    757 | q6  | No               | No          | Backchannel equivalents                                               |
| Sampled 8B: max over questions   | B       | B16   |      6 |      2 | q0  | Yes              | No          | Different meaning; about bad words, not sessions                      |
| Sampled 8B: max over questions   | B       | B13   |      7 |    144 | q0  | Yes              | No          | Different content; GT unrelated to sessions                           |
| Sampled 8B: max over questions   | B       | B03   |      8 |    263 | q6  | Yes              | No          | Invented content vs backchannel                                       |
| Sampled 8B: max over questions   | B       | B02   |      9 |    947 | q6  | Yes              | No          | Entirely different utterance; unrelated to window sizes               |
| Sampled 8B: max over questions   | B       | B09   |     10 |     30 | q1  | Yes              | No          | Different utterance; backchannel                                      |
| Sampled 8B: max over questions   | B       | B04   |     11 |     70 | q1  | No               | No          | Identical backchannel                                                 |
| Sampled 8B: max over questions   | B       | B06   |     12 |     80 | q1  | No               | No          | Filler-only difference                                                |
| Sampled 8B: max over questions   | B       | B18   |     13 |     89 | q1  | Yes              | No          | ASR drops trip to Costa Caparica                                      |
| Sampled 8B: max over questions   | B       | B01   |     14 |    207 | q1  | Yes              | No          | Loses 'diagonal covariance'; adds content; unrelated to posters       |
| Sampled 8B: max over questions   | B       | B10   |     15 |    300 | q1  | Yes              | No          | ASR invents '20 ms minimum'; GT about FFT                             |
| Sampled 8B: max over questions   | B       | B08   |     16 |    354 | q1  | Yes              | No          | 'maximum likelihood' garbled; relates to q8, not q1                   |
| Sampled 8B: max over questions   | B       | B14   |     17 |    358 | q1  | No               | No          | Trivial hallucinated 'you' vs backchannel                             |
| Sampled 8B: max over questions   | B       | B15   |     18 |    385 | q1  | Yes              | No          | ASR invents '60 millisecond maximum'; GT unrelated to posters         |
| Sampled 8B: max over questions   | B       | B07   |     19 |    460 | q1  | No               | No          | Same apart from 'Yeah'                                                |
| Sampled 8B: max over questions   | B       | B19   |     20 |    462 | q1  | No               | No          | Filler/backchannel only                                               |

# Sampled-correctness importance: pilot report

Generated 2026-10-20 01:48 from `./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7`.

## What is reported

- **p(line, q)**: fraction of the K sampled answers graded correct when only that line is clarified.
- **z**: per-question robust z of p across lines, (p − median) / max(1.4826·MAD, 1/K). Line score = max over questions.
- **Labels**: per question, its top-3 lines with z ≥ 3.0; union over questions. No-op lines (GT == ASR) are never labeled.
- **Scores** are quiz scores out of the number of questions, averaged over the K samples.
- **Split-half**: z and labels recomputed from the first K/2 and last K/2 samples separately; agreement shows whether K/2 samples would have been enough.
- **Logp comparison**: logp z as in `generate_logp_deltas.py` (z-score across lines, max over questions). No-op lines excluded.

## Per-meeting summary

| meeting   |   lines |   no-op lines |   K |   base score |   median version score |   max version score | q with base outside 5-95%   |   labeled lines | top-20 dominant q (lines)   |   split-half spearman |   split-half top-20 overlap |   split-half label jaccard |   truncated answers |   judge parse failures |   answer time (min) |   score time (min) |
|:----------|--------:|--------------:|----:|-------------:|-----------------------:|--------------------:|:----------------------------|----------------:|:----------------------------|----------------------:|----------------------------:|---------------------------:|--------------------:|-----------------------:|--------------------:|-------------------:|
| ES2005d   |     559 |            25 |   8 |         6.38 |                   6.12 |                8    | 0%                          |               6 | q4 (15)                     |                 0.274 |                          10 |                      0.375 |                 454 |                      0 |                33.4 |                0   |
| IN1012    |    1047 |             8 |   8 |         5.88 |                   6.12 |                7.75 | 0%                          |               9 | q1 (18)                     |                 0.691 |                          11 |                      0.5   |               11436 |                      0 |               188.3 |                0.1 |
| TS3005d   |    1356 |            46 |   8 |         6    |                   6    |                7.62 | 0%                          |               9 | q1 (11)                     |                 0.653 |                           8 |                      0.4   |                4901 |                      0 |               194.3 |                0.1 |

## Agreement with logp deltas

| meeting   | logp file          |   spearman (line score) |   spearman (per-q z, pooled) |   top-20 overlap | logp dominant q (lines)   | sampled dominant q (lines)   |
|:----------|:-------------------|------------------------:|-----------------------------:|-----------------:|:--------------------------|:-----------------------------|
| ES2005d   | train_Qwen3-8B-FP8 |                  -0.04  |                       -0.057 |                2 | q7 (4)                    | q4 (15)                      |
| ES2005d   | train_32b          |                   0.039 |                       -0.026 |                2 | q3 (7)                    | q4 (15)                      |
| IN1012    | train_Qwen3-8B-FP8 |                   0.088 |                        0.001 |                1 | q9 (9)                    | q1 (18)                      |
| IN1012    | train_32b          |                   0.039 |                        0.017 |                2 | q9 (8)                    | q1 (18)                      |
| TS3005d   | train_Qwen3-8B-FP8 |                   0.172 |                        0.068 |                2 | q4 (6)                    | q1 (11)                      |
| TS3005d   | train_32b          |                   0.006 |                       -0.018 |                2 | q1 (8)                    | q1 (11)                      |

## Top lines per meeting

### ES2005d

| line | q | z | p base → p line | label | ASR | GT |
|---|---|---|---|---|---|---|
| 73 | q4 | 8.0 | 0.00 → 1.00 | yes | that's it, Stan. | That's it stand. |
| 74 | q4 | 8.0 | 0.00 → 1.00 | yes | It does. | Alright, okay. |
| 106 | q6 | 8.0 | 0.00 → 1.00 | yes | It's not in the traditional place but it's quite an uphill place. | It's not in the traditional place, but um it's quite an obvious place. |
| 75 | q4 | 7.0 | 0.00 → 0.88 | yes | It does also lie flat, but that's the yellow sand there represents the charging stand. | It does also lie flat, but that's the that yellow stand there represents the the chargi... |
| 279 | q4 | 6.0 | 0.00 → 0.75 |  | We discussed all the new ideas, but of course we could have reached any proper goals. W... | We discussed all the new ideas, but of course we couldn't reach any proper goals, we co... |
| 61 | q4 | 6.0 | 0.00 → 0.75 |  | All right. | Alright, okay. |
| 68 | q4 | 5.0 | 0.00 → 0.62 |  | I like the feeling. | Well I like the feel of it, I like the feel of it. |
| 46 | q3 | 4.7 | 0.25 → 1.00 | yes | He's not allowed to go. | What tomato? |
| 188 | q4 | 4.0 | 0.00 → 0.50 |  | Yeah, you're the shift button stick. | No maybe the shift button's stuck in. |
| 332 | q4 | 4.0 | 0.00 → 0.50 |  | It's this as well, sorry we've got to mention what we made out of kind of a rubbery, th... | It's this as well, sorry, we forgot to mention it'll be made out of kind of a rubbery l... |

Questions:

- q0: What was the main reason given for flattening the remote-control shape instead of keeping the curved version? — *To save two Euros.*
- q1: Why did the team decide not to prioritize adding the extra scroll buttons to the design? — *Because the team was trying to save money and did not want to spend extra if the current design was acceptable.*
- q2: What feature was described as the main interaction method for the first prototype, with buttons kept to a minimum? — *Speech recognition.*
- q3: Why was the tomato-based prototype included even though tomatoes had not been part of the earlier discussion? — *Because there was some red material left over, so they used it for a tomato design.*
- q4: What was the purpose of the yellow element on the first prototype? — *It was the slogan that needed to be incorporated.*
- q5: Why were the scroll buttons placed with some hesitation on the design? — *Because they could get bashed where people hold the device.*
- q6: What was the stated reason for the big red button being placed in the middle rather than in a traditional location? — *Because it was an obvious place and also out of the way.*
- q7: What specific problem did the team identify with the fruit-shaped designs when discussing company strategy and sales? — *That limiting the product to only a few fruit shapes would limit customers and therefore reduce sales and profit.*
- q8: What final overall evaluation score did the team agree on after averaging the individual ratings? — *Two.*
- q9: What conclusion did the team reach about the project after the evaluation and budget discussion? — *That they were done and could launch the product because it was within budget.*

### IN1012

| line | q | z | p base → p line | label | ASR | GT |
|---|---|---|---|---|---|---|
| 917 | q9 | 8.0 | 0.00 → 1.00 | yes | It's based on the iPhone | It's based on the diphones diphones, right? |
| 36 | q0 | 8.0 | 0.00 → 1.00 | yes | Thank you very much. Thank you very much. | I t I Uh I think it's quite interesting, but only annoying thing is this multiple sessi... |
| 952 | q1 | 4.7 | 0.12 → 1.00 |  | it the voices are called multisine, multisine | It the voices are called Multisyn, Multisyn uh sl uh that's some voice. |
| 1028 | q1 | 4.7 | 0.12 → 1.00 |  | then it takes a lot of time | So it's somebody and Simon King is playing with that, right, the t guy. |
| 960 | q1 | 4.7 | 0.12 → 1.00 |  | So what is this like the database of the speaker who speaks this text. | So voice is like the database of, the parallel speaker who speaks that this text, so. |
| 512 | q1 | 4.7 | 0.12 → 1.00 |  | his friends, some were now in telecom, but we were using it for a speech coding so we j... | He's French um somewhere now in teleconference like mm I don't know where he's working,... |
| 89 | q1 | 4.7 | 0.12 → 1.00 |  | before. | Yeah, w we went to the Costa Caparica there, with the bus. |
| 1002 | q1 | 4.7 | 0.12 → 1.00 |  | But only the problem is again acronyms and sometimes it expands sometimes it may not ex... | Well let's Oh really? |
| 510 | q1 | 4.7 | 0.12 → 1.00 |  | also some like not paper but I saw the algorithm I have been using them that called tem... | There is also some um like uh not paper, but I saw the algorithm I have been using even... |
| 504 | q1 | 4.7 | 0.12 → 1.00 |  | Yeah | If you use yeah, that becomes really Yeah. |

Questions:

- q0: What was identified as the main annoyance of the conference format because it prevented people from attending all talks? — *Multiple simultaneous sessions*
- q1: Why did one participant say posters were often preferable to oral presentations at the conference? — *Because in the same amount of time you can see two or three posters and talk directly to people*
- q2: What specific invited talk was singled out as especially good? — *The invited speaker talk about an implant in the ear*
- q3: What was the main technical advantage claimed for semi-tied covariance matrices over diagonal GMMs in speaker recognition? — *It made the results less sensitive to the choice of feature decorrelation procedure*
- q4: Why did the discussion say full covariance matrices are impractical for large speech systems? — *Because storing and computing with a full 39-by-39 covariance matrix for many models and mixtures is too expensive*
- q5: What was the key idea behind the variable-scale feature extraction paper? — *To adapt the analysis window length to the stationarity of each speech segment instead of using a fixed window*
- q6: What minimum and maximum window sizes were described for the variable-scale framing method? — *Minimum 20 milliseconds and maximum 60 milliseconds*
- q7: Why did the discussion say the frame shift could not simply be changed when using variable-length windows? — *Because changing the shift would change the modulation-spectrum Nyquist frequency for each window*
- q8: What was the main criterion used to choose the split point in the variable-scale segmentation method? — *A maximum-likelihood ratio test based on residual energy*
- q9: Why was Festival described as suitable for real-time text-to-speech on Linux? — *Because it is diphone-based*

### TS3005d

| line | q | z | p base → p line | label | ASR | GT |
|---|---|---|---|---|---|---|
| 36 | q1 | 8.0 | 0.00 → 1.00 | yes | and did you do something with Eiffeluation Criteria? | And uh m did you do something with evaluation criteria? |
| 9 | q1 | 8.0 | 0.00 → 1.00 | yes | Let's see the new one. | That's new one? |
| 99 | q1 | 8.0 | 0.00 → 1.00 |  | Should I take any comments about next time? Well, this is the total interface of the LC... | Would you like to make any comments about next uh Previous page, yeah, indeed. |
| 130 | q9 | 8.0 | 0.00 → 1.00 |  | So, | Looks good. |
| 1109 | q9 | 8.0 | 0.00 → 1.00 | yes | Yeah. All right. | Yep. |
| 1114 | q1 | 8.0 | 0.00 → 1.00 |  | But, wasn't he? | That wasn't me. |
| 778 | q9 | 8.0 | 0.00 → 1.00 |  | The curve is in dimension. | The curve is uh in a dimension. |
| 339 | q9 | 8.0 | 0.00 → 1.00 |  | if they read the menu. | If they read a manual. |
| 754 | q9 | 8.0 | 0.00 → 1.00 |  | 12 and a half. | Twelve and half. |
| 798 | q1 | 8.0 | 0.00 → 1.00 |  | Yeah, so the position is that. I think you should, just describe it as simple as bigger. | Yeah, so that does doesn't doesn't that mu I think Scrap sample speaker? |

Questions:

- q0: Why did the folders in the shared project documents get separated? — *Because the shared project documents folder was getting too busy.*
- q1: What two things were the prototype presenters expected to show before the evaluation discussion began? — *A prototype presentation and the evaluation criteria.*
- q2: Why did the team decide to keep teletext out of the options menu? — *Because teletext should still be used, but it was considered better as a separate button rather than under the options menu.*
- q3: What was the reason given for placing the options menu on the screen where it would not be very irritating? — *Because putting it there would keep it from taking up much of the screen and becoming irritating to see.*
- q4: What change to the remote's shape was proposed to make room for the screen and batteries? — *Make the remote thicker at the top and thinner at the bottom with a side curve.*
- q5: Which feature was identified as the one that should be dropped in order to bring the production cost under the budget? — *The sample speaker.*
- q6: What production-cost limit had to be met for the design to be accepted without redesign? — *Twelve euro fifty.*
- q7: Why did the team decide that the remote should mainly be sold to younger people? — *Because the design, colors, buttons, microphone, LCD screen, and other features were seen as more appealing to younger users.*
- q8: What was the main criticism of the way the team worked during the project? — *They worked past each other and did not communicate enough, so people had information the others did not.*
- q9: What improvement to the work process was suggested to address the communication problem? — *Have short sessions together to coordinate, then continue working separately.*

## Artifacts

- Raw per-meeting outcomes: `./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7/<meeting>.pt` (`correct[V,Q,K]`, `p_correct`, `mean_score`, `answer_table`, ...)
- Binary labels `{meeting: [bool]}`: `./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7/derived/labels_topm3_z3.0.pt`
- Soft scores (`robust_z`, `line_score`, `delta_vs_base`): `./shared/datasets/sampled_deltas/train_Qwen3-8B-FP8_K8_T0.7/derived/robust_z.pt`
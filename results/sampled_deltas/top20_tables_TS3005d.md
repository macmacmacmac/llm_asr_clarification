# TS3005d: top-20 lines per teacher metric

## Quiz questions and gold answers

| q | question | gold answer | wrong on baseline (Qwen3-8B) | wrong on baseline (Qwen3-32B) |
|---|---|---|---|---|
| q0 | Why did the folders in the shared project documents get separated? | Because the shared project documents folder was getting too busy. | No | No |
| q1 | What two things were the prototype presenters expected to show before the evaluation discussion began? | A prototype presentation and the evaluation criteria. | Yes | Yes |
| q2 | Why did the team decide to keep teletext out of the options menu? | Because teletext should still be used, but it was considered better as a separate button rather than under the options menu. | No | No |
| q3 | What was the reason given for placing the options menu on the screen where it would not be very irritating? | Because putting it there would keep it from taking up much of the screen and becoming irritating to see. | Yes | No |
| q4 | What change to the remote's shape was proposed to make room for the screen and batteries? | Make the remote thicker at the top and thinner at the bottom with a side curve. | No | No |
| q5 | Which feature was identified as the one that should be dropped in order to bring the production cost under the budget? | The sample speaker. | Yes | Yes |
| q6 | What production-cost limit had to be met for the design to be accepted without redesign? | Twelve euro fifty. | No | No |
| q7 | Why did the team decide that the remote should mainly be sold to younger people? | Because the design, colors, buttons, microphone, LCD screen, and other features were seen as more appealing to younger users. | No | No |
| q8 | What was the main criticism of the way the team worked during the project? | They worked past each other and did not communicate enough, so people had information the others did not. | No | No |
| q9 | What improvement to the work process was suggested to address the communication problem? | Have short sessions together to coordinate, then continue working separately. | Yes | No |

## Sampled 8B: expected total score

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. A line's score is the expected total quiz score after clarifying it (the number of correct answers out of 10, averaged over the 8 samples), standardized across all lines. The question column is the question whose chance of being answered correctly rose the most.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 154 | Thank you very much. | Yeah. | 4.61 | q9 | Yes |
| 2 | 683 | You got to excel to count. | You got Excel to count. | 4.25 | q9 | Yes |
| 3 | 9 | Let's see the new one. | That's new one? | 3.90 | q1 | Yes |
| 4 | 1236 | Yeah, everything. | Yeah, everything worked. | 3.54 | q9 | Yes |
| 5 | 36 | and did you do something with Eiffeluation Criteria? | And uh m did you do something with evaluation criteria? | 3.54 | q1 | Yes |
| 6 | 931 | And the rest is the same. | And the rest is the same, huh? | 3.54 | q9 | Yes |
| 7 | 469 | Ooh. | Ooh. | 3.54 | q9 | Yes |
| 8 | 212 | And it doesn't make it green than the other people. | I if you don't make it green, then the elder people won't won't like it. | 2.83 | q9 | Yes |
| 9 | 778 | The curve is in dimension. | The curve is uh in a dimension. | 2.83 | q9 | Yes |
| 10 | 325 | at L.O.L. | Yeah. | 2.83 | q1 | Yes |
| 11 | 729 | Dorothy. | Thirteen? | 2.83 | q9 | Yes |
| 12 | 313 | I don't know what this is. | Yep. | 2.83 | q9 | Yes |
| 13 | 1121 | I used it as a normal pen and only you used it to, you used it to, on the computer. Yeah, it did work. I used it too, but hello. | But I didn't I uh I used it too, but oh well. | 2.83 | q1 | Yes |
| 14 | 141 | Okay. | Okay, the evaluation criteria, huh? | 2.83 | q1 | Yes |
| 15 | 569 | I don't know if you have the black one. | Has oh yeah, do did have nah y you have the black one. | 2.47 | q9 | Yes |
| 16 | 1109 | Yeah. All right. | Yep. | 2.47 | q9 | Yes |
| 17 | 1253 | Thank you very much. | Huh. | 2.47 | q1 | Yes |
| 18 | 58 | And the RNR logo just has RNR now, but okay. And the question is so far. | And the R_ and R_ logo, it just says R_ and R_ now, but uh Any questions so far? | 2.47 | q1 | Yes |
| 19 | 51 | This is | View. | 2.47 | q1 | Yes |
| 20 | 1035 | Hey, can we race that? | Hey, but you can erase that. | 2.47 | q9 | Yes |

## Sampled 8B: max over questions

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. For each question, the fraction of the 8 answers graded correct is standardized across all lines. A line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 295 | I think it's a story. | I think it's a three. | 36.18 | q5 | Yes |
| 2 | 120 | Because if you put it on the top, you always get to see the options menu. | 'Cause if you put it on the top you always get see the the options menu. | 22.76 | q3 | Yes |
| 3 | 58 | And the RNR logo just has RNR now, but okay. And the question is so far. | And the R_ and R_ logo, it just says R_ and R_ now, but uh Any questions so far? | 11.34 | q3 | Yes |
| 4 | 1014 | Yeah. | Yeah. | 11.34 | q3 | Yes |
| 5 | 38 | Yeah. | Yep. | 11.34 | q3 | Yes |
| 6 | 1126 | You know, good, it does work. So make some designs. | Yeah, because it shou To make some designs, it is very easy. | 11.34 | q3 | Yes |
| 7 | 490 | Very true? | Very true? | 11.34 | q3 | Yes |
| 8 | 842 | we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've tested the way we've | Point twelve. | 7.53 | q3 | Yes |
| 9 | 6 | As you may notice, I created separate folders because it was maybe a little busy in our shared project document folder. | Um as you may have noticed I uh created separate folders because it was uh tending to get a little busy in our uh shared project documents uh folder. | 7.53 | q3 | Yes |
| 10 | 683 | You got to excel to count. | You got Excel to count. | 5.61 | q9 | Yes |
| 11 | 1236 | Yeah, everything. | Yeah, everything worked. | 5.61 | q9 | Yes |
| 12 | 212 | And it doesn't make it green than the other people. | I if you don't make it green, then the elder people won't won't like it. | 5.61 | q9 | Yes |
| 13 | 313 | I don't know what this is. | Yep. | 5.61 | q9 | Yes |
| 14 | 778 | The curve is in dimension. | The curve is uh in a dimension. | 5.61 | q9 | Yes |
| 15 | 729 | Dorothy. | Thirteen? | 5.61 | q9 | Yes |
| 16 | 339 | if they read the menu. | If they read a manual. | 5.61 | q9 | Yes |
| 17 | 1035 | Hey, can we race that? | Hey, but you can erase that. | 5.61 | q9 | Yes |
| 18 | 1109 | Yeah. All right. | Yep. | 5.61 | q9 | Yes |
| 19 | 815 | I'm not going to die. I'm going to die. | Uh. | 5.61 | q9 | Yes |
| 20 | 130 | So, | Looks good. | 5.61 | q9 | Yes |

## Logp 8B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-8B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 6 | As you may notice, I created separate folders because it was maybe a little busy in our shared project document folder. | Um as you may have noticed I uh created separate folders because it was uh tending to get a little busy in our uh shared project documents uh folder. | 7.18 | q0 | No |
| 2 | 650 | We're going to calculate the production costs, if they are under or at 12, you know, 50. We're good. And if you're not, we're going to redesign it. We have to do that. We're ready for quick. I think yes. | Um and we're going to calculate the production costs, and if they are uh under or at twelve Euro fifty, we're good, and if they're not we're going to uh re-design, but we have to do that uh very very quick I think, yes. | 5.22 | q6 | No |
| 3 | 1067 | You had information. I also had some. | Yeah, you had information I didn't have and then uh Oh right. | 4.53 | q8 | No |
| 4 | 36 | and did you do something with Eiffeluation Criteria? | And uh m did you do something with evaluation criteria? | 4.21 | q1 | Yes |
| 5 | 28 | We're finally getting somewhere. | We're finally getting somewhere hopefully. | 4.17 | q4 | No |
| 6 | 656 | We're going to the hospital. | f fifty five Euros. | 3.63 | q6 | No |
| 7 | 346 | And that's everything it does to speak your connection. | And that's everything it does, the speech recognition. | 3.41 | q4 | No |
| 8 | 483 | We're not going to all mainly be salt to younger people. | Remote control mainly be sold to younger people. | 3.39 | q7 | No |
| 9 | 927 | with you. | Mm yep. | 3.39 | q2 | No |
| 10 | 78 | using the scroll button on the sides. You push the scroll button. And then it's a flat-side. And drop down menu or a full out menu comes out and you get the options becoming available. | Uh using the scroll button on sides uh y uh I uh um Yeah, you push the scroll button and a drop down menu or a pull out menu comes out and uh you get uh you get the options uh becoming available. | 3.35 | q4 | No |
| 11 | 426 | I think it's so 3. | I think it's a three. | 3.33 | q2 | No |
| 12 | 757 | We could use the scroll wheel. We could just adjust the regular scroll wheel. You have to tap the options window button here and scroll down with the button. No one else scroll wheel. I think we should use the sample speaker. | We could use I think we should scrap the sample speaker. | 3.33 | q9 | Yes |
| 13 | 67 | You perhaps you should make it a bit big so people know it's there and to not forget it to function it doesn't really have to be small a big of course | Oh y you perhaps you should make it a bit big, so people know it's there and uh Uh Hmm. | 3.28 | q4 | No |
| 14 | 62 | Just | Yeah. | 3.25 | q4 | No |
| 15 | 868 | the touch. | S touch. | 3.25 | q3 | Yes |
| 16 | 1288 | Get fired as a doorknob. | Can you find it as a J_ PEG? | 3.22 | q1 | Yes |
| 17 | 61 | I just an idea about the growth and the possibility. | yeah, just uh just an idea about how to m th that could also be possible. | 3.17 | q0 | No |
| 18 | 53 | Just an example color so there's one of the colors we would like to see in our photo | Just example colour, so uh there's one of the colours we would like to uh see in our prototype. | 3.13 | q4 | No |
| 19 | 331 | Because it has the regular controls as you can see in the screen now. And you don't have all the other options always on your screen. | Yeah, but because it has the regular uh controls, li uh as you can see in the screen now, and uh you don't have all the other options uh always uh on your screen. | 3.09 | q3 | Yes |
| 20 | 0 | Good. | Good. | 3.09 | q4 | No |

## Logp 32B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-32B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-32B)? |
|---|---|---|---|---|---|---|
| 1 | 757 | We could use the scroll wheel. We could just adjust the regular scroll wheel. You have to tap the options window button here and scroll down with the button. No one else scroll wheel. I think we should use the sample speaker. | We could use I think we should scrap the sample speaker. | 14.58 | q5 | Yes |
| 2 | 88 | OK, it's also nice to see that we make a small menu, the options menu are becoming available when pressing the scroll button, and the opportunity to use the date text, which is still used, and we think that it's very handy to put it not under the options menu, but you know, | Okay, um it's also uh nice to see that um we made a small uh menu, uh the options menu uh becoming available when pressing the uh scroll uh button, and the opportunity to use the teletext, whi which is used uh which should still be used and we think that it's uh very handy to put it uh not uh under the options menu, but in uh Yes. | 10.10 | q2 | No |
| 3 | 6 | As you may notice, I created separate folders because it was maybe a little busy in our shared project document folder. | Um as you may have noticed I uh created separate folders because it was uh tending to get a little busy in our uh shared project documents uh folder. | 10.09 | q0 | No |
| 4 | 831 | But I think the only option is to top the set speaker to knock the samples | But I think the only option is to drop the uh sample speaker. | 8.22 | q5 | Yes |
| 5 | 978 | 12-year-old, 15. | Twelve Euro fifty. | 8.14 | q6 | No |
| 6 | 141 | Okay. | Okay, the evaluation criteria, huh? | 7.35 | q1 | Yes |
| 7 | 1147 | only thing that we work through. | Only thing that we worked through, past each other. | 7.23 | q8 | No |
| 8 | 650 | We're going to calculate the production costs, if they are under or at 12, you know, 50. We're good. And if you're not, we're going to redesign it. We have to do that. We're ready for quick. I think yes. | Um and we're going to calculate the production costs, and if they are uh under or at twelve Euro fifty, we're good, and if they're not we're going to uh re-design, but we have to do that uh very very quick I think, yes. | 6.65 | q6 | No |
| 9 | 1076 | or maybe session of 5 minutes to get over something and then work. | Yeah, no, or maybe session of five minutes together or something, and then work separate. | 6.55 | q9 | No |
| 10 | 20 | I'm going to post it. | And we have a evaluation left here. | 6.54 | q1 | Yes |
| 11 | 1067 | You had information. I also had some. | Yeah, you had information I didn't have and then uh Oh right. | 6.23 | q8 | No |
| 12 | 35 | Thank you very much. | Yes, you are. | 5.64 | q1 | Yes |
| 13 | 483 | We're not going to all mainly be salt to younger people. | Remote control mainly be sold to younger people. | 4.60 | q7 | No |
| 14 | 51 | This is | View. | 4.56 | q2 | No |
| 15 | 36 | and did you do something with Eiffeluation Criteria? | And uh m did you do something with evaluation criteria? | 4.51 | q1 | Yes |
| 16 | 203 | Okay, we have to rate. Yeah, I will have to rate. Is this an infection of? Two. Is one. Four. This is seven. So fancy enough means that it comes to the younger people and the elderly people. I think. I think so. I think so. | 'Cause otherwise I think i I think it does. | 4.05 | q7 | No |
| 17 | 33 | Can we go it? | We could. | 4.05 | q1 | Yes |
| 18 | 19 | That way up there. | Hmm hmm. | 4.03 | q1 | Yes |
| 19 | 23 | Well, not great documents this time. | Well not main documents this time. | 3.83 | q1 | Yes |
| 20 | 17 | Oh, no this is just one big document so you can leave that forever this. | Oh no, this is just one big document, so you can leave that wherever it is. | 3.47 | q1 | Yes |

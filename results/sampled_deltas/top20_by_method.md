# Top-20 lines per method (pilot, train)

Cell = `line (q#) z`. **Bold** = line is in the top 20 of at least one other method.
Sampled: Qwen3-8B, K=8, T=0.7; z = per-question mean/std of p(correct) across lines. Logp: per-question mean/std of the gold-answer logp across lines. No-op lines excluded.

## ES2005d

| rank | sampled 8B | logp 8B | logp 32B |
|---|---|---|---|
| 1 | **106 (q6) 23.1** | **144 (q0) 10.0** | 85 (q5) 13.8 |
| 2 | **73 (q4) 9.1** | **231 (q1) 7.7** | **109 (q6) 12.3** |
| 3 | **74 (q4) 9.1** | **232 (q1) 4.6** | **144 (q0) 10.3** |
| 4 | 75 (q4) 8.0 | **56 (q2) 4.2** | **46 (q3) 8.9** |
| 5 | 279 (q4) 6.8 | **109 (q6) 4.1** | **106 (q6) 8.5** |
| 6 | 61 (q4) 6.8 | 13 (q6) 4.0 | 47 (q3) 7.9 |
| 7 | 68 (q4) 5.6 | **73 (q4) 3.7** | **232 (q1) 7.6** |
| 8 | **46 (q3) 5.3** | 305 (q7) 3.6 | **231 (q1) 7.4** |
| 9 | 412 (q4) 4.5 | **74 (q4) 3.4** | 45 (q3) 7.1 |
| 10 | 15 (q4) 4.5 | 538 (q9) 3.2 | 42 (q3) 4.9 |
| 11 | 332 (q4) 4.5 | 419 (q2) 3.2 | 48 (q3) 4.6 |
| 12 | 392 (q4) 4.5 | 12 (q9) 3.1 | 53 (q4) 4.4 |
| 13 | 188 (q4) 4.5 | 111 (q9) 3.0 | **56 (q2) 4.3** |
| 14 | 171 (q4) 4.5 | 556 (q7) 3.0 | 134 (q0) 4.0 |
| 15 | 37 (q4) 4.5 | 233 (q7) 3.0 | 238 (q0) 3.6 |
| 16 | 422 (q3) 3.7 | 241 (q6) 3.0 | 3 (q3) 3.6 |
| 17 | 294 (q4) 3.3 | 248 (q7) 2.9 | 468 (q8) 3.5 |
| 18 | 58 (q4) 3.3 | 266 (q8) 2.9 | 342 (q3) 3.4 |
| 19 | 28 (q3) 2.9 | 229 (q2) 2.8 | 424 (q7) 3.3 |
| 20 | 431 (q3) 2.9 | 437 (q6) 2.6 | 129 (q1) 3.3 |

Shared lines: sampled 8B ∩ logp 8B = 2, sampled 8B ∩ logp 32B = 2, logp 8B ∩ logp 32B = 5

<details><summary>Line text (union of all three lists)</summary>

| line | in | ASR | GT |
|---|---|---|---|
| 3 | logp 32B | or else I'll use the PowerPoint I guess. | We'll sta I'll use the PowerPoint, I guess. |
| 12 | logp 8B | Thank you for watching. | Hmm. |
| 13 | logp 8B | We... | Um Um we Yeah. |
| 15 | sampled 8B | Not right, OK. OK. | Alright, okay, yeah. |
| 28 | sampled 8B | So, I don't even have a soul of them. | I don't even have this on. |
| 37 | sampled 8B | Okay. | Yeah. |
| 42 | logp 32B | Thank you. | Three? |
| 45 | logp 32B | the other one. | Tomato? |
| 46 | sampled 8B, logp 32B | He's not allowed to go. | What tomato? |
| 47 | logp 32B | No, no, no. | I don't recall a tomato. |
| 48 | logp 32B | Yeah, well, yeah, we had some red left over. | Oh yeah, well yeah, we had v some red left over. |
| 53 | logp 32B | The yellow there is the slogan that we need to incorporate very sim... | The yellow there is the um the slogan, yeah, that we need to incorp... |
| 56 | logp 8B, logp 32B | If you do need buttons, you can flip it over on the some there. But... | If you do need buttons, you can flip it over, and there's some ther... |
| 58 | sampled 8B | Okay so the buttons would be like, you know, individual users, yeah. | Okay, so the buttons would be like, you know individual users, or Y... |
| 61 | sampled 8B | All right. | Alright, okay. |
| 68 | sampled 8B | I like the feeling. | Well I like the feel of it, I like the feel of it. |
| 73 | sampled 8B, logp 8B | that's it, Stan. | That's it stand. |
| 74 | sampled 8B, logp 8B | It does. | Alright, okay. |
| 75 | sampled 8B | It does also lie flat, but that's the yellow sand there represents ... | It does also lie flat, but that's the that yellow stand there repre... |
| 85 | logp 32B | when we weren't sure about putting them there because it kind of co... | We n were weren't sure about putting them there, because um i it's ... |
| 106 | sampled 8B, logp 32B | It's not in the traditional place but it's quite an uphill place. | It's not in the traditional place, but um it's quite an obvious place. |
| 109 | logp 8B, logp 32B | It's kind of real thanks. | It's out of the way as well, I suppose, so. |
| 111 | logp 8B | So there we go. And we have the banana. Oh yeah. At least one too. | So there we go and and um we have the banana-based one too. |
| 129 | logp 32B | So, are there any improvements or issues? Yes, there are issues. Wh... | So are there any um improvements or issues or Oh. |
| 134 | logp 32B | Unfortunately the unit we are currently going to produce minus the ... | Um unfortunately the unit we are currently going to produce minus t... |
| 144 | logp 8B, logp 32B | So basically, in order to save our two euros, I was thinking that w... | So basically, um in order to save our two Euros um I was thinking t... |
| 171 | sampled 8B | I don't look like that character's a curve, I think that's just a s... | I don't think that counts as a curve, I think that's just a shape. |
| 188 | sampled 8B | Yeah, you're the shift button stick. | No maybe the shift button's stuck in. |
| 229 | logp 8B | Right, so that's finance. I don't know what we can do it, we can we... | Right, so that's finances and I dunno what we what could we reckon ... |
| 231 | logp 8B, logp 32B | for maybe we could have something but maybe if what is that said th... | Well maybe we could add something, but maybe if What do you th We'r... |
| 232 | logp 8B, logp 32B | if we're happy with the design as they put 20 money if we don't hav... | Yeah, if we're happy with the design there's no point in spending m... |
| 233 | logp 8B | So if we happy with Zion as name, if we don't have to | So Yeah. |
| 238 | logp 32B | We have under 12 years, 50. | We have under twelve Euros fifty. |
| 241 | logp 8B | Don't worry. | Oh. |
| 248 | logp 8B | room for creativity when we happy with that. | Uh room for creativity, were we happy with that? |
| 266 | logp 8B | Yeah. | Yeah, yeah. |
| 279 | sampled 8B | We discussed all the new ideas, but of course we could have reached... | We discussed all the new ideas, but of course we couldn't reach any... |
| 294 | sampled 8B | to- | Uh. |
| 305 | logp 8B | So is it go back in? No, it's not. It's really, really annoying. Ha... | So So Uh Brian, have you have you finished? |
| 332 | sampled 8B | It's this as well, sorry we've got to mention what we made out of k... | It's this as well, sorry, we forgot to mention it'll be made out of... |
| 342 | logp 32B | Oh, can you pass the tumulties? | Oh could you pass the tomato please. |
| 392 | sampled 8B | 7. | Seven. |
| 412 | sampled 8B | But you can give your views based on technology, whether the techno... | But you can give your views based on technology, whether the techno... |
| 419 | logp 8B | You'll go far back? Basically, requirements. | You will go for one. |
| 422 | sampled 8B | For me personally it is everything fine it may be having good desig... | Uh for me personally it is everything fine, it may be having good d... |
| 424 | logp 32B | the orange and the third one is what you get, that is not a fruit l... | Yeah th Come on that was the tha 'Kay. |
| 431 | sampled 8B | So we are showing our preference for particular fruits, 2 or 3 kind... | So we are showing our preference for particular fruits, two or thre... |
| 437 | logp 8B | some children like to buy banana chips, some apple, some even pinea... | Some children like to buy banana shape, some apple shape, some even... |
| 468 | logp 32B | because it is 1.8 just using 2 so we will do 2.10. | because it is one point eight uh two, so we will do two. |
| 538 | logp 8B | So we can say 211, 22. | So we can say two or one Okay. |
| 556 | logp 8B | Yep. | Yeah. |

</details>

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

## IN1012

| rank | sampled 8B | logp 8B | logp 32B |
|---|---|---|---|
| 1 | **917 (q9) 32.2** | **928 (q9) 13.7** | **332 (q7) 19.5** |
| 2 | **36 (q0) 28.5** | **332 (q7) 9.9** | **928 (q9) 13.1** |
| 3 | 775 (q6) 23.1 | **57 (q2) 9.6** | **917 (q9) 12.6** |
| 4 | 531 (q6) 17.3 | **53 (q2) 7.3** | 15 (q0) 12.1 |
| 5 | 757 (q6) 11.5 | **44 (q1) 7.0** | **918 (q9) 11.8** |
| 6 | **2 (q0) 10.6** | **927 (q9) 7.0** | 340 (q7) 10.0 |
| 7 | 144 (q0) 10.6 | **235 (q4) 6.0** | **36 (q0) 9.1** |
| 8 | 263 (q6) 5.7 | **918 (q9) 5.5** | **927 (q9) 8.6** |
| 9 | 947 (q6) 5.7 | **289 (q6) 5.4** | **235 (q4) 8.5** |
| 10 | 30 (q1) 2.4 | **926 (q9) 5.3** | **57 (q2) 8.4** |
| 11 | 70 (q1) 2.4 | **930 (q9) 4.8** | **68 (q2) 6.1** |
| 12 | 80 (q1) 2.4 | **917 (q9) 4.8** | **926 (q9) 5.9** |
| 13 | 89 (q1) 2.4 | 4 (q0) 4.2 | **930 (q9) 5.7** |
| 14 | 207 (q1) 2.4 | **2 (q0) 4.2** | **44 (q1) 5.6** |
| 15 | 300 (q1) 2.4 | **942 (q9) 3.8** | **942 (q9) 5.4** |
| 16 | 354 (q1) 2.4 | **68 (q2) 3.8** | **289 (q6) 4.8** |
| 17 | 358 (q1) 2.4 | 924 (q9) 3.7 | 293 (q6) 4.8 |
| 18 | 385 (q1) 2.4 | 590 (q6) 3.6 | **53 (q2) 4.8** |
| 19 | 460 (q1) 2.4 | **938 (q9) 3.6** | **938 (q9) 4.7** |
| 20 | 462 (q1) 2.4 | 330 (q7) 3.6 | 887 (q0) 4.4 |

Shared lines: sampled 8B ∩ logp 8B = 2, sampled 8B ∩ logp 32B = 2, logp 8B ∩ logp 32B = 15

<details><summary>Line text (union of all three lists)</summary>

| line | in | ASR | GT |
|---|---|---|---|
| 2 | sampled 8B, logp 8B | I think the most of the bad words I had to begin with is precise time. | I think the the most of the bad words are at the beginning when the... |
| 4 | logp 8B | So you're stuck. I think right. | So you start I think, right? |
| 15 | logp 32B | You can't attend multi-processing as the same thing. | Uh you can't attend multiple session at the same time, so. |
| 30 | sampled 8B | I just would be right. | Yeah, I see see. |
| 36 | sampled 8B, logp 32B | Thank you very much. Thank you very much. | I t I Uh I think it's quite interesting, but only annoying thing is... |
| 44 | logp 8B, logp 32B | So in 20 minutes you can say at least 2-3 posters and you can direc... | So in t twenty minutes you can see at least two, three posters, and... |
| 53 | logp 8B, logp 32B | How many patients did you help? Priced patients are hardly and full... | So how many presentation you you I realised it were that there were... |
| 57 | logp 8B, logp 32B | I like the invite and speaker about the implant in the year. | I liked the invited speaker um about the implant in the ear. |
| 68 | logp 8B, logp 32B | the inventor of that implant and talk for the year, I think. | He's the the inventor of that implant and for for the ear I think. |
| 70 | sampled 8B | Yeah. | Yeah. |
| 80 | sampled 8B | But | Hmm. |
| 89 | sampled 8B | before. | Yeah, w we went to the Costa Caparica there, with the bus. |
| 144 | sampled 8B | Sometimes, it will lead to my danger. If you would have to spend a ... | Some time, yes, but not so many days or Oh yeah, yeah, that's true. |
| 207 | sampled 8B | If you compare it with some baseline, let's say you use just GMM wi... | But if you compare no, if you compare it with some baseline, let's ... |
| 235 | logp 8B, logp 32B | No problem, but T-FTC is so many models and so many, and so many. | Ne no problem, but if you see so many models and so many mixtures y... |
| 263 | sampled 8B | The fun PCO, the MLT. | So from Mm-hmm. |
| 289 | logp 8B, logp 32B | as much as possible like a but they are differently to issue some t... | Yeah, as much as possible like but yeah, definitely you have to ass... |
| 293 | logp 32B | You see, it's more than minimum. | So it's not so mm Two point five milliseconds. |
| 300 | sampled 8B | 20 millies minimum | This is maybe because of F_F_T_U_ or these things. |
| 330 | logp 8B | Well, why to keep variable variability in length of the frames? Wel... | W you want to Actually the problem's again uh uh you see the shift ... |
| 332 | logp 8B, logp 32B | So again if you change this shift the modulation spectrum, the freq... | So again, if you change this shift, the mod the Nyquist the modulat... |
| 340 | logp 32B | So again this may blow some frequency transition or you're... But t... | So again, this may blur some frequency transitions or yeah, but the... |
| 354 | sampled 8B | So then what he does is like he proposed like some like loot ratio ... | So then what he does is like he proposed uh like some like ratio te... |
| 358 | sampled 8B | you | Mm-hmm. |
| 385 | sampled 8B | then you shift to 25 and so forth and then if you the maximum is 60... | Then you shift to twenty five and so forth until you Okay, and and ... |
| 460 | sampled 8B | By the length. Yeah. | By the length. |
| 462 | sampled 8B | And | Mm-hmm. |
| 531 | sampled 8B | He was quoting... It has been used for speech coding... Nobody used... | Ne he was re he was quoting But that Yeah, he was uh quoting that p... |
| 590 | logp 8B | then people feel that they have to show the data. | No okay, so So it should be every papers then should contain all th... |
| 757 | sampled 8B | Yeah | Mm-hmm. |
| 775 | sampled 8B | If you have not | Okay, so we can hope that Nokia will be using tandem tandem P_L_P_s. |
| 887 | logp 32B | But maybe like if you can't really recognize you will be really happy. | Yeah, but maybe like if it can't really recognise you it will be re... |
| 917 | sampled 8B, logp 8B, logp 32B | It's based on the iPhone | It's based on the diphones diphones, right? |
| 918 | logp 8B, logp 32B | A for your phones, right? | Diphone, yeah, yeah. |
| 924 | logp 8B | I've been very used to this game, I think the first game was a good... | Uh Festival is good, like now it comes at all the Linux boxes also,... |
| 926 | logp 8B, logp 32B | This I was tried to install it and there are the voices like cat vo... | 'Cause the the diphone that I was tried to install it and there the... |
| 927 | logp 8B, logp 32B | because I don't know which one is running or if it's time for one o... | Because uh I don't know which which um what i what is running where... |
| 928 | logp 8B, logp 32B | Now, mostly the festival what you get on line excumations is defini... | No, most of the the Festival, uh what you get on Linux machines, is... |
| 930 | logp 8B, logp 32B | you can get it installed in your machine here | So it's uh just uh wha samples of the diphones, and it choo chooses... |
| 938 | logp 8B, logp 32B | And have we tried it? Yes, I've tried and there are some built in l... | Yeah, you can you can easily yes, I d I've tried, but there are som... |
| 942 | logp 8B, logp 32B | it is not die phones you have really large inventory of this speech... | It's not diphones, you have really large inventory of the speech. |
| 947 | sampled 8B | But maybe you can just take your stuff out. | But it it's not here so far. |

</details>

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

## TS3005d

| rank | sampled 8B | logp 8B | logp 32B |
|---|---|---|---|
| 1 | 295 (q5) 36.2 | **6 (q0) 7.2** | **757 (q5) 14.6** |
| 2 | 120 (q3) 22.8 | **650 (q6) 5.2** | 88 (q2) 10.1 |
| 3 | 58 (q3) 11.3 | **1067 (q8) 4.5** | **6 (q0) 10.1** |
| 4 | 1014 (q3) 11.3 | **36 (q1) 4.2** | 831 (q5) 8.2 |
| 5 | 38 (q3) 11.3 | 28 (q4) 4.2 | 978 (q6) 8.1 |
| 6 | 1126 (q3) 11.3 | 656 (q6) 3.6 | 141 (q1) 7.4 |
| 7 | 490 (q3) 11.3 | 346 (q4) 3.4 | 1147 (q8) 7.2 |
| 8 | 842 (q3) 7.5 | **483 (q7) 3.4** | **650 (q6) 6.6** |
| 9 | **6 (q3) 7.5** | 927 (q2) 3.4 | 1076 (q9) 6.5 |
| 10 | 683 (q9) 5.6 | 78 (q4) 3.3 | 20 (q1) 6.5 |
| 11 | 1236 (q9) 5.6 | 426 (q2) 3.3 | **1067 (q8) 6.2** |
| 12 | 212 (q9) 5.6 | **757 (q9) 3.3** | 35 (q1) 5.6 |
| 13 | 313 (q9) 5.6 | 67 (q4) 3.3 | **483 (q7) 4.6** |
| 14 | 778 (q9) 5.6 | 62 (q4) 3.3 | 51 (q2) 4.6 |
| 15 | 729 (q9) 5.6 | 868 (q3) 3.2 | **36 (q1) 4.5** |
| 16 | 339 (q9) 5.6 | 1288 (q1) 3.2 | 203 (q7) 4.0 |
| 17 | 1035 (q9) 5.6 | 61 (q0) 3.2 | 33 (q1) 4.0 |
| 18 | 1109 (q9) 5.6 | 53 (q4) 3.1 | 19 (q1) 4.0 |
| 19 | 815 (q9) 5.6 | 331 (q3) 3.1 | 23 (q1) 3.8 |
| 20 | 130 (q9) 5.6 | 0 (q4) 3.1 | 17 (q1) 3.5 |

Shared lines: sampled 8B ∩ logp 8B = 1, sampled 8B ∩ logp 32B = 1, logp 8B ∩ logp 32B = 6

<details><summary>Line text (union of all three lists)</summary>

| line | in | ASR | GT |
|---|---|---|---|
| 0 | logp 8B | Good. | Good. |
| 6 | sampled 8B, logp 8B, logp 32B | As you may notice, I created separate folders because it was maybe ... | Um as you may have noticed I uh created separate folders because it... |
| 17 | logp 32B | Oh, no this is just one big document so you can leave that forever ... | Oh no, this is just one big document, so you can leave that whereve... |
| 19 | logp 32B | That way up there. | Hmm hmm. |
| 20 | logp 32B | I'm going to post it. | And we have a evaluation left here. |
| 23 | logp 32B | Well, not great documents this time. | Well not main documents this time. |
| 28 | logp 8B | We're finally getting somewhere. | We're finally getting somewhere hopefully. |
| 33 | logp 32B | Can we go it? | We could. |
| 35 | logp 32B | Thank you very much. | Yes, you are. |
| 36 | logp 8B, logp 32B | and did you do something with Eiffeluation Criteria? | And uh m did you do something with evaluation criteria? |
| 38 | sampled 8B | Yeah. | Yep. |
| 51 | logp 32B | This is | View. |
| 53 | logp 8B | Just an example color so there's one of the colors we would like to... | Just example colour, so uh there's one of the colours we would like... |
| 58 | sampled 8B | And the RNR logo just has RNR now, but okay. And the question is so... | And the R_ and R_ logo, it just says R_ and R_ now, but uh Any ques... |
| 61 | logp 8B | I just an idea about the growth and the possibility. | yeah, just uh just an idea about how to m th that could also be pos... |
| 62 | logp 8B | Just | Yeah. |
| 67 | logp 8B | You perhaps you should make it a bit big so people know it's there ... | Oh y you perhaps you should make it a bit big, so people know it's ... |
| 78 | logp 8B | using the scroll button on the sides. You push the scroll button. A... | Uh using the scroll button on sides uh y uh I uh um Yeah, you push ... |
| 88 | logp 32B | OK, it's also nice to see that we make a small menu, the options me... | Okay, um it's also uh nice to see that um we made a small uh menu, ... |
| 120 | sampled 8B | Because if you put it on the top, you always get to see the options... | 'Cause if you put it on the top you always get see the the options ... |
| 130 | sampled 8B | So, | Looks good. |
| 141 | logp 32B | Okay. | Okay, the evaluation criteria, huh? |
| 203 | logp 32B | Okay, we have to rate. Yeah, I will have to rate. Is this an infect... | 'Cause otherwise I think i I think it does. |
| 212 | sampled 8B | And it doesn't make it green than the other people. | I if you don't make it green, then the elder people won't won't lik... |
| 295 | sampled 8B | I think it's a story. | I think it's a three. |
| 313 | sampled 8B | I don't know what this is. | Yep. |
| 331 | logp 8B | Because it has the regular controls as you can see in the screen no... | Yeah, but because it has the regular uh controls, li uh as you can ... |
| 339 | sampled 8B | if they read the menu. | If they read a manual. |
| 346 | logp 8B | And that's everything it does to speak your connection. | And that's everything it does, the speech recognition. |
| 426 | logp 8B | I think it's so 3. | I think it's a three. |
| 483 | logp 8B, logp 32B | We're not going to all mainly be salt to younger people. | Remote control mainly be sold to younger people. |
| 490 | sampled 8B | Very true? | Very true? |
| 650 | logp 8B, logp 32B | We're going to calculate the production costs, if they are under or... | Um and we're going to calculate the production costs, and if they a... |
| 656 | logp 8B | We're going to the hospital. | f fifty five Euros. |
| 683 | sampled 8B | You got to excel to count. | You got Excel to count. |
| 729 | sampled 8B | Dorothy. | Thirteen? |
| 757 | logp 8B, logp 32B | We could use the scroll wheel. We could just adjust the regular scr... | We could use I think we should scrap the sample speaker. |
| 778 | sampled 8B | The curve is in dimension. | The curve is uh in a dimension. |
| 815 | sampled 8B | I'm not going to die. I'm going to die. | Uh. |
| 831 | logp 32B | But I think the only option is to top the set speaker to knock the ... | But I think the only option is to drop the uh sample speaker. |
| 842 | sampled 8B | we've tested the way we've tested the way we've tested the way we'v... | Point twelve. |
| 868 | logp 8B | the touch. | S touch. |
| 927 | logp 8B | with you. | Mm yep. |
| 978 | logp 32B | 12-year-old, 15. | Twelve Euro fifty. |
| 1014 | sampled 8B | Yeah. | Yeah. |
| 1035 | sampled 8B | Hey, can we race that? | Hey, but you can erase that. |
| 1067 | logp 8B, logp 32B | You had information. I also had some. | Yeah, you had information I didn't have and then uh Oh right. |
| 1076 | logp 32B | or maybe session of 5 minutes to get over something and then work. | Yeah, no, or maybe session of five minutes together or something, a... |
| 1109 | sampled 8B | Yeah. All right. | Yep. |
| 1126 | sampled 8B | You know, good, it does work. So make some designs. | Yeah, because it shou To make some designs, it is very easy. |
| 1147 | logp 32B | only thing that we work through. | Only thing that we worked through, past each other. |
| 1236 | sampled 8B | Yeah, everything. | Yeah, everything worked. |
| 1288 | logp 8B | Get fired as a doorknob. | Can you find it as a J_ PEG? |

</details>

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

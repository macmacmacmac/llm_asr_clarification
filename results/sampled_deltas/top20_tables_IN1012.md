# IN1012: top-20 lines per teacher metric

## Quiz questions and gold answers

| q | question | gold answer | wrong on baseline (Qwen3-8B) | wrong on baseline (Qwen3-32B) |
|---|---|---|---|---|
| q0 | What was identified as the main annoyance of the conference format because it prevented people from attending all talks? | Multiple simultaneous sessions | Yes | No |
| q1 | Why did one participant say posters were often preferable to oral presentations at the conference? | Because in the same amount of time you can see two or three posters and talk directly to people | Yes | No |
| q2 | What specific invited talk was singled out as especially good? | The invited speaker talk about an implant in the ear | No | No |
| q3 | What was the main technical advantage claimed for semi-tied covariance matrices over diagonal GMMs in speaker recognition? | It made the results less sensitive to the choice of feature decorrelation procedure | No | No |
| q4 | Why did the discussion say full covariance matrices are impractical for large speech systems? | Because storing and computing with a full 39-by-39 covariance matrix for many models and mixtures is too expensive | No | No |
| q5 | What was the key idea behind the variable-scale feature extraction paper? | To adapt the analysis window length to the stationarity of each speech segment instead of using a fixed window | No | No |
| q6 | What minimum and maximum window sizes were described for the variable-scale framing method? | Minimum 20 milliseconds and maximum 60 milliseconds | Yes | Yes |
| q7 | Why did the discussion say the frame shift could not simply be changed when using variable-length windows? | Because changing the shift would change the modulation-spectrum Nyquist frequency for each window | No | No |
| q8 | What was the main criterion used to choose the split point in the variable-scale segmentation method? | A maximum-likelihood ratio test based on residual energy | No | No |
| q9 | Why was Festival described as suitable for real-time text-to-speech on Linux? | Because it is diphone-based | Yes | No |

## Sampled 8B: expected total score

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. A line's score is the expected total quiz score after clarifying it (the number of correct answers out of 10, averaged over the 8 samples), standardized across all lines. The question column is the question whose chance of being answered correctly rose the most.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 36 | Thank you very much. Thank you very much. | I t I Uh I think it's quite interesting, but only annoying thing is this multiple sessions. | 4.61 | q0 | Yes |
| 2 | 207 | If you compare it with some baseline, let's say you use just GMM with that overall. No, I think they are good results, but just using that also differently... I think it was a slightly better area. And no decoration... Maybe decoration again they do DCT or the other. | But if you compare no, if you compare it with some baseline, let's say you use just G_M_M_ with diagonal covariance matrices A and no decorrelation before like uh Yeah. | 2.44 | q1 | Yes |
| 3 | 1014 | And I thought that that's the guy from Cesario from Portland, who was there? Oh, doing. Oh, this is my this, you're Hanukkah, right? Yeah, but there was... No, they did what they did Portland, Ogeyay and they did LPC-based synthesis. The festival is already like it's the first start, but that that was the guy who is on a wheelchair now. He cannot move even, he's very in like Indic, I'll say that. | And I thought it that's the guy from C_S_L_U_, from Portland, who was there during f Yeah, but there was No, no, but that that was the guy who's on a wheelchair now, he cannot m move even, he's very in like indic how to say that? | 2.44 | q1 | Yes |
| 4 | 949 | okay, I can. | Okay, I'm I'm fine with this, but I can make it run on my machine. | 2.44 | q1 | Yes |
| 5 | 720 | There are already people. But they end for LVCSR, people are using PLP, so they must use these. Nothing or this. And who is the one who makes who actually uses what we try to do here? These are companies, because it's not scientists. Scientists just start from PLP and design a new feature, a new backend or a new anything. But they start from the old stuff. | So there are already people but again like uh, at the end like for L_V_C_S_R_ people are using P_L_P_s or M_F_C_C_s or nothing of these fancy feature Yeah. | 2.44 | q1 | Yes |
| 6 | 354 | So then what he does is like he proposed like some like loot ratio test based on action like loot. | So then what he does is like he proposed uh like some like ratio test based on maximum likelihood. | 2.44 | q1 | Yes |
| 7 | 936 | Yeah, and have you tried it? Yes, I've tried it. And there are some built in like easy voices which you can, there is a cool one completely synthetic which you cannot understand at all. And then there is a iPhone synthesis. I think it is the iPhone synthesis from the, like something like cat voice. Yeah, cat voice, like dinner, like they have so, so, but this doesn't sound very nice and we were playing at the summer school. | Festival is open It's Yeah, cat voice, like there are like they they have few Yeah, new voices like, yeah. | 2.44 | q1 | Yes |
| 8 | 462 | And | Mm-hmm. | 2.44 | q1 | Yes |
| 9 | 775 | If you have not | Okay, so we can hope that Nokia will be using tandem tandem P_L_P_s. | 2.44 | q6 | Yes |
| 10 | 770 | I doubt that at them there are trying to write no but maybe other companies. | I doubt that at Daimler they are trying right now, but maybe other companies, mm. | 2.44 | q1 | Yes |
| 11 | 520 | Yeah, first this is proposed by the hotel, like, even we read one paper in our room in group here, remember that | Yeah, first this is proposed by like uh even we read one paper in our reading group, you remember that? | 2.44 | q1 | Yes |
| 12 | 541 | I just need to. | Uh. | 2.44 | q1 | Yes |
| 13 | 358 | you | Mm-hmm. | 2.44 | q1 | Yes |
| 14 | 780 | No, no, no, I might be. No, this co-han you mean like, I don't know, he like told us that somebody's using that very actively like, okay, it warps for them and | No, no n no, this uh Cohen, you mean like uh uh Hmm. | 2.44 | q1 | Yes |
| 15 | 573 | Yeah, sure. | Yeah, sure. | 2.44 | q1 | Yes |
| 16 | 917 | It's based on the iPhone | It's based on the diphones diphones, right? | 2.44 | q9 | Yes |
| 17 | 591 | The first is difficult to show that it works for all the datas, right? Yeah. It's a game you don't know and every paper then should contain all the results from this evolution or some standard task. No, they know they only want conference in three years or something for speech. But then we all optimize on the same task. | First it's difficult to to show that it works for all all the datas, right, because it's really and Yeah. | 2.44 | q1 | Yes |
| 18 | 1028 | then it takes a lot of time | So it's somebody and Simon King is playing with that, right, the t guy. | 2.44 | q1 | Yes |
| 19 | 837 | You will notice something like a lot of Vitarovages. | Even even not Viterbi, just even something simpler. | 2.44 | q1 | Yes |
| 20 | 521 | I have implemented it. | Mm. | 2.44 | q1 | Yes |

## Sampled 8B: max over questions

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. For each question, the fraction of the 8 answers graded correct is standardized across all lines. A line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 917 | It's based on the iPhone | It's based on the diphones diphones, right? | 32.22 | q9 | Yes |
| 2 | 36 | Thank you very much. Thank you very much. | I t I Uh I think it's quite interesting, but only annoying thing is this multiple sessions. | 28.46 | q0 | Yes |
| 3 | 775 | If you have not | Okay, so we can hope that Nokia will be using tandem tandem P_L_P_s. | 23.14 | q6 | Yes |
| 4 | 531 | He was quoting... It has been used for speech coding... Nobody used it for recognition stuff I think I never heard it. But there... He was a speech coding, so would he used to use it? No, I don't know, of course. Yeah, that's good question. Maybe. Yeah, he was quoting that paper also... He was telling... This... This is kind of optimization created about this temporal decomposition is doing like... | Ne he was re he was quoting But that Yeah, he was uh quoting that paper also, like v he was telling this uh this is kind of optimisation criteria what this temporal decomposition is doing like. | 17.34 | q6 | Yes |
| 5 | 757 | Yeah | Mm-hmm. | 11.54 | q6 | Yes |
| 6 | 2 | I think the most of the bad words I had to begin with is precise time. | I think the the most of the bad words are at the beginning when the people try to put in on. | 10.64 | q0 | Yes |
| 7 | 144 | Sometimes, it will lead to my danger. If you would have to spend a hour and a half that the him and the other you spend while doing with the | Some time, yes, but not so many days or Oh yeah, yeah, that's true. | 10.64 | q0 | Yes |
| 8 | 263 | The fun PCO, the MLT. | So from Mm-hmm. | 5.74 | q6 | Yes |
| 9 | 947 | But maybe you can just take your stuff out. | But it it's not here so far. | 5.74 | q6 | Yes |
| 10 | 30 | I just would be right. | Yeah, I see see. | 2.44 | q1 | Yes |
| 11 | 70 | Yeah. | Yeah. | 2.44 | q1 | Yes |
| 12 | 80 | But | Hmm. | 2.44 | q1 | Yes |
| 13 | 89 | before. | Yeah, w we went to the Costa Caparica there, with the bus. | 2.44 | q1 | Yes |
| 14 | 207 | If you compare it with some baseline, let's say you use just GMM with that overall. No, I think they are good results, but just using that also differently... I think it was a slightly better area. And no decoration... Maybe decoration again they do DCT or the other. | But if you compare no, if you compare it with some baseline, let's say you use just G_M_M_ with diagonal covariance matrices A and no decorrelation before like uh Yeah. | 2.44 | q1 | Yes |
| 15 | 300 | 20 millies minimum | This is maybe because of F_F_T_U_ or these things. | 2.44 | q1 | Yes |
| 16 | 354 | So then what he does is like he proposed like some like loot ratio test based on action like loot. | So then what he does is like he proposed uh like some like ratio test based on maximum likelihood. | 2.44 | q1 | Yes |
| 17 | 358 | you | Mm-hmm. | 2.44 | q1 | Yes |
| 18 | 385 | then you shift to 25 and so forth and then if you the maximum is 60 millisecond frame for him okay so if you don't find anything where what | Then you shift to twenty five and so forth until you Okay, and and you find the point where where what? | 2.44 | q1 | Yes |
| 19 | 460 | By the length. Yeah. | By the length. | 2.44 | q1 | Yes |
| 20 | 462 | And | Mm-hmm. | 2.44 | q1 | Yes |

## Logp 8B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-8B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? |
|---|---|---|---|---|---|---|
| 1 | 928 | Now, mostly the festival what you get on line excumations is definitely based only. | No, most of the the Festival, uh what you get on Linux machines, is diphone based only. | 13.69 | q9 | Yes |
| 2 | 332 | So again if you change this shift the modulation spectrum, the frequency changes for each window | So again, if you change this shift, the mod the Nyquist the modulation spectrum, Nyquist frequency changes for each window. | 9.89 | q7 | No |
| 3 | 57 | I like the invite and speaker about the implant in the year. | I liked the invited speaker um about the implant in the ear. | 9.55 | q2 | No |
| 4 | 53 | How many patients did you help? Priced patients are hardly and full of five work. I like the invited speaker about the implant in the year. Oh, yeah. That's quite good. It was from Austria, right? The guy. I think so. Yeah. Yeah. He's very famous, I think for that right. He's the inventor of the implant and for the year, I think. Even some panel discussions on the human speech recognition. Reducing gap between ASR and | So how many presentation you you I realised it were that there were panel discussions only at the place where I could see the the t t you know, there were some boards saying that there is some panel discussion, uh but I I couldn't get no one knew where is it, what it is, where it happens and why and so forth. | 7.31 | q2 | No |
| 5 | 44 | So in 20 minutes you can say at least 2-3 posters and you can directly talk to people. | So in t twenty minutes you can see at least two, three posters, and you can directly talk to people, so. | 7.04 | q1 | Yes |
| 6 | 927 | because I don't know which one is running or if it's time for one or which a process is. | Because uh I don't know which which um what i what is running where uh if it's diphone or which uh approach this is. | 7.01 | q9 | Yes |
| 7 | 235 | No problem, but T-FTC is so many models and so many, and so many. | Ne no problem, but if you see so many models and so many mixtures yeah. | 6.03 | q4 | No |
| 8 | 918 | A for your phones, right? | Diphone, yeah, yeah. | 5.51 | q9 | Yes |
| 9 | 289 | as much as possible like a but they are differently to issue some the minimum and maximum sizes of your own so he is using one his own minimum is I think to around 12.5 milliseconds | Yeah, as much as possible like but yeah, definitely you have to assume some l uh like minimum and maximum sizes of your own, so he is using minimum of minimum is uh I think twe I don't twelve point five milliseconds I think. | 5.39 | q6 | Yes |
| 10 | 926 | This I was tried to install it and there are the voices like cat voice and these voices this is not no they there are so many voices | 'Cause the the diphone that I was tried to install it and there there are the voices like cat voice and these voices, this is not the one? | 5.33 | q9 | Yes |
| 11 | 930 | you can get it installed in your machine here | So it's uh just uh wha samples of the diphones, and it choo chooses Yeah, on my on my laptop. | 4.82 | q9 | Yes |
| 12 | 917 | It's based on the iPhone | It's based on the diphones diphones, right? | 4.76 | q9 | Yes |
| 13 | 4 | So you're stuck. I think right. | So you start I think, right? | 4.23 | q0 | Yes |
| 14 | 2 | I think the most of the bad words I had to begin with is precise time. | I think the the most of the bad words are at the beginning when the people try to put in on. | 4.16 | q0 | Yes |
| 15 | 942 | it is not die phones you have really large inventory of this speech. So, this is not | It's not diphones, you have really large inventory of the speech. | 3.80 | q9 | Yes |
| 16 | 68 | the inventor of that implant and talk for the year, I think. | He's the the inventor of that implant and for for the ear I think. | 3.76 | q2 | No |
| 17 | 924 | I've been very used to this game, I think the first game was a good game. | Uh Festival is good, like now it comes at all the Linux boxes also, so in so already in this No the there are so many voices, like again Which voice you want to use. | 3.66 | q9 | Yes |
| 18 | 590 | then people feel that they have to show the data. | No okay, so So it should be every papers then should contain all the results from NIST evaluation or some standard task then. | 3.62 | q6 | Yes |
| 19 | 938 | And have we tried it? Yes, I've tried and there are some built in like easy voices which you can There is a dual-one completely synthetic which you cannot understand at all When then there is a iPhone synthesis I think it is the iPhone synthesis from the like something like cat voice Yeah cat voice like they are like they have so so but this doesn't sound very nice and we were playing at the summer school with | Yeah, you can you can easily yes, I d I've tried, but there are some built-in like easy voices which you can y uh there is a default one completely synthetic, which you cannot understand at all, and th then there is a diphone synthesis I think it is the diphone synthesis from the some like something like cat w voice and somethi this is so-so, but it's this doesn't sound very nice, and we were playing at the uh summer school with them, actually with the authors, and they were playing us some other other approaches, and it it sounded like a human, it was great, but Yeah. | 3.62 | q9 | Yes |
| 20 | 330 | Well, why to keep variable variability in length of the frames? Well, why not to keep the same height? Actually the problem is again, you see the safety, the nitric frequency, if you compute modulation spectrum. Yeah. | W you want to Actually the problem's again uh uh you see the shift is the Nyquist frequency if you compared modulation spectrum. | 3.60 | q7 | No |

## Logp 32B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-32B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-32B)? |
|---|---|---|---|---|---|---|
| 1 | 332 | So again if you change this shift the modulation spectrum, the frequency changes for each window | So again, if you change this shift, the mod the Nyquist the modulation spectrum, Nyquist frequency changes for each window. | 19.47 | q7 | No |
| 2 | 928 | Now, mostly the festival what you get on line excumations is definitely based only. | No, most of the the Festival, uh what you get on Linux machines, is diphone based only. | 13.12 | q9 | No |
| 3 | 917 | It's based on the iPhone | It's based on the diphones diphones, right? | 12.59 | q9 | No |
| 4 | 15 | You can't attend multi-processing as the same thing. | Uh you can't attend multiple session at the same time, so. | 12.11 | q0 | No |
| 5 | 918 | A for your phones, right? | Diphone, yeah, yeah. | 11.79 | q9 | No |
| 6 | 340 | So again this may blow some frequency transition or you're... But the problem is again, this bottle like is here like you can't change your micrously-consume resolution. So have two questions. First, essentially how he is doing it that. | So again, this may blur some frequency transitions or yeah, but the problem is again this uh bottleneck is here, like you can't change your Nyquist frequency modulation and Yeah. | 9.97 | q7 | No |
| 7 | 36 | Thank you very much. Thank you very much. | I t I Uh I think it's quite interesting, but only annoying thing is this multiple sessions. | 9.05 | q0 | No |
| 8 | 927 | because I don't know which one is running or if it's time for one or which a process is. | Because uh I don't know which which um what i what is running where uh if it's diphone or which uh approach this is. | 8.60 | q9 | No |
| 9 | 235 | No problem, but T-FTC is so many models and so many, and so many. | Ne no problem, but if you see so many models and so many mixtures yeah. | 8.51 | q4 | No |
| 10 | 57 | I like the invite and speaker about the implant in the year. | I liked the invited speaker um about the implant in the ear. | 8.44 | q2 | No |
| 11 | 68 | the inventor of that implant and talk for the year, I think. | He's the the inventor of that implant and for for the ear I think. | 6.05 | q2 | No |
| 12 | 926 | This I was tried to install it and there are the voices like cat voice and these voices this is not no they there are so many voices | 'Cause the the diphone that I was tried to install it and there there are the voices like cat voice and these voices, this is not the one? | 5.89 | q9 | No |
| 13 | 930 | you can get it installed in your machine here | So it's uh just uh wha samples of the diphones, and it choo chooses Yeah, on my on my laptop. | 5.67 | q9 | No |
| 14 | 44 | So in 20 minutes you can say at least 2-3 posters and you can directly talk to people. | So in t twenty minutes you can see at least two, three posters, and you can directly talk to people, so. | 5.64 | q1 | No |
| 15 | 942 | it is not die phones you have really large inventory of this speech. So, this is not | It's not diphones, you have really large inventory of the speech. | 5.35 | q9 | No |
| 16 | 289 | as much as possible like a but they are differently to issue some the minimum and maximum sizes of your own so he is using one his own minimum is I think to around 12.5 milliseconds | Yeah, as much as possible like but yeah, definitely you have to assume some l uh like minimum and maximum sizes of your own, so he is using minimum of minimum is uh I think twe I don't twelve point five milliseconds I think. | 4.83 | q6 | Yes |
| 17 | 293 | You see, it's more than minimum. | So it's not so mm Two point five milliseconds. | 4.83 | q6 | Yes |
| 18 | 53 | How many patients did you help? Priced patients are hardly and full of five work. I like the invited speaker about the implant in the year. Oh, yeah. That's quite good. It was from Austria, right? The guy. I think so. Yeah. Yeah. He's very famous, I think for that right. He's the inventor of the implant and for the year, I think. Even some panel discussions on the human speech recognition. Reducing gap between ASR and | So how many presentation you you I realised it were that there were panel discussions only at the place where I could see the the t t you know, there were some boards saying that there is some panel discussion, uh but I I couldn't get no one knew where is it, what it is, where it happens and why and so forth. | 4.82 | q2 | No |
| 19 | 938 | And have we tried it? Yes, I've tried and there are some built in like easy voices which you can There is a dual-one completely synthetic which you cannot understand at all When then there is a iPhone synthesis I think it is the iPhone synthesis from the like something like cat voice Yeah cat voice like they are like they have so so but this doesn't sound very nice and we were playing at the summer school with | Yeah, you can you can easily yes, I d I've tried, but there are some built-in like easy voices which you can y uh there is a default one completely synthetic, which you cannot understand at all, and th then there is a diphone synthesis I think it is the diphone synthesis from the some like something like cat w voice and somethi this is so-so, but it's this doesn't sound very nice, and we were playing at the uh summer school with them, actually with the authors, and they were playing us some other other approaches, and it it sounded like a human, it was great, but Yeah. | 4.65 | q9 | No |
| 20 | 887 | But maybe like if you can't really recognize you will be really happy. | Yeah, but maybe like if it can't really recognise you it will be really annoying it's better. | 4.44 | q0 | No |

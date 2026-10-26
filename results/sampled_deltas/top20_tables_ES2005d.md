# ES2005d: top-20 lines per teacher metric

## Quiz questions and gold answers

| q | question | gold answer | wrong on baseline (Qwen3-8B) | wrong on baseline (Qwen3-32B) |
|---|---|---|---|---|
| q0 | What was the main reason given for flattening the remote-control shape instead of keeping the curved version? | To save two Euros. | No | No |
| q1 | Why did the team decide not to prioritize adding the extra scroll buttons to the design? | Because the team was trying to save money and did not want to spend extra if the current design was acceptable. | No | No |
| q2 | What feature was described as the main interaction method for the first prototype, with buttons kept to a minimum? | Speech recognition. | Yes | Yes |
| q3 | Why was the tomato-based prototype included even though tomatoes had not been part of the earlier discussion? | Because there was some red material left over, so they used it for a tomato design. | Yes | Yes |
| q4 | What was the purpose of the yellow element on the first prototype? | It was the slogan that needed to be incorporated. | Yes | No |
| q5 | Why were the scroll buttons placed with some hesitation on the design? | Because they could get bashed where people hold the device. | No | No |
| q6 | What was the stated reason for the big red button being placed in the middle rather than in a traditional location? | Because it was an obvious place and also out of the way. | Yes | Yes |
| q7 | What specific problem did the team identify with the fruit-shaped designs when discussing company strategy and sales? | That limiting the product to only a few fruit shapes would limit customers and therefore reduce sales and profit. | No | No |
| q8 | What final overall evaluation score did the team agree on after averaging the individual ratings? | Two. | No | No |
| q9 | What conclusion did the team reach about the project after the evaluation and budget discussion? | That they were done and could launch the product because it was within budget. | No | No |

## Sampled 8B: expected total score

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. A line's score is the expected total quiz score after clarifying it (the number of correct answers out of 10, averaged over the 8 samples), standardized across all lines. The question column is the question whose chance of being answered correctly rose the most.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? | clarification seems important? |
|---|---|---|---|---|---|---|---|
| 1 | 46 | He's not allowed to go. | What tomato? | 3.43 | q3 | Yes | Partly — sets up that tomatoes weren't discussed |
| 2 | 75 | It does also lie flat, but that's the yellow sand there represents the charging stand. | It does also lie flat, but that's the that yellow stand there represents the the charging stand. | 3.21 | q4 | Yes | No |
| 3 | 240 | I'm happy. | Are we happy? | 2.06 | q2 | Yes | No |
| 4 | 412 | But you can give your views based on technology, whether the technology meets the requirements of the customer's or not. | But you can give your views based on technology, whether the technology meets the requirements of the customers or not? | 2.06 | q4 | Yes | No |
| 5 | 207 | You just select one books. Yeah, this books. | You just select one box outsi yeah, this box. | 1.83 | q2 | Yes | No |
| 6 | 289 | That kind of unfortunately is too quick. Okay. Well... | That kind of unfortunately is too quick. | 1.83 | q2 | Yes | No |
| 7 | 369 | That's a good one. | The board working again, is it? | 1.83 | q2 | Yes | No |
| 8 | 88 | Well, if you hold it, it does actually feel quite ergonomic if you've got some hands. | Well, if you hold it, you can you all can hold it, is it does actually feel quite ergonomic, if you've got small hands. | 1.83 | q2 | Yes | No |
| 9 | 268 | Yeah, everyone got enough info I think. | Yeah, everyone got enough input, I think. | 1.83 | q2 | Yes | No |
| 10 | 387 | This is a requirement is essentially just an upper. | When the user requirement is essentially just to operate the T_V_, so yeah, of course we haven't actually got a working model yet. | 1.83 | q2 | Yes | No |
| 11 | 24 | Aw. | Oh. | 1.83 | q2 | Yes | No |
| 12 | 28 | So, I don't even have a soul of them. | I don't even have this on. | 1.83 | q3 | Yes | No |
| 13 | 281 | They don't. | Mm-hmm. | 1.83 | q2 | Yes | No |
| 14 | 475 | Keep playing wait you want us to wait them | Can you explain what you want us to write there? | 1.83 | q1 | No | No |
| 15 | 249 | I think we're very creative. | W I think we were very creative. | 1.83 | q2 | Yes | No |
| 16 | 150 | in and out. | Mm-hmm. | 1.60 | q2 | Yes | No |
| 17 | 15 | Not right, OK. OK. | Alright, okay, yeah. | 1.60 | q2 | Yes | No |
| 18 | 318 | I'm so sorry. | Sorry uh. | 1.60 | q2 | Yes | No |
| 19 | 511 | Um, see ya, love. | Well it was yellow. | 1.60 | q2 | Yes | No |
| 20 | 465 | So we can say 1.0.2. | So we can say one or two. | 1.60 | q1 | No | No |

## Sampled 8B: max over questions

For each line, the line is replaced with its ground-truth version and Qwen3-8B answers every quiz question 8 times with sampling; Qwen3-8B grades each answer right or wrong against the gold answer. For each question, the fraction of the 8 answers graded correct is standardized across all lines. A line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? | clarification seems important? |
|---|---|---|---|---|---|---|---|
| 1 | 106 | It's not in the traditional place but it's quite an uphill place. | It's not in the traditional place, but um it's quite an obvious place. | 23.09 | q6 | Yes | Yes |
| 2 | 73 | that's it, Stan. | That's it stand. | 9.12 | q4 | Yes | No |
| 3 | 74 | It does. | Alright, okay. | 9.12 | q4 | Yes | No |
| 4 | 75 | It does also lie flat, but that's the yellow sand there represents the charging stand. | It does also lie flat, but that's the that yellow stand there represents the the charging stand. | 7.96 | q4 | Yes | No |
| 5 | 279 | We discussed all the new ideas, but of course we could have reached any proper goals. We could have used this later, but we are using these scroll buttons like this. | We discussed all the new ideas, but of course we couldn't reach any proper goals, we couldn't use these, but we h we are using these scroll buttons like this. | 6.80 | q4 | Yes | No |
| 6 | 61 | All right. | Alright, okay. | 6.80 | q4 | Yes | No |
| 7 | 68 | I like the feeling. | Well I like the feel of it, I like the feel of it. | 5.64 | q4 | Yes | No |
| 8 | 46 | He's not allowed to go. | What tomato? | 5.28 | q3 | Yes | Partly — sets up that tomatoes weren't discussed |
| 9 | 412 | But you can give your views based on technology, whether the technology meets the requirements of the customer's or not. | But you can give your views based on technology, whether the technology meets the requirements of the customers or not? | 4.48 | q4 | Yes | No |
| 10 | 15 | Not right, OK. OK. | Alright, okay, yeah. | 4.48 | q4 | Yes | No |
| 11 | 332 | It's this as well, sorry we've got to mention what we made out of kind of a rubbery, they text. New material that we've got. | It's this as well, sorry, we forgot to mention it'll be made out of kind of a rubbery latex, new material that we've got. | 4.48 | q4 | Yes | No |
| 12 | 392 | 7. | Seven. | 4.48 | q4 | Yes | No |
| 13 | 188 | Yeah, you're the shift button stick. | No maybe the shift button's stuck in. | 4.48 | q4 | Yes | No |
| 14 | 171 | I don't look like that character's a curve, I think that's just a sheep. | I don't think that counts as a curve, I think that's just a shape. | 4.48 | q4 | Yes | No |
| 15 | 37 | Okay. | Yeah. | 4.48 | q4 | Yes | No |
| 16 | 422 | For me personally it is everything fine it may be having good design it may be a meeting all the requirements of the customers like technology wise price wise but there is one thing which limits the customers like we are having only two three designs like we are having one banana design and the other one is orange and yellow orange | Uh for me personally it is everything fine, it may be having good design, it may be uh meeting all the requirements of the customers like technology-wise, price-wise, but there is one thing which limits the customers, like we are having only two, three designs, like we are having one banana design and the other one is orange, and Uh yeah, lower end. | 3.72 | q3 | Yes | No |
| 17 | 294 | to- | Uh. | 3.32 | q4 | Yes | No |
| 18 | 58 | Okay so the buttons would be like, you know, individual users, yeah. | Okay, so the buttons would be like, you know individual users, or Yeah. | 3.32 | q4 | Yes | No |
| 19 | 28 | So, I don't even have a soul of them. | I don't even have this on. | 2.94 | q3 | Yes | No |
| 20 | 431 | So we are showing our preference for particular fruits, 2 or 3 kinds of lab and fruit is. | So we are showing our preference for particular fruits, two or three kinds rather, and Yeah. | 2.94 | q3 | Yes | No |

## Logp 8B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-8B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-8B)? | clarification seems important? |
|---|---|---|---|---|---|---|---|
| 1 | 144 | So basically, in order to save our two euros, I was thinking that we could have a sense of the same shape but just have it flattened. | So basically, um in order to save our two Euros um I was thinking that we could have essentially the same shape, but just have it flattened. | 10.01 | q0 | No | No — ASR already says it |
| 2 | 231 | for maybe we could have something but maybe if what is that said that's that's that's that's a thing that we've got. | Well maybe we could add something, but maybe if What do you th We're trying to save money, so. | 7.70 | q1 | No | Yes |
| 3 | 232 | if we're happy with the design as they put 20 money if we don't have to | Yeah, if we're happy with the design there's no point in spending money, if we don't have to. | 4.59 | q1 | No | Yes |
| 4 | 56 | If you do need buttons, you can flip it over on the some there. But mainly is speak tracking notion. | If you do need buttons, you can flip it over, and there's some there, um but mainly it's speech recognition. | 4.23 | q2 | Yes | Yes |
| 5 | 109 | It's kind of real thanks. | It's out of the way as well, I suppose, so. | 4.09 | q6 | Yes | Yes |
| 6 | 13 | We... | Um Um we Yeah. | 4.04 | q6 | Yes | No |
| 7 | 73 | that's it, Stan. | That's it stand. | 3.74 | q4 | Yes | No |
| 8 | 305 | So is it go back in? No, it's not. It's really, really annoying. Have you finished? | So So Uh Brian, have you have you finished? | 3.57 | q7 | No | No |
| 9 | 74 | It does. | Alright, okay. | 3.44 | q4 | Yes | No |
| 10 | 538 | So we can say 211, 22. | So we can say two or one Okay. | 3.21 | q9 | No | No |
| 11 | 419 | You'll go far back? Basically, requirements. | You will go for one. | 3.18 | q2 | Yes | No |
| 12 | 12 | Thank you for watching. | Hmm. | 3.13 | q9 | No | No |
| 13 | 111 | So there we go. And we have the banana. Oh yeah. At least one too. | So there we go and and um we have the banana-based one too. | 3.03 | q9 | No | No |
| 14 | 556 | Yep. | Yeah. | 2.98 | q7 | No | No |
| 15 | 233 | So if we happy with Zion as name, if we don't have to | So Yeah. | 2.98 | q7 | No | No |
| 16 | 241 | Don't worry. | Oh. | 2.97 | q6 | Yes | No |
| 17 | 248 | room for creativity when we happy with that. | Uh room for creativity, were we happy with that? | 2.94 | q7 | No | No |
| 18 | 266 | Yeah. | Yeah, yeah. | 2.92 | q8 | No | No |
| 19 | 229 | Right, so that's finance. I don't know what we can do it, we can we could add. | Right, so that's finances and I dunno what we what could we reckon we could add? | 2.77 | q2 | Yes | No |
| 20 | 437 | some children like to buy banana chips, some apple, some even pineapple chips, some orange chips | Some children like to buy banana shape, some apple shape, some even pineapple shape, some orange shape. | 2.62 | q6 | Yes | No |

## Logp 32B: max over questions

For each line, the line is replaced with its ground-truth version and we measure how likely Qwen3-32B finds the exact gold answer to each question (the log-probability of the gold answer text). Each question's likelihoods are standardized across all lines, a line's score is its highest standardized value over the questions, and the question column is the question that value comes from.

| rank | line | ASR text | GT text | z | question | question wrong on baseline (Qwen3-32B)? | clarification seems important? |
|---|---|---|---|---|---|---|---|
| 1 | 85 | when we weren't sure about putting them there because it kind of could get bothered. | We n were weren't sure about putting them there, because um i it's it kind of could get bashed. | 13.84 | q5 | No | Yes |
| 2 | 109 | It's kind of real thanks. | It's out of the way as well, I suppose, so. | 12.28 | q6 | Yes | Yes |
| 3 | 144 | So basically, in order to save our two euros, I was thinking that we could have a sense of the same shape but just have it flattened. | So basically, um in order to save our two Euros um I was thinking that we could have essentially the same shape, but just have it flattened. | 10.34 | q0 | No | No — ASR already says it |
| 4 | 46 | He's not allowed to go. | What tomato? | 8.89 | q3 | Yes | Partly — sets up that tomatoes weren't discussed |
| 5 | 106 | It's not in the traditional place but it's quite an uphill place. | It's not in the traditional place, but um it's quite an obvious place. | 8.51 | q6 | Yes | Yes |
| 6 | 47 | No, no, no. | I don't recall a tomato. | 7.88 | q3 | Yes | Partly — sets up that tomatoes weren't discussed |
| 7 | 232 | if we're happy with the design as they put 20 money if we don't have to | Yeah, if we're happy with the design there's no point in spending money, if we don't have to. | 7.55 | q1 | No | Yes |
| 8 | 231 | for maybe we could have something but maybe if what is that said that's that's that's that's a thing that we've got. | Well maybe we could add something, but maybe if What do you th We're trying to save money, so. | 7.45 | q1 | No | Yes |
| 9 | 45 | the other one. | Tomato? | 7.12 | q3 | Yes | Partly — sets up that tomatoes weren't discussed |
| 10 | 42 | Thank you. | Three? | 4.94 | q3 | Yes | No |
| 11 | 48 | Yeah, well, yeah, we had some red left over. | Oh yeah, well yeah, we had v some red left over. | 4.56 | q3 | Yes | No — ASR already says it |
| 12 | 53 | The yellow there is the slogan that we need to incorporate very simple. | The yellow there is the um the slogan, yeah, that we need to incorporate, it's very simple. | 4.43 | q4 | No | No — ASR already says it |
| 13 | 56 | If you do need buttons, you can flip it over on the some there. But mainly is speak tracking notion. | If you do need buttons, you can flip it over, and there's some there, um but mainly it's speech recognition. | 4.26 | q2 | Yes | Yes |
| 14 | 134 | Unfortunately the unit we are currently going to produce minus the extra scroll button and it's going to cost us 14.60 euros. | Um unfortunately the unit we are currently going to produce minus the extra scroll buttons, uh it's gonna cost us fourteen point six Euros. | 4.02 | q0 | No | No |
| 15 | 238 | We have under 12 years, 50. | We have under twelve Euros fifty. | 3.60 | q0 | No | No |
| 16 | 3 | or else I'll use the PowerPoint I guess. | We'll sta I'll use the PowerPoint, I guess. | 3.59 | q3 | Yes | No |
| 17 | 468 | because it is 1.8 just using 2 so we will do 2.10. | because it is one point eight uh two, so we will do two. | 3.45 | q8 | No | Yes |
| 18 | 342 | Oh, can you pass the tumulties? | Oh could you pass the tomato please. | 3.39 | q3 | Yes | No |
| 19 | 424 | the orange and the third one is what you get, that is not a fruit look, but if a person doesn't like banana or orange you are limiting him, | Yeah th Come on that was the tha 'Kay. | 3.32 | q7 | No | No — GT line drops the relevant content |
| 20 | 129 | So, are there any improvements or issues? Yes, there are issues. What are the issues, and just let it lie down, I want to stand. A main problem that we have unfortunately being a finance. | So are there any um improvements or issues or Oh. | 3.31 | q1 | No | No |

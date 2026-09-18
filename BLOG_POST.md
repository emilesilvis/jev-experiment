# Trying Jev on a small classification benchmark

I wanted to see how Jev compares with a small LLM at putting text into the right category. When the answer goes into software, there are two things to get right: choosing the label and returning something the application can use.

A typical generative LLM builds its answer one token at a time. Jev takes a different approach: you provide the possible answers, and it returns a decision with probabilities. [TypeSafe says](https://docs.typesafe.ai/introduction) it can evaluate independent questions in parallel. Its underlying neural architecture isn't fully disclosed, so I'm describing its documented behaviour here.

I compared Jev 1.13.0 with GPT-4o mini's July 2024 version on 1,250 texts. Both received the same inputs and category descriptions, with no worked examples. The tasks covered news topics, question types, and user intentions.

Before the numbers, one detail matters. An answer counted as correct only if it chose the right category and returned probabilities that added up to one, within a small rounding tolerance.

| Task | Jev accuracy | GPT-4o mini accuracy |
|---|---:|---:|
| News topics | 86.8% | 77.0% |
| Question types | 91.6% | 77.0% |
| User intentions | 96.9% | 90.0% |

Giving each task equal weight, that's **91.7% for Jev and 81.3% for GPT**. Jev's median request time was also 36–42% lower, and its estimated token cost was about 65% lower at the rates used in the experiment. Each model's total was under ten US cents. Those timing and cost figures are for this setup, with both models returning probabilities.

Now look at that probability requirement. GPT failed that check 24 times; Jev failed it once. Those responses counted as incorrect.

On user intentions, keeping only the 327 examples where both returned valid answers reduced the gap to **97.2% versus 96.3%**. Much closer. That selected subset doesn't replace the main result, but it shows how much the recorded gap shrinks when invalid responses are excluded. Jev's larger leads on news and question types remained.

One example makes the question-type task concrete: “How far is it from Denver to Aspen?” GPT classified it as asking for a location. Jev classified it as asking for a number. The sentence mentions two places, but the answer is a distance.

What's most interesting to me is that the application's rules are part of the comparison. Here, returning a usable probability distribution was part of the job.

This was a small test against one older LLM version, using public data that either model may have seen during training. Next, I'd try fresh examples and let GPT return just a category, then compare accuracy, time, and cost again. That would help separate the quality of the decision from the difficulty of expressing it as probabilities.

*The [full report](REPORT.md) includes the method, saved results, and limitations.*

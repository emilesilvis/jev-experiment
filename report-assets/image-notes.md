# Jev comparison image

File: [jev-vs-generative-llm.png](jev-vs-generative-llm.png)

Created on 18 September 2026 with the built-in image generation tool. Final size: 1536 × 1024 pixels.

## Suggested caption

A typical generative LLM produces its answer as a sequence of tokens, including when it produces JSON. Jev instead exposes typed questions and returns structured decisions with probabilities; TypeSafe says the questions are evaluated independently in parallel. This is a conceptual view of the input/output flow, not a diagram of Jev's undisclosed neural architecture.

## Alt text

Two side-by-side flowcharts compare a typical generative LLM with Jev. The LLM takes text and instructions, passes them through a generative model and a sequential next-token loop, produces text or JSON, then parses and checks the result before the application uses it. Jev takes text, typed questions and allowed answers, evaluates independent questions in parallel, and returns choices with probabilities to the application. A footer says that this is a conceptual view and Jev's internals are not fully disclosed.

## Sources and scope

- [TypeSafe introduction](https://docs.typesafe.ai/introduction): typed questions, structured values and probabilities, and independent parallel evaluation. The image illustrates Choice questions; Jev also exposes Score and Noul questions.
- [TypeSafe launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev): TypeSafe's description of Jev's parallel sampler and how it differs from autoregressive text generation. Jev-specific behavior is attributed to the vendor, not independently established by this illustration.
- [Hugging Face text-generation guide](https://huggingface.co/docs/transformers/llm_tutorial): next-token generation conditioned on the input and previously generated output.

Sources checked on 18 September 2026. The probability bars are symbolic icons, not measured results. “Parse and check” illustrates a common application flow; SDKs and structured-output features can handle part of that work. Typed output does not guarantee a correct judgment. No speed, cost, calibration, accuracy, or internal-layer claims are encoded in this image.

## Visual check

Passed a visual inspection of the final image for legibility, spelling, arrow direction, intended parallel branches, and the required architecture caveat. A first draft had a dark vignette that reduced title/footer contrast; the final edit corrected it. The raster asset is suitable for embedding in a blog post. It is a simplified conceptual diagram and does not disclose Jev's neural architecture.

## Exact generation prompt

```text
Use case: infographic-diagram
Asset type: polished landscape blog infographic for non-specialists, 3:2 landscape canvas, high resolution.
Primary request: Clearly explain Jev versus a typical generative large language model. This is a conceptual comparison of how inputs become useful outputs, not a disclosed neural network architecture.
Style: precise editorial infographic, spacious white/off-white background, crisp dark charcoal sans-serif typography, restrained blue accent for generative LLM and teal accent for Jev. Beautiful but sober; thin consistent arrows and tidy rounded rectangular cards. Large readable text. No gradients, mascots, brains, robots, photographs, logos, or ornament.
Composition: centered title above two equally weighted side-by-side panels, each running from top to bottom. Left panel title "Typical generative LLM", right panel title "Jev". Use exact wording below, all text clearly readable and correctly spelled. Align the input rows and final application rows. The LLM flow can have more steps; keep the entire graphic balanced.
Title verbatim: "Jev vs a typical generative LLM"
Subtitle verbatim: "From information to an answer your software can use"
LEFT panel exact flow with directional arrows:
1. "Text + instructions"
2. "Generative model"
3. "Sequential next-token loop" — within this area show a small row of three token tiles connected left to right and a neat loop arrow. Tiles can be labelled "token 1", "token 2", "token 3". Do not use concrete example answer text.
4. "Text or JSON"
5. "Parse and check"
6. "Application"
At the bottom of the left panel, explanatory note verbatim: "Structured-output LLMs still generate tokens."
RIGHT panel exact flow with directional arrows:
1. "Text + typed questions + allowed answers"
2. "Questions evaluated independently in parallel" — within this area show three same-level question cards "Q1", "Q2", "Q3" connected from the same input and to the same output. Their parallel layout is a conceptual API flow, not claimed neural layers. Do not connect Q1 to Q2 or Q2 to Q3.
3. "Choices + probabilities" — modest check-list and probability-bars pictogram without numbers or statistical claims.
4. "Application"
At the bottom of the right panel, explanatory note verbatim: "Structured decisions for software."
Bottom full-width footer in dark readable text, prominent enough to notice: "Conceptual view. Jev internals are not fully disclosed."
Tiny source line under footer: "Sources: TypeSafe documentation and launch post; Hugging Face text-generation guide."
Constraints: reproduce the specified text verbatim. The image must distinguish sequential output token generation from Jev's documented independent parallel questions. Do not invent encoder layers, decoder heads, network topology, parameter counts, one-forward-pass claims, speed multipliers, benchmark data, reliability guarantees, zero-error claims, or training details. Do not imply that LLMs cannot produce structured output. Do not imply that all LLMs share a single training method. Keep text inside all boxes with comfortable margins. Aim for strong clarity at blog viewing size.
```

## Exact final edit prompt

The first generated image was the edit target.

```text
Edit this infographic. Make exactly one visual correction: remove the entire dark blurred vignette/glow/gradient backdrop and all panel gradients. Replace every background region with a clean flat solid white or very pale off-white background, including the complete area behind the title and both bottom footer lines. Use flat pale blue and pale teal only as subtle card fills. All title, subtitle, panel, and footer text must be dark charcoal on light background, with clearly legible high contrast. In particular the full title "Jev vs a typical generative LLM" and footer "Conceptual view. Jev internals are not fully disclosed." must be instantly readable. Preserve every word, flow arrow, card, alignment, layout, and illustration from the input image. No new text or shapes. No shadows, no gradients, no vignette, no texture, no glow anywhere.
```

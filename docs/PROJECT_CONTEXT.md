# Project context and design rationale

This note summarises the rationale behind the experiment. It is based on the
research discussion exported as `Deepseekchat.pdf` (a long, image-only PDF) and on
the implementation in `codes/`. The PDF itself is not committed because it is
about 91 MB and is not needed to run the code.

## Research question

The project studies discourse quality on Bluesky at a much finer granularity than
a single "quality" score. Five LLM labellers score every post on **56
discourse-quality dimensions**, grouped into reasoning, conflict, epistemic
stance, interaction and democratic preconditions.

The central modelling question is not only *how well can we predict these
scores?* but also:

> Can a single model emulate all five labellers, or do disagreements between
> labellers force us to keep separate models?

## Recommended experiment ladder

The design follows a deliberate progression, with each step answering a
different question:

1. **Data audit first** — variable distributions, labeller distributions,
   inter-labeller agreement (Krippendorff's α), author statistics, topic
   clustering, length/context statistics and correlation structure. This avoids
   wasting GPU time on models that cannot learn from the label distribution.
2. **Classical baselines** — training mean, Bag-of-Words + Ridge and TF-IDF +
   Ridge. These establish how much signal is available from lexical information
   alone.
3. **Task A: independent models** — one DeBERTa-v3-large regressor per labeller.
   This measures how well each labeller can be emulated when the model is
   allowed to specialise.
4. **Task C: unified model** — one shared encoder with five heads, then PLE /
   mixture-of-experts routing. This tests whether sharing representations across
   labellers helps or hurts.

## Splitting strategy

The primary split is **author-disjoint** (80/10/10 by author DID). Splitting by
author prevents the same author's style and topics from leaking between train and
test. A second **temporal** split (80/10/10 by time) is constructed as a
robustness check. Splits are not stratified by labeller scores, because at
inference time the true score distribution is unknown and stratification would
be optimistic.

## Metrics

The models are evaluated with:

- per-dimension MSE / MAE,
- per-dimension Pearson correlation,
- per-dimension R²,
- mean R² across the 56 dimensions,
- percentage of predictions within 0.25 of the true score.

The headline table reports mean R² across dimensions for each labeller.

## What the experiment found

- **Lexical vs contextual signal.** TF-IDF is a strong baseline for some
  labellers, but DeBERTa-v3-large is substantially better. Discourse quality is
  not just a bag-of-words property.
- **The seesaw / negative-transfer effect.** The unified PLE model improves the
  hardest labeller (`qwen3.5`) and slightly improves `mistral3.2`, but it
  degrades `gemma4`, `gptoss` and `granite4.2` relative to independent models.
  Sharing capacity helps the noisy task but harms labellers that were already
  easier to fit.
- **Dynamic loss weights help but do not remove the conflict.** Weighting the
  tasks changes the balance, but the underlying tension between labellers
  remains.
- **Gate analysis explains the pattern.** Labellers whose scores are harder to
  predict rely mostly on task-specific experts (`gptoss` 95.5%, `gemma4` 85.3%),
  while others use the shared experts much more (`granite4.2` 68.4% shared,
  `mistral3.2` 45.1% shared). This is direct evidence that "one model for all
  labellers" is not automatically the right trade-off.
- **Low agreement is the root difficulty.** The mean Krippendorff's α is about
  0.25, and many dimensions are heavily zero-inflated. Even a perfect model of
  one labeller is not necessarily a good model of another.

## Implications for future work

- Treat labeller identity as a first-class modelling decision, not a nuisance
  variable.
- Explore conditional computation (routing, adapters, per-labeller prompts) rather
  than a single shared head for all labellers.
- Report per-dimension and per-labeller results; aggregate R² hides the
  seesaw effect.
- Consider ordinal or zero-inflated losses, since a large fraction of the targets
  are exactly zero and the score grid is coarse (0, 0.25, 0.50, 0.75, 1.00).

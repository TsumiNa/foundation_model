# Result interpretation for research slides

Read this reference for model comparisons, relative metrics, error distributions,
heatmaps, learning curves, or prediction bands. Use the actual experiment's definitions;
the formulas here are examples, not a prescription to change its metrics.

## Define the evaluation unit and aggregation order

State the independent scientific units: compounds, specimens, subjects, or experiments.
Repeated temperatures, checkpoints, and predictions do not multiply that sample size.
For grouped evaluation, specify whether the held-out unit is a compound, curve, or row.
Identify leakage restrictions and whether the candidate sees additional conditions.

For a comparison with multiple pretrained checkpoints, an aggregation might be:

1. Compute each model's error on each held-out task.
2. Take the median error over checkpoints within each task.
3. Average these task errors with equal weight, or with another explicitly stated weight.

Write this order once in the methods or metric description. Use it consistently.
The median error of several models generally differs from the error of their median
prediction. Compute the intended quantity directly. A baseline with one fit per task
has no across-seed spread; do not fabricate one or silently change its replication.

## Explain standardized and relative errors

An ordinary RMSE is

\[
E=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat y_i)^2}.
\]

Identify whether the inputs are physical values or transformed targets. Physical RMSE
has the target's units. Standardized RMSE is dimensionless and depends on the specified
training-data transformation. A short definition is sufficient when the audience does
not need to know the normalization implementation.

"Normalized RMSE" has several conventions. Name the denominator explicitly, for example

\[
E_{\mathrm{RMS}}=
\frac{\sqrt{N^{-1}\sum_i(y_i-\widehat y_i)^2}}
     {\sqrt{N^{-1}\sum_i y_i^2}}.
\]

This denominator differs from the target range, its standard deviation, or its mean.
The acronym alone does not identify the metric. For a zero denominator, report the
value as undefined or use the experiment's documented treatment. Do not invent an
epsilon or change the metric during report production.

For a lower-is-better error, one relative improvement convention is

\[
I=100\,\frac{\overline E_{\mathrm{reference}}-
                 \overline E_{\mathrm{candidate}}}
                {\overline E_{\mathrm{reference}}}\%.
\]

State the reference method and that positive values favor the candidate. Define the
population and how the barred errors were computed. This is a percentage reduction of
aggregate errors, not the mean of individual task percentages. Those two quantities
can differ substantially. Use a different sign convention only if it is explicit and
consistent throughout the deck. Do not show a percentage against a zero reference.

Define relative values before their first use, preferably on a short metrics slide or
in a readable nearby caption. Preserve the denominator and reference in later captions
when readers might otherwise mistake a comparison against direct FT for one against scratch.

## Heatmaps: local effects and overall comparisons

A useful signed task-level effect is

\[
\Delta E_j=E_{\mathrm{reference},j}-E_{\mathrm{candidate},j}.
\]

Use a diverging scale centered at zero and the same color range across compared methods.
Label positive and negative directions in words. Identify the unit, metric, and any
checkpoint aggregation. Name the chart **effects** when it includes both signs.
Distinguish missing cells from zero effect. If values exceed the color limits, retain
their numerical labels and disclose color saturation.

The number of positive cells measures how often the candidate wins. It does not measure
the mean effect. For example, effects of `+0.10, +0.10, +0.10, -0.80` give three improving
tasks out of four but a mean effect of `-0.125`. The candidate wins more tasks while its
mean error is higher. This is a synthetic illustration, not an experimental result.

When reporting both views:

- Show win counts alongside aggregate errors when useful.
- Investigate whether large degradations dominate the arithmetic mean.
- State when a robust median and a mean favor different methods.
- Inspect scientifically meaningful condition groups, such as pressure or temperature,
  and report their averages without substituting them for the overall population.

Do not equate a lower median error across tasks with a higher paired win rate. These
are different summaries. A statement that a method is "best" must name the metric,
aggregation, evaluation population, and experiment setting that support it.

## Dispersion and uncertainty

Define every band, whisker, and error bar in a caption or legend:

| Quantity | Appropriate meaning |
| --- | --- |
| Between-task SD | Heterogeneity across tasks in the stated population |
| Across-checkpoint SD or IQR | Sensitivity to pretrained initialization |
| Q1–Q3 | Interquartile range of the stated set of values |
| Quartile deviation | Half the interquartile range, \((Q_3-Q_1)/2\) |
| Confidence interval | Uncertainty in an estimated statistic under the stated sampling method |
| Prediction interval | A predictive uncertainty interval with a stated construction and calibration status |

Mean with SD and median with IQR or quartile deviation are different summaries. Specify
the SD convention if it matters to reproducibility. For box plots identify the median,
quartiles, whisker rule, and any extra mean marker.

Checkpoint quantiles are descriptive ensemble dispersion unless calibration was evaluated.
For a small number of materials, many checkpoint pairs remain repeated observations of
those same materials. Resampling for population-level inference must respect that grouping.
Under a nonlinear inverse transform, transform each prediction first and then calculate
physical-unit quantiles or SD. Multiplying SD by an arbitrary scale is not generally valid.

## Matched comparisons, preprocessing, and difficult regimes

Preserve paired-error plots when they identify which matched cases improve or degrade.
Label both axes with their method and metric, include the equality line, and explain
which side favors which method. Define what is matched: held-out unit, condition,
pretrained checkpoint, seed, training budget, and data variant, as applicable.

Report additional data or training consumed by a candidate. Continued pretraining on
other conditions can provide a useful pipeline improvement while also using more data
or compute; do not attribute all of that gain to initialization alone.

For raw-versus-smoothed comparisons, show the intervention and matching explicitly.
Record the smoothing method, window in meaningful physical coordinates, which data
were smoothed, and the evaluation target. Keep the comparison fair under the existing
evaluation protocol. Never alter a holdout target silently to improve reported results.

Describe a negligible aggregate accuracy change as "no practically meaningful aggregate
improvement under the tested protocol." Do not infer statistical equivalence, no change
in individual curves, or a conclusion about every smoothing method.

For a low-temperature analysis, define the temperature interval using observed coverage.
Do not describe observations at an unmeasured zero-temperature limit. Show displaced
transitions, amplitude errors, unexpected minima, or negative physical predictions when
they occur. Distinguish negative standardized targets from negative physical values.
Do not silently clip implausible predictions or infer that raw measurement fluctuations
caused a model artifact without a supporting intervention.

## Scaling law analysis and conclusions

Use **Scaling law analysis** as the preferred section title for training-size experiments.
Name the horizontal axis precisely: training compounds, observations, or another actual
resource. If no parametric law was fitted, explain that these are empirical learning
curves and no fitted power law is claimed. State the held-out population and checkpoint
aggregation at each size. Do not promise a monotonic transfer advantage as data decrease
when the measured curves do not establish it.

The conclusion should give the strongest supported overall result first, followed by
genuine advantages under their conditions. Identify the metric and reference for each
advantage. Mention negative transfer, sensitivity to initialization, or limited coverage
when those affect use. Exploratory subgroup advantages should remain exploratory.
Claims about significance or equivalence require the relevant analysis.

# Quantitative interpretation

Read this reference when presenting quantitative comparisons, grouped summaries,
relative values, heatmaps, or uncertainty. It does not prescribe presentation sections
or a domain-specific analysis. Use the source material's definitions; these examples
are reporting conventions, not permission to change the underlying calculations.

## Quantities, populations, and aggregation

Identify the quantity, units, independent sample unit, and population being summarized.
Repeated readings of the same case do not automatically increase the independent sample
size. State exclusions or differences in coverage that materially affect a comparison.

For grouped data, explain the aggregation order and weighting once. For example, taking
the median of repeat measurements within each case and then averaging cases equally
differs from pooling all readings. A median of individual scores can also differ from
a score calculated after combining the underlying observations. Compute the intended
summary directly. Do not invent replication or spread where none was measured.

Name the measure and whether higher or lower is better. Identify transformed or
dimensionless quantities and what the transformation means. "Normalized" alone is
ambiguous: state its denominator or reference briefly. An implementation tutorial is
unnecessary unless the audience needs it.

## Relative values and reference definitions

Define a relative quantity before its first use, with the reference, denominator, sign,
and aggregation. For a nonnegative, lower-is-better quantity with a positive reference,
a percentage reduction may be

\[
I=100\,\frac{\overline q_{\mathrm{reference}}-
                 \overline q_{\mathrm{candidate}}}
                {\overline q_{\mathrm{reference}}}\%.
\]

Positive values favor the candidate under this convention. State how the barred values
were computed. This is a reduction of aggregate values, which can differ from the mean
of individual percentages. A percentage against a zero reference is undefined; use the
source analysis's documented treatment rather than inventing an epsilon. For a different
measure or sign convention, define it explicitly and use it consistently.

Put the definition on a short comparison slide or in a readable nearby caption. Repeat
the reference in later figures when readers might otherwise assume a different baseline.

## Detailed effects versus aggregate results

For a signed difference such as

\[
\Delta q_j=q_{\mathrm{reference},j}-q_{\mathrm{candidate},j},
\]

identify the case index, quantity, units, and favorable direction. Use a diverging color
scale centered at zero and matching ranges across comparable heatmaps. Distinguish
missing entries from zero. Retain numerical values or otherwise identify values beyond
the color limits. Use neutral titles such as "effects" when both signs occur.

The number of favorable cases measures frequency; the mean difference measures
magnitude. Synthetic differences of `+0.10, +0.10, +0.10, -0.80` have three positive
cases but a negative mean of `-0.125`. A candidate can improve most cases while worsening
the aggregate result. Mark illustrative numbers as examples, not observed results.

- Show favorable-case counts alongside aggregate values when both views matter.
- Explain when a few large changes dominate the mean or when mean and median disagree.
- Report meaningful subgroup results without substituting them for the full population.
- Keep matched-pair plots when they reveal heterogeneous effects; label axes, matching,
  and which side of the equality line favors each alternative.

A "best" claim needs a measure, aggregation, population, and comparison setting.
Localized benefits do not establish universal superiority.

## Dispersion and uncertainty

Define every band, whisker, and error bar in its caption or legend.

| Quantity | Meaning to state |
| --- | --- |
| Across-case SD | Variation across cases in the specified population |
| Across-repeat SD or IQR | Variation across the stated repeat observations |
| Q1–Q3 | Interquartile range of the stated values |
| Quartile deviation | Half the interquartile range, \((Q_3-Q_1)/2\) |
| Confidence interval | Uncertainty in an estimate under the stated sampling method |
| Prediction interval | Interval for an observation with a stated construction and calibration status |

Mean with SD and median with IQR or quartile deviation are different summaries. Define
box-plot medians, quartiles, whiskers, and additional mean markers. Use appropriate
grouping in any inference; repeated observations are not independent cases by default.
Descriptive dispersion is not automatically a calibrated uncertainty interval. Under a
nonlinear conversion, convert individual values before recomputing physical-unit
quantiles or SD; an arbitrary multiplicative rescaling is generally invalid.

## Fair comparisons and conclusions

Identify what is matched and any unequal inputs, resources, conditions, or coverage.
Show the intervention and evaluation basis when comparing a processing change. Keep
observations within their actual measured range and distinguish physical values from
transformed values. Do not silently clip inconvenient outcomes or attribute a feature
to a cause without supporting evidence.

Describe little visible change as no practically meaningful improvement under the tested
conditions. Statistical equivalence or significance requires the relevant analysis.
Distinguish observed trends from a fitted law and empirical associations from causal
effects. Use terminology appropriate to the actual analysis rather than prescribing a
section title from another project.

Lead conclusions with the strongest supported overall finding, then identify benefits
under their conditions. Keep exploratory findings qualified and retain limitations
that change the interpretation. A request to identify advantages does not justify
overstating the evidence.

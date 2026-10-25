# lowcov-allele-frequency

Per-population ALT allele frequencies and collapsed-paralog calls for 12,000 candidate SNPs from a simulated two-population cohort of 240 individuals sequenced at 2x to 4x.

## Difficulty

Population geneticists who process low-pass cohorts do this before anything downstream can start. At 2x to 4x an individual shows one or two reads at most sites, so hard genotype calls are out and the estimate has to come from genotype likelihoods under a per-population Hardy-Weinberg prior, the setting these data imitate (Korneliussen et al. 2014, https://doi.org/10.1186/s12859-014-0356-4).

The miscall rate is not one number. It runs from 0.002 to 0.020 across three batches, and the batch mix differs between the two populations, so an uncorrected estimate is biased more in one population than the other. About one candidate in eight carries no ALT allele at all, so the bias shows up as phantom frequencies on a large block of sites.

Rare alleles are where thresholded genotypes fail worst. A site with a true minor allele frequency of 0.03 has about seven carrier haplotypes in 240, and at this depth most of them are invisible to a caller.

Collapsed paralogs look like variants. Half of them differ from the reference copy by a fixed substitution and show a uniform read fraction across all individuals. The other half segregate in the second copy and look like ordinary polymorphisms with a depth surplus. Site depth varies by a factor with a coefficient of variation of 0.35 among single-copy sites, so depth alone reaches a recall of about 0.57 at a precision of 0.45. A solution has to combine excess heterozygote-like signal, narrow read-fraction spread and depth, and it has to pay for every false positive, because a single-copy site labelled as a paralog also fails the frequency check.

Males are hemizygous outside the pseudo-autosomal region of chrX, which halves their depth and changes the haplotype count. The ALT allele is the major allele at half the sites, so a minor-allele-folded answer fails.

The instruction fixes the deliverable and the conventions. It leaves the science open. Nothing says the error rate varies, that batches exist for a reason, or what a paralog looks like in these reads. The agent has to look. ALT read counts at sites with almost no ALT signal, split by batch, show three different floors. At fixed-difference paralogs every individual shows about the same ALT read fraction, somewhere between 0.29 and 0.50, and segregating paralogs have no homozygous ALT class at all. Male depth on chrX drops by half past 800 kb. Each observation changes the estimator. Then the agent has to decide how hard to flag paralogs, knowing that every false call costs a frequency. Each probe in calibration.md that skips one of these observations fails at least one check on every seed.

Expert time is estimated at six hours: an hour to understand the files, three to build and debug an error-aware likelihood estimator, two to get the paralog call to an acceptable recall without wrecking precision.

## Reference solution

solve.py reads the VCF and the sample table and builds per-sample reference and alternate read-count matrices. Per-site depth is divided by the autosomal mean depth of each sample, with males on non-pseudo-autosomal chrX counted at half ploidy, and scaled to a median of 1 over autosomal sites. That ratio is the depth ratio used below.

The first pass fits one miscall rate per batch on autosomal sites with a depth ratio of 1.6 or less. Fitting is by expectation-maximisation on the marginal likelihood of the read counts. The genotype of each individual is summed out under Hardy-Weinberg proportions in its population, and the ALT frequency of each site in each population is carried along as a latent quantity. Males on non-pseudo-autosomal chrX get a haploid genotype with one haplotype.

Frequencies are then refitted at every site with those rates. The paralog statistic is the gain in log-likelihood when each genotype class is allowed its own ALT read fraction, fitted separately within each population, over the constrained fit with read fractions at the miscall rate, one half, and one minus the miscall rate. A free class soaks up a uniform read fraction across all individuals and the distorted mixtures from a segregating second copy. A gain of 25 or more flags a site. A gain of 8 or more flags it when the depth ratio is 1.5 or higher.

The second pass refits the batch rates on autosomal sites that survived the first flagging, and frequencies are refitted from the first-pass values for 60 more iterations. The paralog statistic is recomputed with the new rates and the same cut-offs. Flagged sites get collapsed_paralog and NA in both frequency columns. Everything else gets single_copy and six decimals.

All cut-offs were fitted on three development cohorts (seeds 11, 12 and 13) that are not the shipped data. On the shipped cohort the run takes about 85 seconds and 610 MB of peak memory in one process and gives byte-identical output on repeat runs. It reads nothing under /tests and needs no network.

## Verification

The verifier has sixteen cases in twelve test functions. Contract checks come first and are cheap: the file exists, the header is exactly chrom, pos, locus_class, af_P1, af_P2, the 12,000 keys match the candidate list once each, locus_class is one of two strings, and frequency cells are either the literal NA on paralog rows or a decimal between 0 and 1 on single_copy rows. Scientific checks follow. They are paralog recall and precision, and for each population the share of truly single_copy sites that fall inside tolerance within four strata of the true minor allele frequency: exactly zero, above zero and below 0.10, 0.10 or higher, and the last group on non-pseudo-autosomal chrX. Reward is 1.0 only if every case passes.

Ground truth is recorded by the generator at the moment the genotypes are drawn (seed 7301). The reference solution contributes nothing to it. The truth is the realised ALT frequency among the 120 sampled individuals of each population, not the population parameter, and the instruction says so, because the two differ by about 0.03 at a frequency of 0.5 from sampling alone and that difference decides the verdict at the common and rare strata.

Tolerances come from the reference error on the shipped cohort. The RMSE is 0.0025 at true frequency zero, about 0.0105 to 0.0109 in the rare stratum and 0.0204 in the common stratum. The absolute bound of 0.008 at zero is about 3.2 times the reference RMSE and sits well below the 95th percentile of the error of both the raw read fraction (0.0252 in P2) and a single global miscall rate (0.0156 in P2), so a missing per-batch correction fails. The rare floor of 0.018 is about 1.7 times the rare RMSE with a 0.28 relative term. The common floor of 0.025 is about 1.2 times the common RMSE with a 0.22 relative term. Tolerance is relative to the minor allele frequency, because near 1 a relative tolerance on the ALT frequency would be ten times too loose.

Pass-share thresholds are 0.94, 0.90, 0.94 and 0.91 for the four strata, 0.90 for paralog recall and 0.94 for precision. On the shipped cohort the reference solution clears every gate (paralog recall 0.938, precision 0.999, zero ~0.97, rare ~0.92–0.93, common ~0.96, chrX common ~0.94–0.95). Two structurally independent alternative estimators that share no code with the reference (different EM schedules and different paralog statistics) were scored with the same verifier on the same truth table; both also clear every gate, with the weaker of the two still above threshold by at least 0.02 on every metric. Defective probes derived from those alternatives (global rather than per-batch error rate, no error correction, depth-only paralog calls, folded minor-allele frequencies) fail at least one gate. The thresholds therefore discriminate method class, not a single implementation.

Negative controls score 0: no output, a constant 0.5, the raw pooled read fraction, a 30-minute baseline with one global error rate and a depth cut, permuted rows and permuted labels. Shortcuts that score 0 include depth-only paralog calls, all or no paralogs, all-zero frequencies, pooled frequencies for both populations, a literal nan in a frequency cell, duplicated rows, an extra column, inference from site metadata and depth alone, a header with no rows, and a search of the agent image for a truth table to copy. Contested conventions are resolved in the instruction where they matter.

On a separate machine with Docker and Harbor the agent image builds within the limit, the offline oracle run with networking disabled writes all 12,000 rows, the Harbor oracle scores 1 on repeated runs, the nop agent scores 0, and the verifier is byte-identical on repeats. Frontier-agent pass rates are not claimed in this submission.

The simulation parameters are author choices and are not offered as published values. The sources for the methods the scheme imitates are Balding and Nichols 1995 (https://doi.org/10.1007/BF01441146) for the differentiation draw, Kim et al. 2011 (https://doi.org/10.1186/1471-2105-12-231) for the genotype-likelihood frequency estimator, and McKinney et al. 2017 (https://doi.org/10.1111/1755-0998.12613) for the paralog signature.

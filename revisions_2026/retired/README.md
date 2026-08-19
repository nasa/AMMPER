# Retired analysis (2026-08-04)

`ablate_optical_leak.py`, `ablation_leak.json`, `ablation_leak.log`

A four-arm ablation of an "optical leak" observation equation -- two extra
parameters that let a fraction of the culture's turbidity signal appear in the
blue and pink channels. Retired, not merely switched off, because the artifact it
was built to explain did not exist.

The leak was introduced to account for measured blue + pink summing to about
1.065 in the wild type, which a conserved three-species model cannot reach. That
sum was produced by an error in this revision's `load_experimental()`, which
divided both series by a shared denominator instead of reproducing the published
in-place normalization (blue by the raw initial total; pink by the
already-normalized blue[0] plus the raw pink[0]). The shared denominator scales
pink by roughly 2.7x. With the published normalization restored, blue + pink is
at or below 1 in every condition and the deficit behaves as the unmeasured
colorless species, which is what the three-species chain predicts.

Nothing in the ablation is therefore evidence about the plate reader. The
numbers it reported (arms A-D at 0.215 / 0.117 / 0.078 / 0.085) were computed
against the wrongly normalized data and do not correspond to anything. They are
kept here only so the record of what was tried is complete.

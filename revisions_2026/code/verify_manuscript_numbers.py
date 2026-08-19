"""
Check every quantitative claim in main.tex and supplemental.tex against the
generated result files, so that a refit cannot silently leave a stale number in
the manuscript.

Reads code/figure_numbers.json (what the figures actually plot),
code/revision_numbers.json, and the fitted constants in ammper_ab_model_fixed.py,
then greps the two .tex sources for the corresponding literals. Every value the
manuscript quotes is compared at the precision at which it is quoted: a value
written to three decimals must match the computed value rounded to three
decimals, not merely be close to it.

Exit status is 0 only if every check passes, so this can be run in a pre-submit
loop. It does not compile LaTeX (pdflatex is not available in this environment);
structural validation of labels, references and figure targets lives in the
checks at the end of the file, which parse the sources directly.
"""

import json
import os
import re
import sys

import ammper_ab_model_fixed as M

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

def _read_supplement():
    """The supplement is named supplemental.tex in the analysis tree and
    supplementary.tex in the manuscript folder; accept either."""
    for name in ('supplemental.tex', 'supplementary.tex'):
        path = os.path.join(ROOT, name)
        if os.path.exists(path):
            return open(path).read()
    raise FileNotFoundError(
        f'no supplemental.tex or supplementary.tex found in {ROOT}')


MAIN = open(os.path.join(ROOT, 'main.tex')).read()
SUPP = _read_supplement()
FIG = json.load(open(os.path.join(HERE, 'figure_numbers.json')))
REV = json.load(open(os.path.join(HERE, 'revision_numbers.json')))

FAILURES = []
CHECKS = 0


def quoted(value, places):
    """The literal a manuscript would use for ``value`` at ``places`` decimals."""
    return f'{value:.{places}f}'


def check(label, value, places, text, name):
    """Assert that ``text`` contains ``value`` rounded to ``places`` decimals."""
    global CHECKS
    CHECKS += 1
    literal = quoted(value, places)
    if literal not in text:
        FAILURES.append(f'{label}: {name} does not contain {literal} '
                        f'(computed {value!r})')


def check_absent(label, wrong, text, name):
    """Assert that a superseded literal is gone from ``text``."""
    global CHECKS
    CHECKS += 1
    if wrong in text:
        FAILURES.append(f'{label}: {name} still contains the superseded '
                        f'value {wrong}')


# ---------------------------------------------------------------- per-condition
# Text S3 quotes the full 12-row error table to three decimals.
for key, value in FIG['wt_mae'].items():
    check(f'MAE {key}', value, 3, SUPP, 'supplemental.tex')
for key, value in FIG['rad51_mae'].items():
    check(f'MAE {key}', value, 3, SUPP, 'supplemental.tex')

wt_mean = sum(FIG['wt_mae'].values()) / len(FIG['wt_mae'])
rad_mean = sum(FIG['rad51_mae'].values()) / len(FIG['rad51_mae'])
check('WT mean MAE', wt_mean, 3, MAIN, 'main.tex')
check('WT mean MAE', wt_mean, 3, SUPP, 'supplemental.tex')
check('rad51 mean MAE', rad_mean, 3, MAIN, 'main.tex')
check('rad51 mean MAE', rad_mean, 3, SUPP, 'supplemental.tex')

# ------------------------------------------------------------ between-dose spread
# These appear in three places: main Results, main Discussion, Fig. S1 caption.
for strain, block in (('WT', FIG['spread_wt']), ('rad51', FIG['spread_rad51'])):
    for species, values in block.items():
        check(f'{strain} {species} predicted spread',
              values['predicted'], 3, MAIN, 'main.tex')
        check(f'{strain} {species} predicted spread',
              values['predicted'], 3, SUPP, 'supplemental.tex')
        check(f'{strain} {species} experimental spread',
              values['experimental'], 3, MAIN, 'main.tex')
        check(f'{strain} {species} experimental spread',
              values['experimental'], 3, SUPP, 'supplemental.tex')

# The report and the figures must agree, or the manuscript has two sources of
# truth for the same quantity. They compute it by the same route only if
# generate_revision_numbers.py averages over replicate simulations.
for strain, fig_key in (('WT', 'spread_wt'), ('rad51', 'spread_rad51')):
    for species in ('blue', 'pink'):
        a = FIG[fig_key][species]['predicted']
        b = REV['dose_response'][strain]['species'][species]['predicted_spread']
        CHECKS += 1
        if abs(a - b) > 5e-4:
            FAILURES.append(f'spread {strain} {species}: figure_numbers '
                            f'{a:.4f} disagrees with revision_numbers {b:.4f}')

# ------------------------------------------------------------------- parameters
NAMES = ('v1', 'v2', 'v3', 'K1', 'K2', 'K3', 'k')


def latex_forms(value):
    """Every plausible way main.tex could typeset ``value``.

    Methods quotes the small parameters as plain decimals and the two large
    half-saturation constants in LaTeX scientific notation, e.g.
    ``1.43 \\times 10^4``. Comparing against a bare ``%g`` string would flag
    those as missing, so the mantissa/exponent form is generated explicitly.
    """
    forms = set()
    for sig in (2, 3, 4):
        forms.add(f'{value:.{sig}g}')
        exponent = 0
        mantissa = value
        while abs(mantissa) >= 10:
            mantissa /= 10.0
            exponent += 1
        if exponent:
            digits = max(sig - 1, 0)
            forms.add(f'{mantissa:.{digits}f} \\times 10^{exponent}')
            forms.add(f'{mantissa:.{digits}f} \\times 10^{{{exponent}}}')
    return forms


for name, value in zip(NAMES, M.SHARED_PARAMS):
    CHECKS += 1
    if not any(form in MAIN for form in latex_forms(value)):
        FAILURES.append(f'parameter {name}: no 2-4 significant-figure rounding '
                        f'of {value!r} appears in main.tex')

check('WT g0', M.WT_G0, 1, MAIN, 'main.tex')
check('WT g0', M.WT_G0, 2, SUPP, 'supplemental.tex')
check('rad51 g0', M.RAD51_G0, 1, MAIN, 'main.tex')
check('rad51 g0', M.RAD51_G0, 2, SUPP, 'supplemental.tex')

# ------------------------------------------------------ pre-correction baselines
# The manuscript reports the corrected results only: it does not tell the story of
# what the submitted version got wrong, because a before/after narrative in the
# paper itself is confusing to read. So these baselines are asserted *absent* from
# both .tex files and *present* in the internal bug note, which is where the
# comparison lives. That note is what gets shared with the author team, and it is
# the file that has to stay numerically in step with the fit.
pub_wt = REV['errors']['WT']
pub_mean = sum(r['published'] for r in pub_wt.values()) / len(pub_wt)
corr_mean = sum(r['corrected_published_params'] for r in pub_wt.values()) / len(pub_wt)

for label, value in (('published WT mean', pub_mean),
                     ('corr+pub WT mean', corr_mean)):
    literal = f'{value:.3f}'
    check_absent(f'{label} leaked into the manuscript', literal, MAIN, 'main.tex')
    check_absent(f'{label} leaked into the manuscript', literal, SUPP,
                 'supplemental.tex')

# The note sits in other/ in the analysis tree and notes/ in the manuscript
# folder; accept either, as with the supplement's two filenames.
BUG_NOTE_PATH = next(
    (p for p in (os.path.join(ROOT, d, 'BUGS_AND_CORRECTIONS.md')
                 for d in ('other', 'notes'))
     if os.path.exists(p)), None)
CHECKS += 1
if BUG_NOTE_PATH is None:
    FAILURES.append('bug note: BUGS_AND_CORRECTIONS.md not found in other/ or '
                    'notes/ -- the before/after comparison has no home outside '
                    'the manuscript')
else:
    BUG_NOTE = open(BUG_NOTE_PATH).read()
    check('published WT mean', pub_mean, 3, BUG_NOTE, 'BUGS_AND_CORRECTIONS.md')
    check('corr+pub WT mean', corr_mean, 3, BUG_NOTE,
          'BUGS_AND_CORRECTIONS.md')
    # The headline numbers have to agree with the fit as well, or the note the
    # team reads will disagree with the paper it explains.
    check('WT mean MAE', wt_mean, 3, BUG_NOTE, 'BUGS_AND_CORRECTIONS.md')
    check('rad51 mean MAE', rad_mean, 3, BUG_NOTE, 'BUGS_AND_CORRECTIONS.md')

# ------------------------------------------------------------- identifiability
# The manuscript must name exactly the (v, K) pairs that the scan finds flat.
scales = sorted(next(iter(REV['identifiability'].values())), key=int)
flat = [pair for pair, rows in REV['identifiability'].items()
        if abs(rows[scales[-1]] - rows[scales[0]]) < 0.01]
# Both halves of the reversible pink<->clear pair are flat with the published
# normalization restored; v1/K1 is not. Asserted as an exact set so that a refit
# which changes which pairs are identifiable cannot pass silently -- it did change
# once already, when the normalization was reverted.
CHECKS += 1
if flat != ['v2/K2', 'v3/K3']:
    FAILURES.append(f'identifiability: scan reports {flat} as flat, but the '
                    f'manuscript text is written for v2/K2 and v3/K3')
for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
    for constant in ('K_2', 'K_3'):
        CHECKS += 1
        if constant not in text:
            FAILURES.append(f'identifiability: {name} does not mention {constant}')

# --------------------------------------------------------------- population spread
for strain in ('WT', 'rad51'):
    block = REV['population'][strain]
    CHECKS += 1
    percent = block['healthy_spread_percent']
    if percent > 8.0:
        FAILURES.append(f'population {strain}: spread {percent:.1f}% exceeds the '
                        f'"less than eight percent" claim in main.tex')

# ------------------------------------------------------------------- structural
for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
    labels = set(re.findall(r'\\label\{([^}]+)\}', text))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', text))
    CHECKS += 1
    dangling = refs - labels
    if dangling:
        FAILURES.append(f'{name}: \\ref to undefined label(s) {sorted(dangling)}')

    # Every tabular must have a consistent column count.
    for match in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}',
                             text, re.S):
        spec, body = match.group(1), match.group(2)
        columns = len(re.findall(r'[lcrp]', re.sub(r'p\{[^}]*\}', 'p', spec)))
        for line in body.split(r'\\'):
            line = re.sub(r'\\multicolumn\{\d+\}\{[^}]*\}\{[^}]*\}', 'X', line)
            if not line.strip() or line.strip().startswith('\\hline'):
                continue
            CHECKS += 1
            cells = line.count('&') + 1
            if cells != columns and 'X' not in line:
                FAILURES.append(f'{name}: tabular row has {cells} cells but the '
                                f'spec declares {columns}: {line.strip()[:60]}')

# Cited keys must exist in the bibliography.
bib_path = os.path.join(ROOT, 'references.bib')
if os.path.exists(bib_path):
    bib = open(bib_path).read()
    keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
        cited = set()
        for group in re.findall(r'\\cite[tp]?\{([^}]+)\}', text):
            cited.update(k.strip() for k in group.split(','))
        CHECKS += 1
        missing = cited - keys
        if missing:
            FAILURES.append(f'{name}: \\cite to key(s) absent from '
                            f'references.bib: {sorted(missing)}')

# Included graphics must exist on disk.
for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
    for target in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
        CHECKS += 1
        candidates = [os.path.join(ROOT, target),
                      *(os.path.join(ROOT, target + ext)
                        for ext in ('.pdf', '.png', '.jpg', '.eps'))]
        if not any(os.path.exists(c) for c in candidates):
            FAILURES.append(f'{name}: \\includegraphics target not found: {target}')


# --------------------------------------------------------------- error metric
# The submitted manuscript defined the error as a mean absolute deviation but the
# code minimized a sum of squared residuals over the first eight timepoints, so
# the equation and the number beside it were different quantities. Everything
# reported now is MAE. These checks keep the two from drifting apart again.
CHECKS += 1
if 'mean absolute error' not in MAIN:
    FAILURES.append('metric: main.tex does not name the reported error as the '
                    'mean absolute error')
CHECKS += 1
if r'|x_i - y_i|' not in MAIN:
    FAILURES.append('metric: main.tex Equation (error) is not an absolute '
                    'deviation -- expected |x_i - y_i|')
CHECKS += 1
if r'\frac{1}{n}' not in MAIN:
    FAILURES.append('metric: main.tex Equation (error) is not a mean -- expected '
                    'a 1/n prefactor')
# Every error either document reports is now on the MAE scale, because the legacy
# search -- SMAC3 Bayesian optimization on a sum of squared residuals over the
# first eight timepoints, fitted to the wild type at 0 Gy alone -- is no longer
# described anywhere: it produced none of the reported values, so both documents
# were cut back to the differential evolution fit that did. Nothing on the legacy
# scale may reappear, since a squared-residual loss standing beside a mean
# absolute error invites a comparison between two different quantities.
for _phrase in ('SMAC', 'Bayesian optimization', 'sum of squared residuals'):
    for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
        CHECKS += 1
        # The withdrawal note in supplemental.tex names SMAC deliberately, so
        # LaTeX comment lines are excluded before testing.
        _body = '\n'.join(line for line in text.split('\n')
                          if not line.lstrip().startswith('%'))
        if _phrase.lower() in _body.lower():
            FAILURES.append(f'legacy optimizer: {name} describes "{_phrase}", '
                            f'but the superseded SMAC3 route was removed from '
                            f'both documents -- every reported number comes from '
                            f'differential evolution on the MAE objective')
# 0.522 was quoted in the submitted version as a model error and is not
# reproducible as one under either objective; it is the mean of the archived
# per-dose Bayesian-optimization errors (0.5221). With the legacy search gone
# there is no longer any sentence in which it can legitimately appear.
for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
    CHECKS += 1
    if '0.522' in text:
        FAILURES.append(f'unreproducible 0.522 baseline: {name} still contains '
                        f'the superseded value 0.522')


# ----------------------------------------------------------------- gamma mode
# The gamma analysis is reported in Results ("The aB model transfers to a gamma
# radiation environment") and Supplementary Text S2. Its numbers come from
# fitted_parameters_gamma.json and gamma_hit_rate_calibration.json, and are
# checked here at the precision the prose quotes them, on the same footing as the
# proton numbers above. Both files are required: the gamma claims cannot be
# validated without them, and silently skipping the checks would let a stale
# gamma number survive a refit.
GAMMA_FIT = json.load(open(os.path.join(HERE, 'fitted_parameters_gamma.json')))
GAMMA_CALIB = json.load(open(os.path.join(HERE,
                                          'gamma_hit_rate_calibration.json')))

_reported = GAMMA_FIT['transferred']
for label, value, places, texts in (
        ('gamma MAE, all twelve', _reported['mean'], 3, (MAIN, SUPP)),
        ('gamma MAE, WT', _reported['wt_mean'], 3, (MAIN, SUPP)),
        ('gamma MAE, rad51', _reported['rad51_mean'], 3, (MAIN, SUPP)),
        ('gamma MAE, refit on gamma', GAMMA_FIT['refit']['mean'], 3, (MAIN, SUPP)),
        ('gamma g0, WT', _reported['g0']['WT'], 2, (SUPP,)),
        ('gamma g0, rad51', _reported['g0']['rad51'], 2, (SUPP,)),
        ('gamma hit rate, shared', GAMMA_CALIB['shared'], 2, (MAIN, SUPP)),
        ('gamma hit rate, WT alone', GAMMA_CALIB['per_strain']['WT'], 2,
         (MAIN, SUPP)),
        ('gamma hit rate, rad51 alone', GAMMA_CALIB['per_strain']['rad51'], 2,
         (SUPP,)),
):
    for text in texts:
        check(label, value, places, text,
              'main.tex' if text is MAIN else 'supplemental.tex')

# The mutant residual is quoted as concentrated at the two highest doses, and the
# wild-type one as uniform over a stated range. Check the values and the ordering
# claim, not only the literals.
for key, places in (('grad51_20', 3), ('grad51_30', 3)):
    check(f'gamma per-dose MAE {key}', _reported['rad51_per_dose'][key], places,
          SUPP, 'supplemental.tex')
_wt_dose = list(_reported['wt_per_dose'].values())
for label, value in (('gamma WT residual range, low', min(_wt_dose)),
                     ('gamma WT residual range, high', max(_wt_dose))):
    check(label, value, 3, SUPP, 'supplemental.tex')
CHECKS += 1
if not (max(_reported['rad51_per_dose'].values())
        == _reported['rad51_per_dose']['grad51_30']):
    FAILURES.append('gamma: the supplement says the mutant residual is largest '
                    'at 30 Gy, but the computed maximum is at another dose')

# The degeneracy check is the argument that recalibrating k is a rescaling of the
# dose axis rather than a change to the simulator, so both the observed and the
# predicted survival triples must appear.
for entry in GAMMA_CALIB['degeneracy_check']:
    for field, places in (('observed_survival', 3), ('predicted_survival', 3)):
        check(f"gamma degeneracy k={entry['hit_rate']:g} {field}",
              entry[field], places, SUPP, 'supplemental.tex')

# The uncalibrated rate is 54x the calibrated one; the supplement states that
# factor, so derive it rather than trusting the sentence.
CHECKS += 1
_factor = GAMMA_CALIB['published_hit_rate'] / GAMMA_CALIB['shared']
if f'{_factor:.0f}' not in SUPP:
    FAILURES.append(f'gamma: supplemental.tex does not quote the ratio of the '
                    f'uncalibrated to the calibrated hit rate ({_factor:.0f})')

# The three configurations described in Text S2 must match the fit file: the
# uncalibrated rate, the transferred proton kinetics, and the refit. The two
# intermediate states of the analysis ('submitted', 'ordering_fixed') are
# deliberately absent from the manuscript -- they are lab-notebook material and
# live in notes/BUGS_AND_CORRECTIONS.md -- so their means must NOT appear.
for key in ('published_rate', 'transferred', 'refit'):
    check(f'gamma configuration {key} mean', GAMMA_FIT[key]['mean'], 3, SUPP,
          'supplemental.tex')
for key in ('submitted', 'ordering_fixed'):
    CHECKS += 1
    if f"{GAMMA_FIT[key]['mean']:.3f}" in SUPP:
        FAILURES.append(f'gamma: supplemental.tex quotes the mean of the '
                        f'withdrawn {key} configuration, which is an '
                        f'intermediate state of the analysis and belongs in '
                        f'notes/BUGS_AND_CORRECTIONS.md')
# The argument for calibrating k is that the uncalibrated rate is not merely
# worse but disqualifying, so Text S2 states the ratio per strain. Derive
# both rather than trusting the sentence.
for label, ratio in (
        ('gamma uncalibrated/calibrated error ratio, overall',
         GAMMA_FIT['published_rate']['mean'] / GAMMA_FIT['transferred']['mean']),
        ('gamma uncalibrated/calibrated error ratio, WT',
         GAMMA_FIT['published_rate']['wt_mean']
         / GAMMA_FIT['transferred']['wt_mean']),
        ('gamma uncalibrated/calibrated error ratio, rad51',
         GAMMA_FIT['published_rate']['rad51_mean']
         / GAMMA_FIT['transferred']['rad51_mean'])):
    check(label, ratio, 1, SUPP, 'supplemental.tex')
# Text S2 claims the calibrated rate wins at every irradiated dose, with the
# unirradiated mutant control as the stated exception. Check that the exception
# set is exactly that.
_exceptions = set()
for side in ('wt_per_dose', 'rad51_per_dose'):
    for condition, uncalibrated in GAMMA_FIT['published_rate'][side].items():
        if uncalibrated <= GAMMA_FIT['transferred'][side][condition]:
            _exceptions.add(condition)
CHECKS += 1
if _exceptions != {'grad51_0'}:
    FAILURES.append(f'gamma: Text S2 names the unirradiated mutant '
                    f'control as the only condition where calibration does not '
                    f'help, but the computed exception set is {sorted(_exceptions)}')

# The gamma mode must no longer be described as failing or as unvalidated: that
# was the submitted diagnosis and it was wrong.
for phrase in ('did not reproduce the gamma', 'failed to recapitulate',
               'exploratory gamma radiation mode'):
    for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
        CHECKS += 1
        if phrase in text:
            FAILURES.append(f'gamma: {name} still describes the gamma mode with '
                            f'the superseded phrase "{phrase}"')
CHECKS += 1
if 'transfers to a gamma radiation environment' not in MAIN:
    FAILURES.append('gamma: main.tex has no Results subsection reporting the '
                    'gamma transfer, but Text S2 says it corresponds to one')

# ------------------------------------------- withdrawn supplementary figures
# Five figures were withdrawn from the supplement over this revision round at the
# authors' discretion, while their generating code was kept: the grid-search vs
# legacy-optimizer comparison, the SMAC3 convergence curves, the gamma error
# decomposition, the gamma hit-rate calibration, and the gamma-against-proton
# comparison. Their images must not be \includegraphics'd back in without also
# restoring the figure-count expectation above, and no callout may point at a
# figure that no longer exists.
# Match only actual \includegraphics uses, not the LaTeX comment that records the
# withdrawal (which names the stems on purpose, so a future reader knows what was
# removed and where the code still lives).
_included = set(re.findall(r'\\includegraphics\[[^]]*\]\{([^}]*)\}', SUPP))
_included = {os.path.basename(_path).rsplit('.', 1)[0] for _path in _included}
for _stem in ('bo_vs_grid_search', 'smac_convergence',
              'gamma_error_decomposition', 'gamma_hit_rate_calibration',
              'gamma_vs_proton'):
    CHECKS += 1
    if _stem in _included:
        FAILURES.append(f'withdrawn figure {_stem} is referenced by '
                        f'supplemental.tex again; if it is being reinstated, '
                        f'update the expected figure count and renumber')

# ------------------------------------------------ supplementary figure numbering
# A \label whose name disagrees with the number LaTeX assigns is invisible in the
# PDF but misdirects every callout, so the top-level label of the nth figure
# environment must be fig:Sn and sublabels must be prefixed by it.
_figure_section = SUPP[SUPP.index(r'\section{Supplementary figures}'):]
_environments = re.findall(r'\\begin\{figure\}(.*?)\\end\{figure\}',
                           _figure_section, re.S)
CHECKS += 1
if len(_environments) != 7:
    FAILURES.append(f'supplementary figures: found {len(_environments)} figure '
                    f'environments, expected 7')
for index, body in enumerate(_environments, start=1):
    _labels = re.findall(r'\\label\{(fig:[^}]+)\}', body)
    CHECKS += 1
    if not _labels:
        FAILURES.append(f'supplementary figure {index}: no \\label')
        continue
    expected = f'fig:S{index}'
    if _labels[-1] != expected:
        FAILURES.append(f'supplementary figure {index}: top-level label is '
                        f'{_labels[-1]}, but it renders as S{index}')
    for sublabel in _labels[:-1]:
        CHECKS += 1
        if not sublabel.startswith(expected):
            FAILURES.append(f'supplementary figure {index}: sublabel {sublabel} '
                            f'is not prefixed by {expected}')

# Every supplementary figure must be cited from the main text: that is the
# contents policy stated in the supplement header.
_cited_numbers = set()
for first, last in re.findall(
        r'Fig(?:s|ure|ures)?\.?~?\s*S([0-9]+)(?:[A-C])?(?:--S([0-9]+))?', MAIN):
    if last:
        _cited_numbers.update(range(int(first), int(last) + 1))
    else:
        _cited_numbers.add(int(first))
for index in range(1, len(_environments) + 1):
    CHECKS += 1
    if index not in _cited_numbers:
        FAILURES.append(f'supplementary figure S{index} is never cited from '
                        f'main.tex, contrary to the stated contents policy')

# publiplots is used for every figure, so it must be cited where the figures are
# described, and the key must resolve (the bibliography check above covers the
# latter only if the key is actually cited). The layout note lives in main.tex
# Methods now that the supplement's statistics-and-code section was folded into
# it, so either document satisfies this.
CHECKS += 1
if 'botas2025publiplots' not in SUPP and 'botas2025publiplots' not in MAIN:
    FAILURES.append('neither document cites botas2025publiplots, but '
                    'every supplementary figure is laid out with publiplots')

# ------------------------------------------------- re-measured supplementary numbers
_SUPP_NUMBERS = os.path.join(HERE, 'supplementary_figure_numbers.json')
SUPP_FIG = json.load(open(_SUPP_NUMBERS))

# Damage distribution (Fig. S4): the zero-inflation count is quoted in main.tex.
_damage = SUPP_FIG['damage']
for label, value in (('damage total runs', _damage['total_runs']),
                     ('damage zero-damage runs', _damage['zero_damage_runs'])):
    CHECKS += 1
    if str(value) not in MAIN:
        FAILURES.append(f'{label}: main.tex does not contain {value}')

# Run time (Fig. S5): the crossover and the peak speedup are both quoted, and the
# submitted "minutes to seconds" claim must be gone, since it is not what the
# re-measured timings show.
_timing = SUPP_FIG['ros_timing']
CHECKS += 1
# LaTeX writes a thousands separator as 1{,}000 so that the comma is set as a
# digit separator rather than as punctuation, so both forms are admissible.
_crossover = _timing['crossover_ros_points']
if not any(form in MAIN for form in (f'{_crossover:,}',
                                     f'{_crossover:,}'.replace(',', '{,}'),
                                     str(_crossover))):
    FAILURES.append(f'ros timing: main.tex does not quote the measured crossover '
                    f'({_crossover:,} ROS points)')
check('ros timing max speedup', _timing['max_speedup'], 1, MAIN, 'main.tex')
for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
    check_absent('unmeasured run-time claim', 'minutes to seconds', text, name)
    check_absent('extrapolated run-time curve', 'second-degree polynomial',
                 text, name)

# The grid-search-against-Bayesian-optimization comparison is no longer a
# supplementary figure: both arms were run on the legacy objective (sum of squared
# residuals over the first eight timepoints), so neither produced a number the
# manuscript reports, and showing it next to the mean absolute errors invited a
# comparison across two different scales. supplementary_figure_numbers.json still
# carries the numbers, so assert the converse -- that they are NOT quoted.
_search = SUPP_FIG['search']
# mean_reduction is deliberately not on this list: it is the difference of the two
# means, 0.064, which to three places is also the gamma rad51-delta mean absolute
# error that both documents legitimately report. Absence-checking it flags that
# instead of the withdrawn figure.
for label, value, places in (('grid search mean', _search['grid_mean'], 3),
                             ('legacy-search mean', _search['bo_mean'], 3),
                             ('Mann-Whitney p', _search['mannwhitney_p'], 3)):
    for text, name in ((MAIN, 'main.tex'), (SUPP, 'supplemental.tex')):
        CHECKS += 1
        if f'{value:.{places}f}' in text:
            FAILURES.append(f'{label}: {name} quotes {value:.{places}f} from the '
                            f'withdrawn legacy-optimizer comparison, which '
                            f'is on the legacy squared-residual scale and not on '
                            f'the scale of the reported mean absolute errors')

# Dose-spread ratios (Figs. S1 and S2), quoted in both documents.
for strain in ('WT', 'rad51'):
    for species in ('blue', 'pink'):
        block = SUPP_FIG[f'spread_{strain}'][species]
        for which, places in (('predicted', 3), ('experimental', 3)):
            check(f'{strain} {species} {which} spread', block[which], places,
                  SUPP, 'supplemental.tex')

def main():
    for failure in FAILURES:
        print('FAIL ' + failure)
    print(f'\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())

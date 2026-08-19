"""
Shared publication style for the AMMPER-2 revision figures.

Every supplementary figure is built through this module so that the whole
supplement is typographically consistent: one font size scale, one set of
species colors, one dose colormap, one output routine. The supplementary figures
in the submitted version were made by several independent scripts over a period
of years, with different fonts, marker sizes and axis conventions in each; the
reviewer's complaint about how well the model matches the data is easier to
address when the figures showing that match are legible and comparable.

Layout is delegated to publiplots, which fixes the size of the *axes* in
millimeters and grows the canvas to fit the decorations, rather than fixing the
figure and letting the axes shrink. That is what makes panels of different
figures line up at the same physical size on the page.

    Botas, J. (2025). PubliPlots: Publication-ready plotting for Python.
    https://github.com/jorgebotas/publiplots

Cited in the manuscript wherever these figures are used; the bib key is
``botas2025publiplots``.

Conventions fixed here
----------------------
BLUE / PINK      the two measured alamarBlue species, used with the same
                 meaning in every figure: oxidized resazurin and reduced
                 resorufin. Chosen to read as "blue" and "pink" while staying
                 distinguishable in grayscale and to common forms of color
                 vision deficiency.
CLEAR            the unmeasured colorless species, shown as a gray dashed line
                 wherever it appears, so that a reader cannot mistake it for a
                 measurement.
dose_colors()    perceptually ordered colormap for a dose series, so that
                 "higher dose" is always "further along the colormap" and the
                 ordering can be read without consulting the legend.
PANEL_*          axes sizes in mm for the recurring panel shapes.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import publiplots as pp              # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

#: The two measured species. Carried over from the submitted figures so that a
#: reader comparing the revision against the original sees the same color coding.
BLUE = '#2E5EAA'
PINK = '#D64161'

#: The colorless species, which is modeled but never measured.
CLEAR = '#7F7F7F'

#: Corrected / as-published contrast, used only in the bug-illustration figure.
CORRECTED = '#1B7837'
PUBLISHED = '#762A83'

#: Strain accents, for figures where the two strains share an axes.
WT_COLOR = '#1F6F8B'
RAD51_COLOR = '#E07B39'

#: Axes bounding boxes in mm, per panel. publiplots guarantees these exactly, so
#: a 2x3 grid of PANEL_SMALL and a 2x2 grid of PANEL_MEDIUM have panels of
#: predictable relative size when printed.
PANEL_SMALL = (40.0, 30.0)     # one cell of a 2x3 or 3x3 dose grid
PANEL_MEDIUM = (52.0, 38.0)    # one cell of a 2x2 grid
PANEL_WIDE = (75.0, 45.0)      # single-panel figure, full text width
PANEL_SQUARE = (55.0, 55.0)    # heatmaps and correlation panels

#: Formats written for every figure. PNG for reading and for the LaTeX build that
#: has no vector version of the raster panels; PDF because the journal asks for
#: vector artwork where it exists.
FORMATS = ('png', 'pdf')

#: Raster resolution. ASM asks for 300 dpi minimum for halftones and 600 for
#: line art; 600 costs little here and covers both.
DPI = 600

STRAIN_LABEL = {'WT': 'Wild type', 'rad51': r'$rad51\Delta$'}


def init():
    """Apply the publiplots baseline. Call once at the top of a figure script."""
    pp.init_rcparams()


def dose_colors(n, cmap='viridis'):
    """``n`` colors ordered by dose, dark-to-light along ``cmap``.

    Stops at 0.88 rather than 1.0: the top of viridis is a pale yellow that
    disappears against white, which mattered in the submitted Figure S1 where the
    30 Gy curve was the hardest to see and is the one carrying the claim.
    """
    return plt.get_cmap(cmap)(np.linspace(0.05, 0.88, n))


def species_legend(experimental=True, predicted=True, clear=False):
    """Legend handles for the species, in a fixed order across all figures."""
    handles = []
    if experimental:
        handles += [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=BLUE,
                   markeredgecolor=BLUE, markersize=4,
                   label='Blue, measured'),
            Line2D([0], [0], marker='s', color='none', markerfacecolor=PINK,
                   markeredgecolor=PINK, markersize=4,
                   label='Pink, measured')]
    if predicted:
        handles += [
            Line2D([0], [0], color=BLUE, linewidth=1.6, label='Blue, predicted'),
            Line2D([0], [0], color=PINK, linewidth=1.6, label='Pink, predicted')]
    if clear:
        handles += [Line2D([0], [0], color=CLEAR, linewidth=1.4, linestyle='--',
                           label='Colorless, predicted (not measured)')]
    return handles


def dose_legend(doses, colors=None, cmap='viridis'):
    """Legend handles for a dose series, labeled in Gy."""
    colors = dose_colors(len(doses), cmap) if colors is None else colors
    return [Line2D([0], [0], color=color, linewidth=1.6, label=f'{dose:g} Gy')
            for dose, color in zip(doses, colors)]


def panel_letter(ax, letter, dx=-0.16, dy=1.12):
    """Bold panel letter outside the axes, at a fixed offset in axes units."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left')


def inset_label(ax, text, loc='upper left'):
    """Small in-axes label, e.g. the dose of a panel in a dose grid."""
    x, y, ha, va = {'upper left': (0.04, 0.96, 'left', 'top'),
                    'upper right': (0.96, 0.96, 'right', 'top'),
                    'lower left': (0.04, 0.04, 'left', 'bottom'),
                    'lower right': (0.96, 0.04, 'right', 'bottom')}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85,
                      edgecolor='0.7', linewidth=0.5))


def tidy(ax, grid=True):
    """Grid and spines, applied identically everywhere."""
    if grid:
        pp.add_grid(ax)
    pp.adjust_spines(ax)


def concentration_axes(ax, xmax, xlabel=None, ylabel=None):
    """Shared limits and labels for a concentration-fraction time course."""
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.05, 1.05)
    pp.set_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    tidy(ax)


def save(fig, directory, stem, formats=FORMATS, dpi=DPI, tight=False):
    """Write a figure to ``directory`` in every requested format.

    Returns the list of paths written, so a script can report exactly what it
    produced rather than leaving the caller to guess at the extensions.

    ``tight`` passes bbox_inches='tight'. publiplots deliberately defaults it to
    None so that the saved canvas is exactly the one it laid out, which is what
    keeps axes at the requested millimeter size; that default is right for any
    figure whose decorations all sit inside the canvas. It clips a figure-level
    legend anchored below the axes, though, because publiplots reserved no room
    for something it did not place. 'tight' only grows the canvas -- the axes keep
    their physical size -- so panels still print at comparable size across
    figures.
    """
    os.makedirs(directory, exist_ok=True)
    written = []
    for extension in formats:
        path = os.path.join(directory, f'{stem}.{extension}')
        # transparent=False: the manuscript is printed on white, and a
        # transparent background renders as black in some PDF viewers' dark mode,
        # which has caught out figure checks before.
        pp.savefig(path, dpi=dpi if extension != 'pdf' else None,
                   transparent=False, facecolor='white',
                   bbox_inches='tight' if tight else None)
        written.append(path)
    plt.close(fig)
    return written


#: Where figures go. The supplement directory holds only figures the supplement
#: includes, per the manuscript's contents policy; anything illustrative that the
#: paper does not cite goes elsewhere.
def output_dirs():
    """``(supplementary, main_panels, bug_illustration)`` output directories."""
    here = os.path.dirname(os.path.abspath(__file__))
    revision = os.path.dirname(here)
    return (os.path.join(revision, 'figures', 'supplementary_material'),
            os.path.join(revision, 'figures', 'main_figure_panels'),
            os.path.join(revision, 'figures', 'bug_illustration'))

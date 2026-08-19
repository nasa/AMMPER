"""
Copy the manuscript sources and figures into the analysis tree, translating paths.

The same manuscript exists in two places, and they are not interchangeable:

  manuscripts/AMMPER/         main.tex + supplementary.tex, figures under
                              figures/main_figures/ and figures/supplementary/
                              -- the authoring copy, and the one that gets
                              uploaded to Overleaf
  revisions_2026/             main.tex + supplemental.tex, figures under
                              Panels/ and Figures/ -- the copy that sits next to
                              the code, so verify_manuscript_numbers.py can check
                              the prose against the generated numbers

The prose is identical; only the graphics directory names differ, because the
submitted layout used Panels/ and Figures/. Keeping them in step by hand means
re-deriving that mapping every time, and it went wrong once already (the analysis
tree was edited while the manuscript copy was the current one, and the stale
version was mistaken for the fresh one). This script makes the manuscript folder
authoritative and does the translation mechanically.

    python3 sync_manuscript_trees.py            # report what differs
    python3 sync_manuscript_trees.py --write     # copy manuscript -> analysis tree

Figures are copied only when their contents differ, and a figure present in the
analysis tree but absent from the manuscript folder is reported rather than
deleted -- that asymmetry is usually a rename worth looking at.
"""

import argparse
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
MANUSCRIPT = '/home/sagemaker-user/manuscripts/AMMPER'

#: (manuscript file, analysis file, [(manuscript path fragment, analysis path fragment)])
SOURCES = [
    ('main.tex', 'main.tex', [('figures/main_figures/', 'Panels/')]),
    ('supplementary.tex', 'supplemental.tex', [('figures/supplementary/', 'Figures/'),
                                               ('figures/main_figures/', 'Panels/')]),
]

#: (manuscript figure directory, analysis figure directory)
FIGURE_DIRS = [
    ('figures/main_figures', 'Panels'),
    ('figures/supplementary', 'Figures'),
]


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.md5(handle.read()).hexdigest()


def _graphicspath_span(text):
    """Character span of the argument of ``\\graphicspath``, braces included.

    The argument is a brace group containing further brace groups, which a
    character-class regex cannot match -- it stops at the first closing brace --
    so the braces are counted explicitly.
    """
    start = text.find(r'\graphicspath')
    if start < 0:
        return None
    open_brace = text.find('{', start)
    if open_brace < 0:
        return None
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return open_brace, index + 1
    return None


def translate(text, substitutions):
    """Rewrite graphics paths from manuscript layout to analysis-tree layout.

    Only \\includegraphics targets and the \\graphicspath argument are touched, so
    a directory name appearing in prose or in a comment is left alone.
    """
    for manuscript_fragment, analysis_fragment in substitutions:
        text = re.sub(
            r'(\\includegraphics(?:\[[^\]]*\])?\{)' + re.escape(manuscript_fragment),
            lambda match: match.group(1) + analysis_fragment, text)

    span = _graphicspath_span(text)
    if span:
        start, end = span
        argument = text[start:end]
        for manuscript_fragment, analysis_fragment in substitutions:
            argument = argument.replace(manuscript_fragment, analysis_fragment)
        text = text[:start] + argument + text[end:]
    return text


def sync_sources(write):
    changed = []
    for manuscript_name, analysis_name, substitutions in SOURCES:
        source = os.path.join(MANUSCRIPT, manuscript_name)
        target = os.path.join(ANALYSIS, analysis_name)
        wanted = translate(open(source).read(), substitutions)
        current = open(target).read() if os.path.exists(target) else None
        if wanted == current:
            print(f'same       {analysis_name}')
            continue
        changed.append(analysis_name)
        if write:
            open(target, 'w').write(wanted)
            print(f'UPDATED    {analysis_name}  <- {manuscript_name}')
        else:
            print(f'DIFFERS    {analysis_name}  <- {manuscript_name}')
        leftover = [g for g in re.findall(
            r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', wanted)
            if 'figures/' in g]
        if leftover:
            print(f'  WARNING: untranslated graphics paths remain: {leftover}')
    return changed


def sync_figures(write):
    changed, orphans = [], []
    for manuscript_dir, analysis_dir in FIGURE_DIRS:
        source_dir = os.path.join(MANUSCRIPT, manuscript_dir)
        target_dir = os.path.join(ANALYSIS, analysis_dir)
        # Union of both directories, not just the target's contents: iterating
        # over the target alone silently skips figures newly added on the
        # manuscript side, which is exactly the case that matters after a figure
        # rebuild. Files only in the target are reported as orphans below.
        for name in sorted(set(os.listdir(source_dir)) | set(os.listdir(target_dir))):
            source = os.path.join(source_dir, name)
            target = os.path.join(target_dir, name)
            if not os.path.isfile(source):
                if os.path.isfile(target):
                    orphans.append(f'{analysis_dir}/{name}')
                continue
            if os.path.exists(target) and digest(source) == digest(target):
                continue
            changed.append(f'{analysis_dir}/{name}')
            if write:
                shutil.copy2(source, target)
                print(f'UPDATED    {analysis_dir}/{name}')
            else:
                print(f'DIFFERS    {analysis_dir}/{name}')
    for orphan in orphans:
        print(f'ORPHAN     {orphan} has no counterpart in the manuscript folder')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='apply the changes instead of only reporting them')
    arguments = parser.parse_args()

    changed = sync_sources(arguments.write) + sync_figures(arguments.write)
    if not changed:
        print('\nthe two trees agree')
        return 0
    if arguments.write:
        print(f'\n{len(changed)} file(s) updated; re-run '
              f'verify_manuscript_numbers.py')
        return 0
    print(f'\n{len(changed)} file(s) differ; re-run with --write to sync')
    return 1


if __name__ == '__main__':
    sys.exit(main())

"""
AMMPER core simulation modules.

This package holds the pieces of the agent-based model that are imported by the
simulation entry points in ``src/`` (AMMPER.py, AMMPERCLI.py, AMMPERBulk_aB.py,
AMMPERBulk_GAMMAfinal.py):

    cellDefinition            the Cell agent
    genTraverse_groundTesting  proton track generation, ground-test environments
    genTraverse_deepSpace      proton track generation, deep-space environment
    genROS                     ROS generation, diffusion-and-decay ("complex") model
    genROSOld                  ROS generation, static-and-eternal ("naive") model
    genROSDiffusion            standalone diffusion experiments
    cellPlot                   per-generation visualization
    cellPlot_deepSpace         per-generation visualization, deep-space variant
    GammaRadGen                exploratory gamma radiation event generation

The modules import one another by bare name (``from cellDefinition import Cell``)
rather than as a package, which is how they were originally written. The
``ammper_bootstrap`` module placed at the repository root puts this directory on
``sys.path`` so those bare imports continue to resolve after the August 2026
reorganization.
"""

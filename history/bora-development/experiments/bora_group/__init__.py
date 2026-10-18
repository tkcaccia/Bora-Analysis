"""Identify expert-derived, anonymized coarse groups after Bora refinement."""


def anonymous_palette(n):
    from experiments.bora_supervised.viewer import palette
    colours=palette(n)
    if n>=1:colours[1]=[65,125,235]
    if n>=2:colours[2]=[235,170,55]
    return colours

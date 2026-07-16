"""Analysis-side diagnostics — CW04 Q parameter + oracle-measurement utilities (measure).

Analysis-side and NON-differentiable (numpy/scipy permitted here only: MST + convex hull
for Q, FFT 2-point / CIC / POT measurement in :mod:`gravoturb.diagnostics.measure`).
The sqrt(N) normalization in m_bar = L_MST / sqrt(N*A) is mandatory; its absence was the
root cause of the discredited Q~0.13 headline. Validated against CW04 anchors (AC5).
"""

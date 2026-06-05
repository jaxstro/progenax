"""Substructure diagnostics — CW04 Q parameter (Cartwright & Whitworth 2004).

Analysis-side and NON-differentiable (numpy/scipy MST + convex hull permitted here only).
The sqrt(N) normalization in m_bar = L_MST / sqrt(N*A) is mandatory; its absence was the
root cause of the discredited Q~0.13 headline. Validated against CW04 anchors (AC5).
"""

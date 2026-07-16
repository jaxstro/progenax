"""3D density-field realization: GRF + rank copula, dense-tail mask, star sampling.

JAX-native. Produces a density field with the BM19 one-point marginal AND a turbulent
P(k) propto k^-beta, selects the self-gravitating dense tail, and samples star positions.
The cornerstone check is f_tail_actual ~= f_dense (AC6).
"""

# Stored results

Written by `scripts/verify.py` and read by `scripts/make_figs.py`. Distortions are average
trace distances; `c` is the overlap of the two pure targets.

| File | Section | Fields |
|---|---|---|
| `results_two.json` | `two` | `ellipse_random_max`: largest value of (ξ/S)² + ζ² over 20000 random measure-and-prepare protocols (at most one by Lemma 11); `cstar`, `Vstar`: maximizer and maximum of V(c); `gp_at_cstar`: guess-and-prepare at `cstar`; `gain_at_cstar`: relative gain 1 − V/(guess-and-prepare) there; `gp_peak_c`, `gp_peak`: maximum of guess-and-prepare, at c = √3/2 with value 1/8; `phistar_0861`: optimal polar angle φ⋆ at c = 0.861 |
| `results_sdp.json` | `sdp` | `rows`: for each overlap `c`, the closed form `V` of Eq. (21), the primal value `sdp` and the dual value `dual` of the semidefinite program over channels with a PPT Choi operator, and `gap` = primal − dual (Table 2) |
| `results_trine.json` | `trine` | `sdp`, `sdp_dual`: unrestricted optimum (1/4); `fid2_min`, `fid2_max`: range of the trace distance over the fidelity-optimal two-outcome protocols; `scan`: acquisition-limited optimum for record sizes d = 1 to 4; `vis`: visible optimum for d = 1 to 3; `haar_fixed`, `haar_rand`: Haar-random basis averaged over fixed-basis protocols and as a randomized channel |
| `results_sic.json` | `sic` | `sdp`, `sdp_dual`: unrestricted optimum (1/3); `scan`: acquisition-limited optimum for d = 1 to 5; `vis`: visible optimum for d = 1 to 4; `fid2_min`, `fid2_max`: range of the trace distance over the fidelity-optimal two-outcome protocols, 1/(2√2) and 1/√6; `haar_fixed`, `haar_rand`: Haar-random basis averaged over fixed-basis protocols and as a randomized channel |

Keys of `scan` and `vis` are record sizes.

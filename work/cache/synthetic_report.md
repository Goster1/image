# Synthetic round trip + sensitivity of the pipeline (work/44_synth_*.py)

Edge files: {'edges_cart310.json': 'reviewed', 'edges_cart80.json': 'reviewed', 'edges_scene.json': 'reviewed'}; 61 structural edges + 13 sticker sides; 62 valid corners of 20 stickers.

## Truths and noise models

* T1: f=1250, pp=(955,530), k1=-0.26, k2=0.06 (prescribed test lens); poses = pose-only fit to the real corners (synthetic vs real corners 6.4 px RMS)
* T2: f=1400, pp=(935,410), k1=-0.32, k2=0.08 (near the current edge-based estimates (lines joint f~1397, pp~(913,387); VP f~1394, pp~(951,416); plumb k1~-0.30..-0.34 at f 1400)); poses = pose-only fit to the real corners (synthetic vs real corners 6.8 px RMS)
* scenario a: corners: per-coordinate sigma = per-frame radial std / sqrt(2*7) (median 0.037 px); edges: AR(1) white noise with the per-edge sigma (median 0.150 px) and lag-1 correlation (median 0.44) of the high-frequency part of the real residuals
* scenario b: a + per-sticker 3D placement error sigma 4.96 mm (isotropic; moves corners and traced sides) + per cart-row height error sigma 33.1 mm (calibrated: sticker-only part -> 2.83 px, total drawing-fit RMS median 5.14 px vs real 5.34 px) + per-edge bow = real low-frequency residual profile (median rms 0.073 px) with random sign + per-edge direction deviation (VP-group edges) = real angle to the reference VP (median 0.048 deg, max 1.50 deg) with random sign
* scenario c: deterministic what-if: top stickers raised by the diagnosed end-board heights (+64/+71/+61/+81 mm) + noise a; median drawing-fit RMS 4.81 px

## Honesty summary (f, or k1 for the plumb-line; mapping = rotation-compensated error vs truth, median over the region)

| scenario | estimator | param | bias | scatter | RMSE | claimed cov / sandwich / boot | RMSE / claim | mapping cart band: actual (bias) | claimed cov / sandwich / boot | corners: actual | claimed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T2_a | M k1 ppfix | f | +8.2 | nan | 8.2 | 7.6 / 8.7 / 10.5 | 1.1 / 0.9 / 0.8 | 8.12 (8.12) | 2.15 / 1.71 / 4.57 | 19.33 | 4.27 / 5.02 / 13.72 |
| T2_a | M k1k2 ppfix | f | +24.6 | nan | 24.6 | 6.7 / 7.5 / 11.0 | 3.7 / 3.3 / 2.2 | 7.71 (7.71) | 2.74 / 2.79 / 3.84 | 52.90 | 8.47 / 9.38 / 13.28 |
| T2_a | M k1 ppfree | f | -0.2 | nan | 0.2 | 1.2 / 1.1 / - | 0.1 / 0.2 / - | 2.22 (2.22) | 1.59 / 4.00 / - | 29.14 | 2.02 / 5.56 / - |
| T2_a | M k1k2 ppfree | f | +0.4 | nan | 0.4 | 0.2 / 0.2 / 0.5 | 1.7 / 2.0 / 0.9 | 0.11 (0.11) | 0.13 / 0.10 / 0.21 | 1.25 | 0.60 / 0.57 / 1.43 |
| T2_a | plumb c-fixed | k1 | -0.0179 | nan | 0.0179 | - / 0.0020 / 0.0058 | - / 9.0 / 3.1 | 5.68 (5.68) | - / 0.24 / 0.41 | 19.54 | - / 1.82 / 3.45 |
| T2_a | plumb c-free | k1 | -0.0005 | nan | 0.0005 | - / 0.0017 / - | - / 0.3 / - | 0.13 (0.13) | - / 0.35 / - | 1.14 | - / 1.11 / - |
| T2_a | VP | f | -1.8 | nan | 1.8 | - / 9.0 / 64.6 | - / 0.2 / 0.0 | 2.21 (2.21) | - / 2.71 / 23.48 | 8.26 | - / 3.83 / 35.34 |
| T2_a | lines joint | f | +5.6 | nan | 5.6 | - / 4.0 / 4.2 | - / 1.4 / 1.4 | 1.87 (1.87) | - / 1.19 / 1.63 | 1.69 | - / 1.71 / 2.58 |
| T2_a | combined | f | +0.4 | nan | 0.4 | 0.2 / - / 0.4 | 1.9 / - / 1.0 | 0.19 (0.19) | 0.08 / - / 0.15 | 0.37 | 0.23 / - / 0.52 |

## Round trip T2_a (1 replicates)

Parameters (plumb-line k's are k/f0^2, k/f0^4 at f0 = 1300 in both estimate and truth):

| estimator | param | truth | bias +- se | scatter (robust) | claimed 1-sigma: cov / sandwich / bootstrap (median over replicates) | RMS z (cov / sandwich) | coverage of +-1 sigma (cov / sandwich) |
|---|---|---|---|---|---|---|---|
| M k1 ppfix | f | 1400.0 | +8.2 +- nan | nan (0.0) | 7.61 / 8.72 / 10.55 | 1.08 / 0.94 | 0.00 / 1.00 |
| M k1 ppfix | k1 | -0.3200 | +0.0565 +- nan | nan (0.0000) | 0.0031 / 0.0040 / 0.0139 | 18.21 / 14.08 | 0.00 / 0.00 |
| M k1k2 ppfix | f | 1400.0 | +24.6 +- nan | nan (0.0) | 6.72 / 7.53 / 10.95 | 3.65 / 3.26 | 0.00 / 0.00 |
| M k1k2 ppfix | k1 | -0.3200 | -0.0604 +- nan | nan (0.0000) | 0.0154 / 0.0180 / 0.0289 | 3.92 / 3.35 | 0.00 / 0.00 |
| M k1k2 ppfix | k2 | 0.0800 | +0.1163 +- nan | nan (0.0000) | 0.0256 / 0.0291 / 0.0475 | 4.55 / 3.99 | 0.00 / 0.00 |
| M k1 ppfree | f | 1400.0 | -0.2 +- nan | nan (0.0) | 1.21 / 1.14 / - | 0.15 / 0.15 | 1.00 / 1.00 |
| M k1 ppfree | cx | 935.0 | +5.7 +- nan | nan (0.0) | 1.58 / 3.14 / - | 3.59 / 1.81 | 0.00 / 0.00 |
| M k1 ppfree | cy | 410.0 | -6.7 +- nan | nan (0.0) | 1.83 / 2.69 / - | 3.68 / 2.50 | 0.00 / 0.00 |
| M k1 ppfree | k1 | -0.3200 | +0.0391 +- nan | nan (0.0000) | 0.0006 / 0.0010 / - | 69.52 / 38.41 | 0.00 / 0.00 |
| M k1k2 ppfree | f | 1400.0 | +0.4 +- nan | nan (0.0) | 0.25 / 0.21 / 0.48 | 1.67 / 2.00 | 0.00 / 0.00 |
| M k1k2 ppfree | cx | 935.0 | -0.3 +- nan | nan (0.0) | 0.43 / 0.32 / 0.86 | 0.78 / 1.08 | 1.00 / 0.00 |
| M k1k2 ppfree | cy | 410.0 | +0.4 +- nan | nan (0.0) | 0.45 / 0.37 / 0.80 | 0.77 / 0.94 | 1.00 / 1.00 |
| M k1k2 ppfree | k1 | -0.3200 | -0.0014 +- nan | nan (0.0000) | 0.0008 / 0.0007 / 0.0011 | 1.81 / 2.13 | 0.00 / 0.00 |
| M k1k2 ppfree | k2 | 0.0800 | +0.0028 +- nan | nan (0.0000) | 0.0015 / 0.0013 / 0.0026 | 1.83 / 2.15 | 0.00 / 0.00 |
| plumb c-fixed | k1 | -0.2759 | -0.0179 +- nan | nan (0.0000) | - / 0.0020 / 0.0058 | - / 8.96 | - / 0.00 |
| plumb c-fixed | k2 | 0.0595 | +0.0114 +- nan | nan (0.0000) | - / 0.0011 / 0.0075 | - / 10.58 | - / 0.00 |
| plumb c-free | cx | 935.0 | +0.0 +- nan | nan (0.0) | - / 1.24 / - | - / 0.0292 | - / 1.00 |
| plumb c-free | cy | 410.0 | +3.3 +- nan | nan (0.0) | - / 8.30 / - | - / 0.40 | - / 1.00 |
| plumb c-free | k1 | -0.2759 | -0.0005 +- nan | nan (0.0000) | - / 0.0017 / - | - / 0.27 | - / 1.00 |
| plumb c-free | k2 | 0.0595 | -0.0004 +- nan | nan (0.0000) | - / 0.0014 / - | - / 0.31 | - / 1.00 |
| VP | f | 1400.0 | -1.8 +- nan | nan (0.0) | - / 8.96 / 64.64 | - / 0.20 | - / 1.00 |
| VP | cx | 935.0 | +17.2 +- nan | nan (0.0) | - / 3.84 / 6.22 | - / 4.49 | - / 0.00 |
| VP | cy | 410.0 | +1.1 +- nan | nan (0.0) | - / 8.44 / 10.32 | - / 0.13 | - / 1.00 |
| lines joint | f | 1400.0 | +5.6 +- nan | nan (0.0) | - / 4.04 / 4.16 | - / 1.39 | - / 0.00 |
| lines joint | cx | 935.0 | +1.2 +- nan | nan (0.0) | - / 1.00 / 1.37 | - / 1.16 | - / 0.00 |
| lines joint | cy | 410.0 | -0.9 +- nan | nan (0.0) | - / 1.71 / 2.57 | - / 0.54 | - / 1.00 |
| lines joint | k1 | -0.3200 | -0.0020 +- nan | nan (0.0000) | - / 0.0013 / 0.0016 | - / 1.52 | - / 0.00 |
| lines joint | k2 | 0.0800 | -0.0003 +- nan | nan (0.0000) | - / 0.0006 / 0.0017 | - / 0.44 | - / 1.00 |
| combined | f | 1400.0 | +0.4 +- nan | nan (0.0) | 0.21 / - / 0.37 | 1.88 / - | 0.00 / - |
| combined | cx | 935.0 | +0.3 +- nan | nan (0.0) | 0.34 / - / 0.45 | 0.85 / - | 1.00 / - |
| combined | cy | 410.0 | +0.2 +- nan | nan (0.0) | 0.40 / - / 1.13 | 0.44 / - | 1.00 / - |
| combined | k1 | -0.3200 | +0.0009 +- nan | nan (0.0000) | 0.0003 / - / 0.0005 | 3.02 / - | 0.00 / - |
| combined | k2 | 0.0800 | -0.0019 +- nan | nan (0.0000) | 0.0005 / - / 0.0009 | 3.48 / - | 0.00 / - |

Mapping error vs truth (rotation compensated; median over the region of the per-point RMS over replicates):

| estimator | region | actual RMS error | of which bias | scatter | claimed: cov / sandwich / bootstrap | actual / claimed |
|---|---|---|---|---|---|---|
| M k1 ppfix | centre | 8.59 | 8.59 | 0.00 | 0.68 / 0.67 / 2.14 | 12.7 / 12.9 / 4.0 |
| M k1 ppfix | cart_band | 8.12 | 8.12 | 0.00 | 2.15 / 1.71 / 4.57 | 3.8 / 4.7 / 1.8 |
| M k1 ppfix | corners | 19.33 | 19.33 | 0.00 | 4.27 / 5.02 / 13.72 | 4.5 / 3.8 / 1.4 |
| M k1k2 ppfix | centre | 9.26 | 9.26 | 0.00 | 0.58 / 0.63 / 0.81 | 15.9 / 14.8 / 11.4 |
| M k1k2 ppfix | cart_band | 7.71 | 7.71 | 0.00 | 2.74 / 2.79 / 3.84 | 2.8 / 2.8 / 2.0 |
| M k1k2 ppfix | corners | 52.90 | 52.90 | 0.00 | 8.47 / 9.38 / 13.28 | 6.2 / 5.6 / 4.0 |
| M k1 ppfree | centre | 0.49 | 0.49 | 0.00 | 1.52 / 3.98 / - | 0.3 / 0.1 |
| M k1 ppfree | cart_band | 2.22 | 2.22 | 0.00 | 1.59 / 4.00 / - | 1.4 / 0.6 |
| M k1 ppfree | corners | 29.14 | 29.14 | 0.00 | 2.02 / 5.56 / - | 14.5 / 5.2 |
| M k1k2 ppfree | centre | 0.09 | 0.09 | 0.00 | 0.0601 / 0.0483 / 0.13 | 1.4 / 1.8 / 0.6 |
| M k1k2 ppfree | cart_band | 0.11 | 0.11 | 0.00 | 0.13 / 0.0988 / 0.21 | 0.9 / 1.1 / 0.5 |
| M k1k2 ppfree | corners | 1.25 | 1.25 | 0.00 | 0.60 / 0.57 / 1.43 | 2.1 / 2.2 / 0.9 |
| plumb c-fixed | centre | 8.71 | 8.71 | 0.00 | - / 0.0024 / 0.0050 | 3683.0 / 1729.9 |
| plumb c-fixed | cart_band | 5.68 | 5.68 | 0.00 | - / 0.24 / 0.41 | 24.0 / 14.0 |
| plumb c-fixed | corners | 19.54 | 19.54 | 0.00 | - / 1.82 / 3.45 | 10.7 / 5.7 |
| plumb c-free | centre | 0.16 | 0.16 | 0.00 | - / 0.47 / - | 0.3 |
| plumb c-free | cart_band | 0.13 | 0.13 | 0.00 | - / 0.35 / - | 0.4 |
| plumb c-free | corners | 1.14 | 1.14 | 0.00 | - / 1.11 / - | 1.0 |
| VP | centre | 0.99 | 0.99 | 0.00 | - / 0.81 / 5.40 | 1.2 / 0.2 |
| VP | cart_band | 2.21 | 2.21 | 0.00 | - / 2.71 / 23.48 | 0.8 / 0.1 |
| VP | corners | 8.26 | 8.26 | 0.00 | - / 3.83 / 35.34 | 2.2 / 0.2 |
| lines joint | centre | 0.42 | 0.42 | 0.00 | - / 0.27 / 0.42 | 1.5 / 1.0 |
| lines joint | cart_band | 1.87 | 1.87 | 0.00 | - / 1.19 / 1.63 | 1.6 / 1.1 |
| lines joint | corners | 1.69 | 1.69 | 0.00 | - / 1.71 / 2.58 | 1.0 / 0.7 |
| combined | centre | 0.03 | 0.03 | 0.00 | 0.0330 / - / 0.10 | 1.1 / 0.3 |
| combined | cart_band | 0.19 | 0.19 | 0.00 | 0.0833 / - / 0.15 | 2.3 / 1.3 |
| combined | corners | 0.37 | 0.37 | 0.00 | 0.23 / - / 0.52 | 1.6 / 0.7 |

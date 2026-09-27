## Tabuľka metód (automaticky, 97_report_tables.py)

| metóda | rozmery z výkresu | f [px] | cx | cy | k1 | k2 | k3 | fold | Δ mapovania vs hlavný: stred / pás / rohy [px] | neistota zobrazenia metódy (stred / pás / rohy) | pozn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M6 kombinovaný – HLAVNÝ | áno (všetky nálepky, plná váha) | 1456 ± 14 | 928 ± 5 | 502 ± 9 | -0.345 | +0.096 | +0.000 | ok | 0.0 / 0.0 / 0.0 | 1.7 / 6.0 / 9.0 | nálepky + 61 hrán + strany zakrytých nálepiek |
| M6 kombinovaný, robustné váhy nálepiek | áno (top-nálepky ~0 váha) | 1475 | 930 | 504 | -0.356 | +0.103 | +0.000 | ok | 1.3 / 5.9 / 7.6 | – / – / – | Cauchy váhy po nálepkách |
| M6 kombinovaný, k1,k2,k3 | áno | 1455 | 930 | 504 | -0.358 | +0.131 | -0.024 | ok | 0.1 / 0.9 / 1.2 | – / – / – | k3 len konzistentné |
| M6 kombinovaný, pp v strede obrazu | áno | 1458 | 960 | 540 | -0.368 | +0.113 | +0.000 | ok | 3.0 / 2.1 / 9.8 | – / – / – |  |
| M6 kombinovaný, fx≠fy | áno | 1439/1452 | 928 | 505 | -0.342 | +0.093 | +0.000 | ok | 0.8 / 4.8 / 9.3 | – / – / – | fx≠fy neurčené |
| M6 kombinovaný, +p1,p2 | áno | 1389 | 945 | 449 | -0.310 | +0.077 | +0.000 | ok | 4.8 / 19.5 / 21.8 | – / – / – | p1,p2 pohltia nesúlad geometrie |
| M4 spoločný fit z čiar (nezávislá impl., spoločná zvislica) | nie | 1473 ± 22 | 931 ± 8 | 438 ± 46 | -0.349 | +0.095 | +0.000 | ok | 4.7 / 6.4 / 11.9 | 3.3 / 7.0 / 11.8 |  |
| M4 spoločný fit z čiar, podlahová čiara ako 1 priamka | nie | 1479 ± 21 | 939 ± 7 | 509 ± 42 | -0.362 | +0.109 | +0.000 | ok | 1.6 / 6.9 / 9.2 | – / – / – |  |
| M4 spoločný fit z čiar (orchestrátor, samostatné zvislice) | nie | 1425 ± 124 | 927 ± 6 | 503 ± 8 | -0.333 | +0.090 | +0.000 | ok | 2.2 / 10.0 / 14.2 | – / – / – |  |
| M3 úbežníky (nezávislá impl.) | nie | 1479 ± 48 | 929 ± 9 | 426 ± 49 | -0.349 | +0.094 | +0.000 | ok | 5.6 / 8.6 / 14.9 | 4.8 / 14.9 / 22.1 | pp = stred skreslenia |
| M3 úbežníky (orchestrátor) | nie | 1419 ± 114 | 941 ± 7 | 513 ± 5 | -0.348 | +0.102 | +0.000 | ok | 2.6 / 12.9 / 23.2 | – / – / – |  |
| M2 plumb-line (skreslenie; f z M3) | nie | 1479 ± 48 | 929 ± 10 | 426 ± 50 | -0.349 | +0.094 | +0.000 | ok | 5.6 / 8.6 / 14.9 | 3.5 / 2.5 / 7.9 | len k1/f², k2/f⁴ a stred |
| M9 Brown k1,k2 (štúdia modelov, spoločné dáta) | áno | 1434 ± 43 | 927 ± 10 | 502 ± 43 | -0.336 | +0.091 | +0.000 | ok | 1.5 / 6.8 / 9.3 | 4.3 / 15.5 / 22.1 |  |
| M9 Brown k1,k2,k3 | áno | 1432 ± 44 | 931 ± 10 | 505 ± 35 | -0.353 | +0.137 | -0.031 | ok | 1.7 / 8.5 / 11.5 | 4.1 / 15.1 / 21.1 |  |
| M9 divízny l1,l2 | áno | 1431 | 935 | 508 | -0.356 | +0.138 | -0.028 | ok | 1.7 / 9.0 / 13.0 | 1.8 / 7.6 / 10.7 | prevedený na Brown |
| M9 Kannala-Brandt k1,k2 | áno | 1431 | 931 | 506 | -0.354 | +0.140 | -0.032 | ok | 1.8 / 8.9 / 11.9 | 1.8 / 7.7 / 10.8 | prevedený na Brown |
| M7 dva vozíky (normály podlahy) | áno | 1378 ± 81 | 955 | 522 | -0.328 | +0.090 | +0.000 | ok | 5.5 / 26.0 / 41.6 | 5.9 / 25.9 / 35.8 | len konzistencia |
| M8 tvar nálepiek | nie (len štvorce) | 1601 ± 174 | 979 ± 49 | 471 ± 26 | -0.443 | +0.165 | +0.000 | ok | 10.4 / 43.6 / 54.1 | 12.8 / 49.4 / 70.9 | príliš slabé |
| M5 kváder s policami (výkres) | áno | 1359 ± 36 | 960 | 540 | -0.210 | +0.000 | +0.000 | ok | 6.6 / 24.1 / 14.1 | 2.9 / 13.4 / 21.9 | vychýlené geometriou |
| M1 len nálepky (výkres) | áno | 1326 ± 20 | 960 | 540 | -0.246 | +0.025 | +0.000 | ok | 9.0 / 39.3 / 50.1 | 2.9 / 15.4 / 36.9 | vychýlené, prehnuté |

## Reziduá nálepiek s hlavným objektívom (geometria z výkresu, pózy vozíkov robustne)

| vozík | id | poloha | RMS [px] | priemerný posun (x, y) [px] |
|---|---|---|---|---|
| 80 | 0 | TOP-TL | 2.9 | (-1.5, +0.9) |
| 80 | 1 | TOP-TR | 3.3 | (-1.2, -2.6) |
| 80 | 2 | TOP-BR | 5.6 | (-3.2, -4.4) |
| 80 | 3 | A-LEFT | 7.6 | (+6.7, -2.8) |
| 80 | 4 | A-RIGHT | 14.8 | (+6.1, +13.5) |
| 80 | 5 | B-LEFT | 8.9 | (+6.7, -5.7) |
| 80 | 6 | B-RIGHT | 10.1 | (+2.4, +9.8) |
| 80 | 7 | C-LEFT | 8.8 | (-4.9, -7.3) |
| 80 | 8 | C-RIGHT | 7.3 | (-4.8, +5.5) |
| 80 | 92 | TOP-BL | 5.0 | (-2.9, +2.8) |
| 310 | 0 | TOP-TL | 5.1 | (-2.8, -3.0) |
| 310 | 1 | TOP-TR | 3.5 | (+1.7, -2.3) |
| 310 | 2 | TOP-BR | 8.8 | (+5.9, +6.1) |
| 310 | 3 | A-LEFT | 15.5 | (-0.1, +15.5) |
| 310 | 4 | A-RIGHT | 10.2 | (-10.0, +0.6) |
| 310 | 5 | B-LEFT | 12.3 | (+7.2, +9.9) |
| 310 | 6 | B-RIGHT | 5.8 | (-4.9, -2.4) |
| 310 | 7 | C-LEFT | 10.1 | (+5.4, +8.5) |
| 310 | 8 | C-RIGHT | 2.1 | (+0.4, -1.8) |
| 310 | 322 | TOP-BL | 3.9 | (+0.3, +1.3) |

## Rozpočet neistoty zobrazenia (1σ, px, medián oblasti)

| oblasť | štatistická | systematická | spolu (medián) | spolu (max) |
|---|---|---|---|---|
| centre | 1.6 | 2.7 | 3.2 | 4.4 |
| cart_band | 5.5 | 7.1 | 10.9 | 15.9 |
| corners | 7.9 | 11.0 | 16.1 | 19.7 |

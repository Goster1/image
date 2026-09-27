## Tabuľka metód (automaticky, 97_report_tables.py)

| metóda | rozmery z výkresu | f [px] | cx | cy | k1 | k2 | k3 | prehnutie | Δ zobrazenia vs hlavný: stred / pás / rohy [px] | neistota zobrazenia metódy (stred / pás / rohy) | pozn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **M6 kombinovaný – HLAVNÝ** | áno (všetky nálepky) | 1469 ± 47 | 928 ± 19 | 503 ± 29 | -0.352 | +0.099 | +0.000 | nie | 0.0 / 0.0 / 0.0 | 4.0 / 15.3 / 21.6 | celková neistota |
| M6, samostatné váhy top / police | áno | 1473 | 929 | 504 | -0.354 | +0.101 | +0.000 | nie | 0.3 / 1.2 / 1.4 | – / – / – | variant |
| M6, robustné váhy nálepiek | áno (top-nálepky ~0 váha) | 1476 | 930 | 505 | -0.357 | +0.103 | +0.000 | nie | 0.5 / 2.2 / 2.7 | – / – / – | variant |
| M6, stĺpiky ∥ zvislica scény | áno | 1467 | 931 | 507 | -0.354 | +0.102 | +0.000 | nie | 0.3 / 0.9 / 2.1 | – / – / – | variant |
| M6, bez strán 80:3/1, 80:7/3 | áno | 1443 | 928 | 502 | -0.339 | +0.092 | +0.000 | nie | 1.8 / 8.1 / 11.0 | – / – / – | variant |
| M6, bez lemu E vozíka 80 | áno | 1468 | 931 | 505 | -0.352 | +0.100 | +0.000 | nie | 0.2 / 0.4 / 0.7 | – / – / – | variant |
| M6, podlahové čiary len na priamosť | áno | 1469 | 928 | 492 | -0.350 | +0.098 | +0.000 | nie | 0.8 / 0.5 / 1.6 | – / – / – | variant |
| M6 kombinovaný, k1,k2,k3 | áno | 1470 | 930 | 505 | -0.364 | +0.132 | -0.022 | nie | 0.1 / 0.2 / 0.4 | – / – / – | k3 len konzistentné |
| M6 kombinovaný, pp v strede obrazu | áno | 1467 | 960 | 540 | -0.372 | +0.116 | +0.000 | nie | 3.0 / 2.7 / 11.0 | – / – / – |  |
| M6 kombinovaný, fx≠fy | áno | 1431/1463 | 928 | 509 | -0.345 | +0.095 | +0.000 | nie | 1.6 / 10.5 / 20.3 | – / – / – | fx≠fy neurčené |
| M6 kombinovaný, +p1,p2 | áno | 1398 | 946 | 453 | -0.314 | +0.079 | +0.000 | nie | 4.9 / 20.6 / 24.5 | – / – / – | p1,p2 pohltia nesúlad geometrie |
| M4 spoločný fit z čiar (nezávislá impl., spoločná zvislica) | nie | 1473 ± 22 | 931 ± 8 | 438 ± 46 | -0.349 | +0.095 | +0.000 | nie | 4.5 / 3.1 / 9.3 | 3.3 / 7.0 / 11.8 |  |
| M4 spoločný fit z čiar, podlahová čiara ako 1 priamka | nie | 1479 ± 21 | 939 ± 7 | 509 ± 42 | -0.362 | +0.109 | +0.000 | nie | 0.9 / 2.9 / 3.9 | – / – / – |  |
| M4 spoločný fit z čiar (prvá implementácia, samostatné zvislice) | nie | 1425 ± 124 | 927 ± 6 | 503 ± 8 | -0.333 | +0.090 | +0.000 | nie | 3.0 / 14.0 / 19.6 | – / – / – | cy ± len štatistická |
| M3 úbežníky (nezávislá impl.) | nie | 1479 ± 48 | 929 ± 9 | 426 ± 49 | -0.349 | +0.094 | +0.000 | nie | 5.4 / 4.9 / 11.8 | 4.8 / 14.9 / 22.1 | pp = stred skreslenia |
| M3 úbežníky (prvá implementácia, `30_lines_methods.py`) | nie | 1419 ± 114 | 941 ± 7 | 513 ± 5 | -0.348 | +0.102 | +0.000 | nie | 3.5 / 16.9 / 28.6 | – / – / – | cy ± len štatistická |
| M2 plumb-line (skreslenie; f z M3) | nie | 1479 ± 48 | 929 ± 10 | 426 ± 50 | -0.349 | +0.094 | +0.000 | nie | 5.4 / 4.9 / 11.8 | 3.5 / 2.5 / 7.9 | len k1/f², k2/f⁴ a stred |
| M9 Brown k1,k2 (štúdia modelov, spoločné dáta) | áno | 1434 ± 43 | 927 ± 10 | 502 ± 43 | -0.336 | +0.091 | +0.000 | nie | 2.3 / 10.7 / 14.7 | 4.3 / 15.5 / 22.1 |  |
| M9 Brown k1,k2,k3 | áno | 1432 ± 44 | 931 ± 10 | 505 ± 35 | -0.353 | +0.137 | -0.031 | na bariére | 2.7 / 12.4 / 16.9 | 4.1 / 15.1 / 21.1 |  |
| M9 divízny l1,l2 | áno | 1431 | 935 | 508 | -0.356 | +0.138 | -0.028 | nie | 2.7 / 12.9 / 18.4 | 1.8 / 7.6 / 10.7 | prevedený na Brown |
| M9 Kannala-Brandt k1,k2 | áno | 1431 | 931 | 506 | -0.354 | +0.140 | -0.032 | na bariére | 2.7 / 12.8 / 17.3 | 1.8 / 7.7 / 10.8 | prevedený na Brown |
| M7 dva vozíky (normály podlahy) | áno | 1378 ± 81 | 955 | 522 | -0.328 | +0.090 | +0.000 | nie | 6.4 / 29.9 / 47.0 | 5.9 / 25.9 / 35.8 | len konzistencia |
| M8 tvar nálepiek | nie (len štvorce) | 1601 ± 174 | 979 ± 49 | 471 ± 26 | -0.443 | +0.165 | +0.000 | nie | 9.7 / 39.4 / 48.4 | 12.8 / 49.4 / 70.9 | príliš slabé |
| M5 kváder s policami (výkres) | áno | 1359 ± 58 | 960 | 540 | -0.210 | +0.000 | +0.000 | na bariére | 7.5 / 28.5 / 19.0 | 2.9 / 13.4 / 21.9 | vychýlené geometriou |
| M1 len nálepky (výkres) | áno | 1326 ± 35 | 960 | 540 | -0.246 | +0.025 | +0.000 | na bariére | 10.1 / 43.1 / 55.4 | 2.9 / 15.4 / 36.9 | vychýlené geometriou |

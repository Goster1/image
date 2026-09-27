# Nezávislé (slepé) určenie parametrov objektívu staničnej kamery

Kamera Tracer WEB007, obraz 1920×1080, 7 fotiek z 2026-08-11 08:58. Všetko je odvodené len z dát v tomto repozitári.
Údaj výrobcu ani žiadne iné čísla zvonka som nepoužil. Kód je v `work/`, výstupy v `results/`.

## 1. Výsledok

**Hlavný výsledok** (`results/lens_result.json`), konvencia OpenCV (pinhole + Brown-Conrady):

| parameter | hodnota | 1σ | stav |
|---|---|---|---|
| fx = fy | **1455,7 px** | ± 34 px (2,3 %) | určené len spojením nálepiek a hrán; samotné hrany dávajú f krehko |
| cx | **927,6 px** | ± 12 px | určené |
| cy | **502,3 px** | ± 35 px | len konzistentné (426–513 podľa predpokladov o čiarach na podlahe) |
| k1 | **−0,345** | ± 0,021 | určené spolu s k2 (silná korelácia −0,93) |
| k2 | **+0,096** | ± 0,024 | určené spolu s k1 |
| p1, p2 | 0 (fixované) | – | dáta ich neurčia (voľné iba absorbujú nesúlad geometrie) |
| k3 | 0 (fixované) | (±0,015 ako systematika) | len konzistentné (−0,02…−0,03), zlepšenie nevýznamné |

`K = [[1455.68, 0, 927.59], [0, 1455.68, 502.32], [0, 0, 1]]`, `dist = [-0.3450, 0.0955, 0, 0, 0]`.
V pixelových jednotkách, ktoré nezávisia od f: k1/f² = −1,63·10⁻⁷ px⁻², k2/f⁴ = 2,13·10⁻¹⁴ px⁻⁴.
Model je invertovateľný až po rohy obrazu (nie je prehnutý).

**Neistota zobrazenia** (1σ posun priemetu pevného lúča, s kompenzáciou rotácie kamery, medián oblasti / max):

| oblasť | 1σ [px] | max v oblasti [px] |
|---|---|---|
| stred (r < 150 px) | **3,2** | 4,4 |
| pás vozíkov | **10,9** | 15,9 |
| rohy (200×150 px) | **16,1** | 19,7 |

Pri použití objektívu so súčasne odhadovanou pózou určuje veľkosť chyby hlavne neistota f: 1 % vo f je ~7 px v páse vozíkov.

**Najdôležitejšie zistenie:** geometria nálepiek podľa výkresu **nie je konzistentná so žiadnou dierkovou kamerou**
(RMS 5 px aj pri úplne voľnej projektívnej kamere, šum je ~0,15 px). Top-nálepky vychádzajú voči nálepkám
na policiach o ~60–80 mm vyššie a koncové dosky sú naklonené (kap. 5). Podľa zadania rozmery neupravujem. Hlavný výsledok
preto vznikol zo všetkých nálepiek s geometriou z výkresu **spolu so 61 overenými hranami konštrukcie**, ktoré nepoužívajú
žiadne rozmery. Každý blok dát má váhu podľa vlastného rozptylu. Nálepky samotné by s výkresom dali f ≈ 1300–1360,
cy ≈ 480–540 a skreslenie, ktoré sa prehne ešte vo vnútri obrazu a odporuje priamosti hrán. Tento výsledok je vychýlený
(kap. 5, 9) a uvádzam ho len v porovnaní.

**Porovnanie metód v skratke** (podrobne kap. 8):

| metóda | f [px] | pp [px] | poznámka |
|---|---|---|---|
| hlavný: nálepky (výkres) + hrany | 1456 ± 34 | (928, 502) | |
| len hrany, spoločná zvislica | 1473 ± 22 | (931, 438 ± 46) | bez rozmerov |
| len hrany, podlahová čiara ako jedna priamka | 1479 ± 21 | (939, 509) | |
| len hrany, samostatné zvislice | 1425 ± 124 | (927, 503) | f nestabilné |
| úbežníky (+ plumb-line) | 1479 ± 48 | (929, 426) | |
| štúdia modelov, spoločné dáta | 1434 ± 43 | (927, 502) | |
| dva vozíky na jednej podlahe | 1378 ± 81 | – | konzistenčná kontrola |
| model vozíka ako kvádra (výkres) | 1359 ± 58 | stred obrazu | vychýlené geometriou |
| len nálepky (výkres) | 1300–1375 | – | vychýlené, RMS 5,3–5,9 px |

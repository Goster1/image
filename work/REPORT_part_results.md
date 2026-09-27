## 7. Výber modelu (primerane dátam)

Zo štúdie 16 modelov na rovnakých dátach (`work/43_altmodels_*`, `results/altmodels_*.png`, `work/cache/altmodels_summary.md`):

* **Radiálne skreslenie potrebuje práve 2 stupne voľnosti.** k1 samotné hrany nenarovná (0,29–0,51 px, naráža
  na bariéru prehnutia, ΔQAIC +302). Všetky dvoj- a trojparametrové radiálne tvary (Brown k1k2, k1k2k3, divízny l1l2,
  racionálny k4, Kannala-Brandt k1k2) dosiahnu 0,228–0,229 px. To je hranica daná vlastnou nerovnosťou hrán
  a ich mapovania sa líšia o ≤ 0,4 / 2,0 / 2,7 px (stred / pás / rohy).
* **k3** vychádza −0,02 až −0,03 ± 0,02: je len konzistentné, preto ho fixujem na 0 a jeho vplyv je v systematike.
* **p1, p2:** na hranách vychádzajú ~0 (−0,001 ± 0,006; 0,0015 ± 0,003) a fit je nestabilný. S nálepkami s geometriou z výkresu
  vyskočí p1 na 0,009 a f na 1389, lebo tangenciálne členy len pohlcujú nesúlad geometrie. Preto p1 = p2 = 0.
* **fx ≠ fy** nie je určené (fx 1383 ± 91 vs. fy 1426 ± 34 z hrán), preto fx = fy (štvorcové pixely).
* **Hlavný bod voľný:** fixácia do stredu obrazu stojí ΔQAIC +21…+24 a zhoršuje krížovú predikciu. cx je určené,
  cy len konzistentné.
* Nie-Brownove modely sa dajú previesť na OpenCV Brown s chybou ≤ 0,1 px (medián). Brown k1, k2 je teda primeraný.

## 8. Porovnanie všetkých metód

Tabuľka je vygenerovaná zo súborov metód (`work/97_report_tables.py`). Všetky metódy sú aj v `results/lens_by_method.json`
(48 záznamov vrátane variantov) a na grafe `results/methods_comparison.png`. Stĺpec „Δ mapovania“ je posun
zobrazenia metódy voči hlavnému výsledku po kompenzácii rotácie (medián oblasti).

TABLE_METHODS

**Prečo sa metódy líšia – čo z toho vyplýva:**

1. **Nálepky s geometriou z výkresu verzus hrany (f ~1300–1360 vs. ~1430–1480).** Rozdiel nespôsobuje objektív, ale geometria.
   Nálepky nesedia so žiadnou dierkovou kamerou a samy vynútia prehnuté, nefyzikálne skreslenie. Keď sa uvoľní výška
   top-dosiek (+60–80 mm), dajú nálepky rovnaké skreslenie ako hrany a f = 1430–1495. Syntetický test so zdvihnutými doskami
   (scenár T2_c) reprodukuje presne tento vzor: fity len z nálepiek sú vychýlené o −115 až −143 px vo f, kombinovaný odhad nie (+0,1 px).
2. **Hrany medzi sebou (f 1425 vs. 1473–1479).** Rozdiel spôsobuje jediný predpoklad: či majú oba vozíky rovnakú zvislicu.
   Vozík 80 má len dve krátke, takmer kolineárne hrany stĺpika, takže jeho vlastná zvislica je slabo určená. Hrany samotné dávajú f
   krehko: vynechanie jednej krátkej hrany stĺpika posunie f o ~100 px.
3. **cy (426–513).** Hlavne ho určuje skupina čiar na podlahe (modrá čiara a biela páska považované za rovnobežné).
   Bez nej je cy ≈ 440–500 ± 30. Ak sa obe časti modrej čiary berú ako jedna priamka, vychádza cy = 509.
   Nálepky s uvoľnenou geometriou dávajú cy ≈ 390. **cy je z dát len konzistentné.**
4. **Zvislice scény verzus stĺpiky vozíkov** sa líšia o 0,5–0,8°. Podlaha alebo predmety nie sú presne zvislé, preto
   majú zvislice scény vo všetkých fitoch vlastný smer.

## 9. Overenie

### 9.1 Reziduá

* **Hlavný fit, bloky:** priamosť hrán 0,23 px, konzistencia s úbežníkmi 0,36 px, strany zakrytých nálepiek 1,7 px,
  rohy nálepiek 14 px na bod (v spoločnom fite). S hlavným objektívom a pózou vozíka fitovanou len z jeho nálepiek je to
  **7,6 px RMS (max 13,6 px)**, čo je nesúlad geometrie, nie šum. Tabuľka po nálepkách je nižšie, obrázky
  `results/residuals_stickers_main.png` (šípky ×10) a `results/predicted_stickers_crops.png` (výrezy: detegované zelené,
  predpovedané červené). Reziduá hrán: `results/residuals_edges_main.png`. Obraz po odstránení skreslenia je
  `results/undistorted_mean.png`: hrany políc, pásky aj modrá čiara sú vizuálne rovné.
* **Len nálepky, s geometriou z výkresu:** RMS 5,3–5,9 px (max 9,6–11,4 px) pri šume ~0,13 px (medián rozptylu medzi snímkami).
  Po vozíkoch 5,3 / 5,4 px, po radoch top 4,9 / A 8,0 / B 3,6 / C 5,4 px (`results/markers_residuals.png`).

TABLE_STICKERS

### 9.2 Profil ohniska (`results/f_profiles.png`, `results/combined_f_profile.png`)

* **Len nálepky:** RMS má minimum pri f ≈ 1310–1360 a rastie o ~1 px na každých ~100 px posunu f. Minimum je plytké a
  vychýlené geometriou.
* **Len hrany:** RMS hrán má minimum pri f ≈ 1430–1440 (so samostatnými zvislicami) a je veľmi plochý:
  +0,005 px pri ±30 px. f zo samotných hrán je slabé.
* **Kombinovaný (hlavný):** minimum χ² pri 1456. Zvýšenie χ² vážené blokmi je prudké len formálne. Reálnu neistotu
  (±34 px) určuje bootstrap, rozptyl alternatív a syntetický test.

### 9.3 Krížová validácia

* **Len nálepky** (`20_markers_ba.py`): vynechanie jednej nálepky dáva predikčnú chybu 8,4–9,1 px (v rámci fitu 5,3–5,9).
  Vynechanie radu: top 16–22 px, A 10–11, B 5–6, C 8–10 px. Vynechanie vozíka (vnútorné parametre z jedného vozíka,
  póza druhého refitovaná): 5,5–6 px pre jednoduché modely a 20–90 px pre modely s voľným pp alebo k2, ktoré sa prehýbajú.
* **Kombinovaný (hlavný), vynechanie radu / vozíka / oblasti hrán** (`51_combined_cv.py`):

TABLE_CV

* **Jackknife po hranách** (štúdia modelov, spoločné dáta): f 1413–1457 pri vynechaní jedného člena. Samotné hrany:
  vynechanie jednej krátkej hrany stĺpika vozíka 310 posunie f o ~+100 px.

### 9.4 Syntetický round-trip (`work/44_synth_*`, `work/cache/synthetic_report.md`, `results/synthetic_*.png`)

Syntetické dáta majú presne reálnu geometriu a viditeľnosť (tie isté nálepky a rohy, tie isté hrany prevedené na ideálne priamky)
a vznikli zo známeho objektívu. Použité scenáre:

* **a** – len šum detekcie;
* **b** – realistický šum: náhodná chyba polohy nálepiek 5 mm a výšky radov 33 mm, kalibrovaná tak, aby RMS nálepiek
  bolo 5,4 px ako v reálnych dátach, plus ohyby a smerové chyby hrán z reálnych reziduí;
* **c** – systematicky zdvihnuté top-dosky podľa diagnostiky.

| scenár | odhad | chyba f (bias ± scatter) | chyba zobrazenia stred / pás / rohy [px] |
|---|---|---|---|
| a (T2) | kombinovaný | +0,1 ± 0,3 | 0,1 / 0,1 / 0,3 |
| a (T2) | len nálepky k1k2 pp voľný | +0,1 ± 0,3 | – / 0,1 / 0,6 |
| b (T2) | **kombinovaný** | **+1 ± 36 (robustne 17)** | **2,9 / 10,9 / 16,1** |
| b (T2) | spoločný fit z čiar | +13 ± 26 | – / 8,9 / 13,3 |
| b (T2) | len nálepky (rôzne modely) | −19…+11 ± 65–100 | 21–62 / 45–186 v rohoch |
| c (T2) | kombinovaný | +0,1 ± 0,6 | – / 0,2 / 0,6 |
| c (T2) | len nálepky | **−115 až −143** | 32–38 / 49–97 v rohoch |

* **Postup vráti známy objektív.** Kombinovaný odhad je nevychýlený vo všetkých scenároch.
* **Formálne kovariancie podhodnocujú chybu 9–15×.** Bootstrap ju naopak prekrýva (pomer 0,4–1,0).
* **Hlavný výsledok berie konzervatívne väčšiu z dvoch hodnôt:** empirický rozpočet (bootstrap ⊕ rozptyl alternatív)
  a RMSE zo scenára b. Tak vznikli hodnoty f ± 34, k1 ± 0,021 a zobrazenie 3,2 / 10,9 / 16,1 px.
* Metóda plumb-line s pevným stredom (stred obrazu) je vychýlená, ak skutočný stred nie je v strede obrazu
  (T2: k1 o −0,018). Preto je stred skreslenia v hlavnom fite voľný (= hlavný bod).

### 9.5 Citlivosť na malé chyby detekcie rohov a hrán

Na reálnych dátach pre hlavný (kombinovaný) odhad:

| porucha | veľkosť | Δf [px] | Δcx, Δcy [px] | zmena zobrazenia stred / pás / rohy [px] |
|---|---|---|---|---|
| dilatácia alebo erózia čierneho štvorca | ±0,3 px na stranu | ∓0,9 | < 0,1 | 0,07 / 0,3 / 0,4 |
| posun všetkých rohov v x alebo y | 0,2 px | < 0,3 | < 0,02 | < 0,1 |
| zvislá mierka rohov (rolling-shutter) | 5·10⁻⁴ | +0,05 | < 0,03 | < 0,03 |
| zvislá mierka celého obrazu | 5·10⁻⁴ | +2,0 | 0,3; 1,1 | 0,14 / 0,6 / 1,0 |
| ArUco rohy namiesto hranových | 0,44 px RMS | −0,04 | < 0,01 | < 0,02 |
| rohy opravené o posun hrán k čiernej | 0,96 px RMS | −0,04 | < 0,01 | < 0,02 |
| hrany posunuté k tmavej strane | ±0,3 px | ±0,9 | ±0,3 (y) | 0,07 / 0,27 / 0,47 |
| hrany posunuté von od stredu | 0,3 px | −0,7 | −0,3; +1,0 | 0,10 / 0,22 / 0,27 |

**Chyby detekcie sú zanedbateľné** voči modelovej a geometrickej neistote: menej než 1 px zobrazenia pri poruchách
0,3 px. Pri samotných nálepkách sú citlivosti väčšie (až 2–3 px zobrazenia v páse, desiatky px v rohoch pri
prehnutých modeloch).

## 10. Neistota zobrazenia

`results/mapping_uncertainty.png` ukazuje mapu 1σ posunu priemetu po celom obraze v troch zložkách: štatistická
(bootstrap: prevzorkovanie nálepiek a hrán v rámci skupín úbežníkov, 36 replík, každá s celým fitom), systematická
(rozptyl 11 prijateľných alternatívnych metód a modelov voči hlavnému výsledku) a celková.

| oblasť | štatistická | systematická | rozpočet spolu | syntetický test (scenár b) | **výsledok** |
|---|---|---|---|---|---|
| stred | 1,6 | 2,7 | 3,2 | 2,9 | **3,2 px** |
| pás vozíkov | 5,5 | 7,1 | 9,0 | 10,9 | **10,9 px** |
| rohy | 7,9 | 11,0 | 13,5 | 16,1 | **16,1 px** |

* Definícia: posun priemetu pevného lúča po kompenzácii rotácie kamery. Čistú rotáciu pohltí póza, takže
  ide o chybu, ktorá sa prejaví pri súčasnom odhade pózy. Bez kompenzácie rotácie sú čísla väčšie
  (hlavne v strede, kde sa prejaví posun hlavného bodu).
* Dominuje neistota f (1 % vo f ≈ 7 px v páse vozíkov, ≈ 10 px v rohoch). Samotné skreslenie (pri pevnom f) je
  určené na ~2 / 1,3 / 5 px (plumb-line).
* Modely ohodnotené ako prehnuté (všetky modely len z nálepiek s výkresom) nemajú v rohoch zmysluplné zobrazenie.
  Ich mapovanie sa v štatistike nepoužíva.

## 11. Čo je určené, čo len konzistentné, čo sa určiť nedá

**Určené z dát:**
* Radiálny profil skreslenia (k1 a k2 spolu, v pixelových jednotkách k1/f² = −1,63·10⁻⁷, k2/f⁴ = 2,1·10⁻¹⁴):
  z priamosti 61 hrán s rezíduom 0,22 px, zhodne v dvoch nezávislých implementáciách.
* cx = 928 ± 12 px.
* f = 1456 ± 34 px, ale len spojením nálepiek a hrán (hrany samé ±50–120 px, nálepky samé sú vychýlené).
* Neplatnosť geometrie výkresu voči obrazu: nezávisí od objektívu (dvojpomer, všeobecná projektívna kamera).

**Len konzistentné s dátami:**
* cy v rozsahu ~426–513 px podľa predpokladu o čiarach na podlahe (hlavná hodnota 502 ± 35).
* k3 ≈ −0,02…−0,03 (nevýznamné zlepšenie).
* Konkrétna fyzikálna príčina nesúladu: top vyššie o 60–80 mm, alebo police nižšie; sklony dosiek 4–13°.

**Z dát sa určiť nedá:**
* p1, p2 (tangenciálne) a fx ≠ fy (pomer strán pixla); predpokladá sa 0 a 1.
* Či je posunutá top-doska, alebo police (pozorovateľný je len relatívny posun).
* Skreslenie za rohmi obrazu (extrapolácia), ani to, či má objektív skutočne tvar Brown k1, k2, alebo iný
  dvojparametrový radiálny tvar. V obraze sa líšia o ≤ 2,7 px a tento rozdiel je v systematike.

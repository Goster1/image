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

## 2. Dáta a predspracovanie

* **Vstup:** 7 fotiek 1920×1080 (`data/stills/`), statická scéna, pevná kamera. ChArUco fotky nie sú priložené
  (`data/charuco_photos/` je prázdny), takže kalibrácia stojí len na vozíkoch a scéne.
* **Chvenie medzi snímkami (rolling-shutter / vibrácie):** snímky sa líšia o zvislý posun a zvislú mierku,
  `dy = a_f + b_f·(y−540)` s |a_f| ≤ 1 px a |b_f| ≤ 6·10⁻⁴; `dx` je ~0,05 px (`work/02_initial_calib.py`).
  Každú snímku som týmto modelom zarovnal na spoločný priemer a vytvoril priemerný obraz
  (`work/03_mean_image.py`). Po kompenzácii je rozptyl rohov medzi snímkami ~0,13 px (medián), priemer 7 snímok má teda šum ~0,05 px.
  Neznáma absolútna deformácia priemerného snímku je ~1,5·10⁻⁴ vo zvislej mierke (≈ 0,2 px vo fy), čo je zanedbateľné.
* **Súradnice:** OpenCV konvencia (stred ľavého horného pixla = (0,0)); model pinhole + Brown-Conrady,
  `dist = [k1, k2, p1, p2, k3]`.

## 3. Detekcia nálepiek (`work/01_*`, `work/11_markers_*`, `work/12_review_markers*`)

* **Všetkých 20 nálepiek** (2 vozíky × id 0–8 + unikátne 92/322) je nájdených. 12 nájde bežný ArUco
  detektor, 8 ďalších (šikmé, čiastočne zakryté) sa našlo riadenou detekciou: predikcia polohy,
  vyrovnanie výrezu podľa tvaru susednej nálepky a vlastné čítanie bitovej mriežky 6×6 len z viditeľných buniek.
  **ID každej nálepky je potvrdené z obsahu obrazu** (Hammingova vzdialenosť 0 k očakávanému kódu na ≥ 9 plne
  viditeľných bitoch; test proti všetkým 1000 kódom × 4 natočenia). Pri 310:5 a 310:7 je jednoznačnosť len medzi id
  použitými na vozíkoch (9 plne viditeľných bitov), čo je uvedené v `results/detections.json`.
* **Rohy:** každú stranu čierneho štvorca som sub-pixelovo dohľadal pozdĺž normály a rohy vypočítal ako priesečníky
  priamok strán (vzniká menší bias než pri cornerSubPix). Výsledok je priemer zo 7 zarovnaných snímok. Roh je platný len vtedy, keď sú obe jeho strany viditeľné
  a rovné. **Platných rohov je 62 z 80**, 18 je zakrytých (hranou dosky, susednou kartičkou, stĺpikom, lemom police).
* **Šum a systematika detekcie:**
  - rozptyl rohu medzi snímkami: medián 0,13 px (hranová metóda), 0,20 px (ArUco cornerSubPix);
  - hranové rohy vs. ArUco rohy: RMS 0,44 px, v priemere 0,16 px dovnútra po uhlopriečke;
  - **systematický posun hrán k čiernej strane o 0,66 ± 0,23 px** (tónová krivka alebo doostrenie kamery). Zmeral som ho na vnútornej
    mriežke kódu (hrany buniek musia ležať na mriežke 15 mm), vo všetkých snímkach je rovnaký. Opravené rohy sú v
    `corners_px_bias_corrected`. Na výsledok nálepkových fitov má malý vplyv (RMS 5,34 → 5,26 px, f bez zmeny).
* **Natočenie v rovine (rot_k):** top-nálepky majú 0, nálepky na policiach 2. Druhé najlepšie natočenie je horšie o ≥ 25 px RMS, takže voľba je jednoznačná.
* Súbor `results/detections.json` obsahuje všetko na zopakovanie výpočtu: rohy (v spoločnom snímku aj v súradniciach
  jednotlivých fotiek), priradenie rohov k fyzickým rohom (`corners_3d_mm`, vzorec cez rot_k), platnosť, rozptyly
  a spôsob potvrdenia id. Obrázky: `results/markers_overlay.png`, `markers_crops.png`, `markers_rect_views.png`,
  `markers_edge_bias.png`.

## 4. Hrany konštrukcie (`work/10_edges_*`, `work/12_review_edges_*`, `work/13_edges_dedupe.py`, `work/91_merge_edges.py`)

* Hrany som hľadal v priemernom obraze: kandidáti z LSD a Canny reťazcov, potom sub-pixelové dohľadanie pozdĺž normály
  (extrém gradientu + parabola, iteratívne podľa hladkej krivky). Skreslené priamky sú v obraze krivky,
  preto ich nikde nenútim do priamky v pixlovom priestore.
* **Každú hranu som overil na zväčšených výrezoch** (priebeh hrany, zväčšené okná, narovnaný pás) a potom ju nezávisle
  zrevidoval druhý prechod: či ide o konštrukciu a nie o tovar, tieň, odlesk alebo potlač, či body neskáču na susednú hranu,
  orezanie okolo držiakov etikiet a pod.
* Po revízii a odstránení duplicít medzi oblasťami (modrá čiara na podlahe a zadná tyč vozíka 80 boli v dvoch súboroch)
  zostalo **61 rovných hrán v 3D**:
  - vozík 310: 18 (predné lemy políc A–E, zadné pozdĺžniky, koncové dosky, 2 stĺpiky),
  - vozík 80: 17,
  - scéna: 26 (biela páska a modrá čiara na podlahe, škára, zvislé hrany modrého objektu a stĺpika, rúry a rámy na
    pravej strane, rúrka regálu pri ľavom okraji).
* **Vyradené ako nerovné:** priečne priehradky políc (y_lip*, y_div*), vnútorná hrana dosky 80/X0 a ďalšie.
  Nezávislá implementácia vyradila aj oblúkový lem E vozíka 80 (0,74–0,80 px, fyzicky ohnutý).
* **Hrany koncových dosiek sú rovné, ale nie sú rovnobežné s osami vozíka na presnosť úbežníkov** (napr. predná hrana dosky
  310/X0 je o 5,9° mimo úbežníka piatich lemov políc). Preto ich v skupinách úbežníkov nepoužívam, slúžia len na priamosť.
* Výstupy: `results/edges.json` (všetky hrany vrátane vyradených, s dôvodmi), `results/edges_cart310.png`,
  `edges_cart80.png`, `edges_scene.png`, `edges_all.png`; výrezy v `work/cache/edgecrops_*`, `review_*`.

## 5. Nesúlad nálepiek s výkresom – pozorovanie (`work/42_geomdiag_*`, `work/41_cuboid_*`)

Podľa zadania sú rozmery vozíka a polohy nálepiek presné. Hlavný výsledok ich preto nemení a všetko nižšie je
**len pozorovanie**. Zisťuje, kde a o koľko sa dáta s výkresom rozchádzajú; žiadny rozmer tým neupravujem.

**Pozorovanie (nezávisí od ohniska, hlavného bodu ani pózy):**

1. **Geometria z výkresu nesedí so žiadnou dierkovou kamerou.** Všeobecná projektívna kamera 3×4 na každý vozík
   (11 stupňov voľnosti, ľubovoľné vnútorné parametre, skreslenie z plumb-line) dáva RMS 4,9–5,1 px pri šume
   ~0,1–0,2 px. Úplný metrický fit dáva 5,3 px (max 9,6 px). Nie je to teda problém modelu objektívu.
2. **Každá nálepka je sama osebe čistý rovný štvorec:** s voľnou pózou pre každú nálepku RMS klesne na 0,17 px.
   Detekcia je teda v poriadku; nesedí vzájomná poloha a orientácia nálepiek.
3. **Dvojica nálepiek na jednej koncovej doske** sedí s výkresom na 0,22–0,39 px, ak sa použijú rohy opravené o
   posun hrán (bez opravy vychádza zdanlivý kód 86,6–88 mm namiesto 90 mm).
4. **Nálepky nad sebou (top, A, B, C na rovnakom X, Y) nie sú kolineárne:** odchýlky 0,23–2,55 px pri šume
   0,10–0,20 px, čo zodpovedá bočnému rozptylu ~1–6 mm.
5. **Dvojpomer (cross-ratio) stĺpcov, čistý projektívny invariant:** top-nálepka vychádza o **+50 až +85 mm vyššie**
   voči nálepkám na policiach A, B, C na troch koncoch (80/X1600: +50/+64; 310/X0: +75; 310/X1600: +55/+85 mm).
   Na konci 80/X0 dvojpomer sedí (−7/0 mm), ale nesedí tam test s úbežníkom stĺpikov
   (A−B : B−C = 1,27 namiesto 1,00).

**Kde a koľko px** (diagnostické fity s jedným uvoľneným prvkom, ostatné podľa výkresu):

| prvok | výkres | diagnostika | posun rohov | kde v obraze |
|---|---|---|---|---|
| výška koncovej dosky 80/X0 | 0 mm | +64 ± 8 mm (dvojpomer −7…0) | 16–19 px | x 1404–1656, y 25–116 (vpravo hore) |
| výška koncovej dosky 80/X1600 | 0 mm | +71 ± 5 mm (dvojpomer +50…+64) | 24 px | x 1288–1516, y 899–952 (vpravo dole) |
| výška koncovej dosky 310/X0 | 0 mm | +61 ± 6 mm (dvojpomer +63…+75) | 20 px | x 563–807, y 886–977 (vľavo dole) |
| výška koncovej dosky 310/X1600 | 0 mm | +81 ± 8 mm (dvojpomer +55…+85) | 20–23 px | x 226–480, y 31–178 (vľavo hore) |
| sklon koncových dosiek pozdĺž X | 0° | 4–13° (vonkajší koniec vyššie) | 0,7–4 px na roh | horný a dolný okraj obrazu |
| bočný rozptyl nálepiek nad sebou | 0 mm | 1–6 mm | 0,2–3 px | všetky 4 stĺpce |
| zadná horná tyč vozíka 80 (model s výkresom) | Z = 0 | −56 mm | −17 px | pravý vozík |

* Najkompaktnejšie vysvetlenie je, že **top-nálepky (koncové dosky) sú o +60 až +80 mm vyššie voči nálepkám na policiach**,
  alebo ekvivalentne, že police sú nižšie. Medzi týmito dvoma možnosťami sa z dát rozhodnúť nedá.
  Samotný tento prvok zníži RMS z 5,3 na 2,9 px. Keď k tomu uvoľním aj sklony dosiek, klesne na 1,9 px.
  Žiadny kompaktný prvok nezníži RMS na úroveň šumu, časť nesúladu je nálepka po nálepke (kartičky, ohnuté dosky).
* Na výrezoch vidno, že koncové dosky sú tenké (~10 mm) biele dosky s potlačeným hárkom, nie krabice.
  60–90 mm vysoká bočná stena by mala 20–35 px a nie je viditeľná. Zadné rohy dosiek vozíka 80 sú zdvihnuté, doska
  310/X0 má previsnutú chlopňu. Fyzickú príčinu (rám, výšky políc, uchytenie) z fotiek určiť nemožno.
* **Dôsledok pre objektív:** nálepky s geometriou z výkresu vynútia f ≈ 1300–1360 px, cy ≈ 480 a skreslenie
  k1 ≈ −0,09, k2 ≈ −0,23 (pri f0 = 1300). Také skreslenie **sa prehne ešte vo vnútri obrazu** (nie je invertovateľné po
  rohy) a odporuje priamosti 61 hrán. Keď sa uvoľní výška dosiek, nálepky samé dajú k1 = −0,295, k2 = 0,105 (pri 1300),
  teda prakticky rovnaké skreslenie ako plumb-line (−0,29; 0,07), a f ≈ 1430–1495.
  **Nálepky a hrany sa teda zhodnú na objektíve, len ak sa pripustí táto odchýlka geometrie.**
  Obrázky: `results/geomdiag_*.png`, `results/cuboid_overlay.png`, `results/cuboid_offsets.png`,
  `results/residuals_stickers_main.png`, `results/predicted_stickers_crops.png`; podrobne `work/cache/geometry_diagnosis.md`,
  `work/cache/cuboid_offsets.md`.

## 6. Metódy (každá dotiahnutá do čísla s neistotou)

Všetky rezíduá hrán sa merajú **v skreslenom obraze**. Rezíduum je vzdialenosť pozorovaného bodu od dopredne
skreslenej priamky ("forward"). Meranie v neskreslenom obraze by šum systematicky ťahal k slabšiemu skresleniu
(syntetický test: pri σ = 0,2 px je k1 vychýlené o +0,01 až +0,04). Každý fit má navyše **bariéru proti prehnutiu**:
radiálne skreslenie musí byť monotónne až po rohy obrazu, inak nie je invertovateľné.
Všetky modely z nálepiek s geometriou výkresu sú prehnuté (fold margin −0,03 až −0,27).

| # | metóda | čo používa | skript |
|---|---|---|---|
| M1 | nálepky (BA) | 62 rohov, rozmery z výkresu; 2 pózy vozíkov | `20_markers_ba.py` |
| M2 | plumb-line | priamosť 61 hrán (len skreslenie a jeho stred) | `30_lines_methods.py`, `40_lines_indep_a_plumb.py` |
| M3 | úbežníky | smery osí X/Y/Z vozíkov, zvislice scény, rovnobežné čiary na podlahe; kolmosť → f, pp | `30_lines_methods.py`, `40_lines_indep_b_vp.py` |
| M4 | spoločný fit z čiar | M2 + M3 naraz (bez rozmerov) | `30_lines_methods.py`, `40_lines_indep_c_joint.py` |
| M5 | model vozíka (kváder s policami) | nálepky + presné hrany modelu (výkres) + kontrola predikcie všetkých hrán | `41_cuboid_*.py` |
| M6 | **kombinovaný (hlavný)** | nálepky (výkres, robustné váhy po nálepkách) + hrany (úbežníky vrátane podlahy, priamosť) + viditeľné strany čiastočne zakrytých nálepiek | `50_combined.py`, `combined.py` |
| M7 | dva vozíky na jednej podlahe | súhlas normál podlahy z póz oboch vozíkov ako funkcia f | `42_geomdiag_d_twocart.py` |
| M8 | tvar nálepiek | každá nálepka ako vodorovný štvorec (bez polôh a výšok) + nadir | `42_geomdiag_e_markershape.py` |
| M9 | iné modely objektívu | Brown (rôzne sady), racionálny, Kannala-Brandt, divízny; tie isté dáta | `43_altmodels_*.py` |

Poznámky k jednotlivým metódam:

* **M1 – nálepky:** výber modelu pomocou leave-one-out (najlepší model s LOO 8,2 px), profil f,
  klastrový bootstrap po nálepkách, vynechanie radu a vozíka. Všetky modely majú RMS 5,0–5,7 px a sú prehnuté vo vnútri obrazu.
  Výsledok je **vychýlený** geometriou (kap. 5), nie zašumený.
* **M2 – plumb-line:** skreslenie je určené v pixelových jednotkách (k1/f², k2/f⁴), f z priamosti určiť nemožno.
  Model k1 samotný nestačí (0,51 px, naráža na bariéru). Pre k1, k2 (voľný stred) je rezíduum 0,217 px,
  čo zodpovedá vlastnej nerovnosti hrán. Stred skreslenia je v x určený (924–938), v y len voľne (423–516 podľa modelu).
* **M3 – úbežníky:** VP skupín s ML smerom; smerová chyba jedného člena ~0,16°, čo je ďaleko nad šumom bodov.
  Kolmosť osí vozíka a spoločná zvislica oboch vozíkov (stĺpiky sa zhodujú na 0,07–0,38°) dávajú f.
  Hlavný bod zo samotných úbežníkov určiť nemožno (berie sa stred skreslenia). Zvislice scény sa od stĺpikov vozíkov
  líšia o 0,5–0,8° (podlaha nie je dokonale vodorovná / predmety nie sú presne zvislé), preto majú vlastný smer.
* **M4 – spoločný fit z čiar:** f závisí od toho, či majú oba vozíky spoločnú zvislicu: so spoločnou 1473 px,
  so samostatnými 1425–1446 px, keďže vozík 80 má len dve krátke hrany stĺpika. Vynechanie jednej krátkej hrany stĺpika
  vozíka 310 posunie f o ~100 px, takže **f zo samotných hrán je krehké** (jackknife σ ≈ 150 px).
* **M5 – kváder:** s rozmermi z výkresu f = 1359 ± 58 px, rezíduá 6 px; lemy políc sedia na −2…+9 px.
  Zadná horná tyč vozíka 80 je −17 px (−56 mm) mimo; po uvoľnení výšky top-dosky (+65 mm) na +1,5 px.
* **M6 – kombinovaný:** blokové váhy sa určujú variančnými komponentmi (každý blok má redukované χ² ≈ 1).
  Nálepky majú robustnú váhu 1/(1+(RMS_nálepky/1,5 px)²), takže nálepky, ktoré nesedia s ostatnými dátami,
  sa samy potlačia. Váhy aj rezíduá sú v tabuľke (kap. 9). Rozmery z výkresu sa nemenia.
* **M7, M8:** len konzistenčné kontroly; M8 je na f príliš slabá (± 170 px).
* **M9:** dáta podporujú práve 2 radiálne stupne voľnosti. k3 je len konzistentné, p1, p2 a fx ≠ fy nie sú určené.
  Voľba medzi primeranými modelmi pridá 0,4 / 2,0 / 2,7 px (stred / pás / rohy). Nie-Brownove modely sa dajú
  previesť na Brown [k1, k2, p1, p2, k3] s chybou pod 0,1 px (medián).

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
zobrazenia metódy voči hlavnému výsledku po kompenzácii rotácie (medián oblasti). Uvedené ± sú neistoty, ktoré udáva sama metóda.

| metóda | rozmery z výkresu | f [px] | cx | cy | k1 | k2 | k3 | fold | Δ mapovania vs hlavný: stred / pás / rohy [px] | neistota zobrazenia metódy (stred / pás / rohy) | pozn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M6 kombinovaný – HLAVNÝ | áno (všetky nálepky, plná váha) | 1456 ± 14 stat. (± 34 celkovo) | 928 ± 5 (± 12) | 502 ± 9 (± 35) | -0.345 | +0.096 | +0.000 | ok | 0.0 / 0.0 / 0.0 | 1.7 / 6.0 / 9.0 | nálepky + 61 hrán + strany zakrytých nálepiek |
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
| M1 len nálepky (výkres), model s najlepším LOO | áno | 1326 ± 20 | 960 | 540 | -0.246 | +0.025 | +0.000 | na bariére | 9.0 / 39.3 / 50.1 | 2.9 / 15.4 / 36.9 | vychýlené; bez bariéry prehnuté |

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

Reziduá nálepiek s hlavným objektívom (geometria z výkresu, póza vozíka robustne z jeho nálepiek; `work/cache/final_sticker_residuals.json`):

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

| záhyb | f | cx | cy | k1 | k2 | predikcia vynechaných dát |
|---|---|---|---|---|---|---|
| vynechaný rad top (8 nálepiek) | 1474 | 929 | 504 | -0.355 | +0.101 | rohy vynechaných nálepiek: 22.6 px RMS (v rámci fitu ~7,6) |
| vynechaný rad A | 1428 | 928 | 501 | -0.333 | +0.089 | rohy vynechaných nálepiek: 9.0 px RMS (v rámci fitu ~7,6) |
| vynechaný rad B | 1474 | 930 | 505 | -0.356 | +0.102 | rohy vynechaných nálepiek: 10.0 px RMS (v rámci fitu ~7,6) |
| vynechaný rad C | 1436 | 927 | 501 | -0.335 | +0.090 | rohy vynechaných nálepiek: 8.7 px RMS (v rámci fitu ~7,6) |
| vynechané nálepky vozíka 80 | 1437 | 927 | 502 | -0.336 | +0.091 | nálepky vozíka, len póza refitovaná: 7.5 px RMS |
| vynechané nálepky vozíka 310 | 1483 | 927 | 503 | -0.357 | +0.102 | nálepky vozíka, len póza refitovaná: 7.8 px RMS |
| vynechané hrany vozíka 310 | 1484 | 911 | 497 | -0.343 | +0.087 | priamosť vynechaných hrán: 0.251 px (v rámci fitu 0,231) |
| vynechané hrany vozíka 80 | 1431 | 943 | 503 | -0.336 | +0.092 | priamosť vynechaných hrán: 0.253 px (v rámci fitu 0,231) |
| vynechané hrany scény | 1473 | 934 | 533 | -0.365 | +0.117 | priamosť vynechaných hrán: 0.223 px (v rámci fitu 0,231) |
| **rozptyl cez záhyby (std)** | **23** | 8 | 10 | 0.011 | 0.009 | |

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

## 12. Čo by meranie najviac spresnilo

Poradie podľa očakávaného prínosu:

1. **Fotky kalibračnej dosky ChArUco** (`spec/charuco-board`) touto kamerou, 15–30 záberov. Doska by mala pokryť celé pole,
   hlavne rohy a okraje, a byť v rôznych sklonoch (±30–45°) a vzdialenostiach. To je jediná cesta, ako určiť f a hlavný bod
   na ±1–2 px a skreslenie až do rohov, a nezávisí od geometrie vozíkov. Pri webkamere treba zapnúť pevné zaostrenie
   a snímať v tom istom režime (1920×1080, rovnaký softvér) ako stanica.
2. **Zmerať skutočnú geometriu nálepiek** (meter alebo laserový diaľkomer): výšku povrchu koncových dosiek
   voči policiam A–C na všetkých 4 koncoch vozíkov, sklon a rovinnosť dosiek, polohu kartičiek na policiach.
   Dáta ukazujú odchýlku +60–80 mm a sklony 4–13° (kap. 5). S opravenou geometriou by nálepky samy dali f
   s presnosťou ~±15 px a hlavný bod.
3. **Pridať priame 3D referencie so známymi rozmermi v rohoch obrazu:** napríklad dlhé rovné latky alebo pásky na podlahe
   v dvoch kolmých smeroch cez celé zorné pole (priamosť + kolmosť = úbežníky) a zvislé tyče pri okrajoch.
   Dnes sú rohy obrazu pokryté slabo (vľavo hore regál, hore v strede pohyblivá osoba), hlavný bod v y určuje
   najmä jedna skupina čiar na podlahe.
4. **Nálepky na pevných, rovných podložkách** namiesto papiera na doskách, viac nálepiek na rôznych výškach a
   ďalej od stredu (rohy obrazu); ideálne aj zvisle, aby sa zlepšila kondícia f.
5. **Viac záberov s pohybom vozíka** (ten istý vozík v rôznych polohách a natočeniach). Každá póza pridá nezávislú
   perspektívu a zmenšia sa korelácie f–pp–pózy.
6. **Stabilizácia kamery / snímanie videa bez chvenia:** snímky sa líšia o ±1 px vo zvislom smere (rolling-shutter
   a vibrácie). Priemerovanie viacerých snímok alebo pevnejšie uchytenie znížia šum hrán aj rohov.
7. **Vypnúť doostrenie alebo tónové úpravy vo webkamere, ak sa dá:** hrany sú systematicky posunuté o ~0,66 px k tmavej strane.
   Na priamosť to vplyv nemá, ale na absolútne polohy rohov áno.

## 13. Postup a reprodukovateľnosť

Prostredie: Python 3, `pip install -r requirements.txt`. Použité verzie: OpenCV 5.0 (contrib, headless), numpy 2.4, scipy 1.17,
matplotlib 3.11. Všetky skripty sa spúšťajú z adresára `work/`. Priebežné súbory sú v `work/cache/` (veľké obrázky a `.npy`
sa necommitujú a vytvoria sa znova v krokoch 1–3).

| krok | skript(y) | výstup |
|---|---|---|
| 1 | `01_detect_markers_initial.py`, `02_initial_calib.py` | prvá detekcia, chvenie snímok, počiatočný fit |
| 2 | `03_mean_image.py` | priemerný obraz so skompenzovaným chvením |
| 3 | `11_markers_a_guided.py`, `11_markers_b_corners.py`, `11_markers_c_fit.py`, `11_markers_d_diag.py`, `12_review_markers_check.py`, `12_review_markers.py`, `12_review_markers_plot.py` | `results/detections.json`, `markers_*.png` |
| 4 | `10_edges_cart310_trace.py` (+ `_summary`, `_boardcheck`), `10_edges_cart80_trace.py`, `10_edges_scene_trace.py` / `_final.py`, `12_review_edges_{cart310,cart80,scene}.py`, `13_edges_dedupe.py`, `91_merge_edges.py` | `results/edges.json`, `edges_*.png` |
| 5 | `20_markers_ba.py` | metóda M1 |
| 6 | `30_lines_methods.py 40` (orchestrátor) a `40_lines_indep_{a,b,c,d,e,g,h,i,j,f}_*.py` (nezávislá implementácia) | M2–M4 |
| 7 | `41_cuboid_{fit,predict,diag}.py` | M5 |
| 8 | `42_geomdiag_{a,b,c,f,d,e,g,h}_*.py` | diagnostika geometrie, M7, M8 |
| 9 | `43_altmodels_{fit,diaggeom,report,review}.py` | M9 |
| 10 | `run_combined.sh` (= `50_combined.py fit`, 3× `boot`, `finalize`), `51_combined_cv.py all` | hlavná metóda M6 |
| 11 | `44_synth_{a_setup,b_roundtrip,c_sensitivity,d_report}.py` | syntetický test a citlivosti |
| 12 | `92_extract_alternatives.py`, `95_final.py`, `96_plots.py`, `98_finalize_json.py`, `90_compare_methods.py`, `97_report_tables.py` | `results/lens_result.json`, `lens_alternatives.json`, `lens_by_method.json`, grafy, tabuľky |

Zdieľané moduly: `common.py` (model kamery, špecifikácia, kontrola prehnutia), `calib.py` (BA), `edgelib.py` (sub-pixelové hrany),
`lineselfcal.py` (plumb-line, úbežníky, spoločný fit z čiar), `combined.py` (spoločný odhad), `evaltools.py`
(neistota zobrazenia), `markerdata.py`, `linedata.py`, `altmodels.py`, `synthlines.py`.

**Rozhodnutia a predpoklady (zapísané podľa zadania):**
* **Chvenie medzi snímkami:** modelujem ho ako zvislý posun a mierku pre každú snímku a pracujem s priemerom 7 snímok.
* **Rezíduá hrán:** merajú sa v skreslenom obraze (forward). Model nesmie byť prehnutý vo vnútri obrazu (bariéra).
* **Hrany koncových dosiek** sa nepoužívajú v úbežníkoch (nie sú rovnobežné s osami vozíka), len na priamosť.
* **Čiary na podlahe:** biela páska (obe hrany) tvorí jednu skupinu. Modrá čiara vľavo, biela priečna páska
  a modrá čiara nad ňou sa považujú za rovnobežné, čo určuje cy (variant bez tohto predpokladu je v systematike).
* **Zvislice scény** majú vlastný smer, nie sú viazané na stĺpiky vozíkov.
* **Váhy v hlavnom fite:** blokové variančné komponenty. Všetky nálepky majú plnú váhu (geometria z výkresu). Robustná
  varianta (top-nálepky prakticky vyradené, f = 1475) je alternatíva a patrí do systematiky.
* **Neistoty:** konzervatívne väčšia z hodnôt (rozpočet: bootstrap ⊕ alternatívy; syntetický round-trip, scenár b).

## 14. Súbory vo `results/`

* **Výsledky:** `lens_result.json` (hlavný), `lens_alternatives.json` (6 rovnocenne podporených variantov), `lens_by_method.json`
  (každá metóda a variant zvlášť, pole `method`).
* **Detekcie a hrany:** `detections.json` (rohy nálepiek s priradením k fyzickým rohom), `edges.json` (všetky hrany
  s bodmi a overením).
* **Obrázky:**
  - hrany: `edges_all.png`, `edges_cart310.png`, `edges_cart80.png`, `edges_scene.png`;
  - nálepky: `markers_overlay.png`, `markers_crops.png`, `markers_rect_views.png`, `markers_edge_bias.png`;
  - reziduá a predikcia: `residuals_stickers_main.png`, `predicted_stickers_crops.png`, `residuals_edges_main.png`,
    `markers_residuals.png`, `markers_complete_residuals.png`;
  - profil ohniska: `f_profiles.png`, `combined_f_profile.png`, `markers_f_profile.png`, `cuboid_f_profile.png`;
  - neistota a porovnanie: `mapping_uncertainty.png`, `methods_comparison.png`, `undistorted_mean.png`;
  - diagnostika geometrie: `geomdiag_*.png`, `cuboid_*.png`;
  - čiarové metódy (nezávislá implementácia): `40_lines_indep_*.png`;
  - modely objektívu: `altmodels_*.png`;
  - syntetický test a citlivosti: `synthetic_*.png`.

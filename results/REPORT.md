# Nezávislé (slepé) určenie parametrov objektívu staničnej kamery

Kamera Tracer WEB007, obraz 1920×1080, 7 fotiek z 2026-08-11 08:58. Všetko je odvodené len z dát v tomto repozitári.
Údaj výrobcu ani žiadne iné čísla zvonka som nepoužil. Kód je v `work/`, výstupy v `results/`.

## 1. Výsledok

**Hlavný výsledok** (`results/lens_result.json`), konvencia OpenCV (pinhole + Brown-Conrady):

| parameter | hodnota | 1σ | stav |
|---|---|---|---|
| fx = fy | **1468,6 px** | ± 47 px (3,2 %) | určené, ale len spojením hrán a nálepiek (kap. 6, M6); samotné hrany dávajú f krehko |
| cx | **928,2 px** | ± 19 px | určené (vynechanie hrán vozíka 310 / 80 posunie cx o −17 / +14 px) |
| cy | **503,2 px** | ± 29 px | len konzistentné (nesú ho úbežníky; rozumné alternatívne predpoklady o čiarach na podlahe a zvisliciach dávajú 426–513) |
| k1 | **−0,352** | ± 0,021 | určené spolu s k2 (korelácia −0,88) |
| k2 | **+0,099** | ± 0,022 | určené spolu s k1 |
| p1, p2 | 0 (fixované) | – | dáta ich neurčia (voľné iba pohltia nesúlad geometrie) |
| k3 | 0 (fixované) | – | len konzistentné (−0,03 ± 0,025), zlepšenie nevýznamné; vplyv je v systematike |

`K = [[1468.63, 0, 928.21], [0, 1468.63, 503.23], [0, 0, 1]]`, `dist = [-0.3516, 0.0994, 0, 0, 0]`.
V pixelových jednotkách, ktoré nezávisia od f: k1/f² = −1,63·10⁻⁷ px⁻², k2/f⁴ = 2,14·10⁻¹⁴ px⁻⁴.
Model je invertovateľný až po rohy obrazu (radiálna funkcia nie je prehnutá). k1 a k2 sú silno korelované: na ďalšie výpočty
treba použiť kovarianciu (`uncertainty_details.covariance` v `lens_result.json`) alebo priamo neistotu zobrazenia nižšie,
nie jednotlivé σ ako nezávislé.

**Neistota zobrazenia** (1σ posun priemetu pevného lúča po kompenzácii rotácie kamery; medián oblasti / max):

| oblasť | 1σ [px] | max v oblasti [px] |
|---|---|---|
| stred (r < 150 px) | **4,0** | 5,4 |
| pás vozíkov | **15,3** | 22,8 |
| rohy (200×150 px) | **21,6** | 23,6 |

Pri použití objektívu so súčasne odhadovanou pózou určuje veľkosť chyby hlavne neistota f: 1 % vo f posunie zobrazenie
o ~4,6 px v páse vozíkov (medián, max 6,4) a ~6,2 px v rohoch. Ak by sa lúče porovnávali
bez nového odhadu pózy (bez kompenzácie rotácie), je neistota 32 / 36 / 40 px (stred / pás / rohy),
lebo vtedy sa naplno prejaví neistota hlavného bodu (kap. 10).

**Najdôležitejšie zistenie:** geometria nálepiek podľa výkresu **nie je konzistentná so žiadnou dierkovou kamerou**
(RMS 4,9–5,1 px aj pri úplne voľnej projektívnej kamere 3×4, šum detekcie je ~0,15 px). Top-nálepky vychádzajú voči
nálepkám na policiach o ~60–80 mm vyššie a koncové dosky sú naklonené (kap. 5, diagnostické fity; dvojpomer dáva +50…+85 mm
na troch zo štyroch koncov, na konci 80/X0 0 mm, kde však nesedí pomer rozostupov políc). Podľa zadania rozmery neupravujem:
hlavný výsledok používa **všetkých 20 nálepiek s geometriou z výkresu spolu so 61 overenými hranami konštrukcie**
(tie nepoužívajú žiadne rozmery), váhy blokov dát sú určené z ich vlastného rozptylu (variančné komponenty).
Nálepky samy by s výkresom dali f ≈ 1300–1375 a cy 290–540 podľa modelu, so skreslením, ktoré odporuje priamosti hrán.
Tento výsledok je vychýlený (kap. 5, 9.4) a uvádzam ho len v porovnaní. Top-nálepky sú navyše merateľne naklonené,
vonkajší koniec dosky vyššie (zmerané z tvaru nálepiek): 80/X0 +13,7°, 80/X1600 +5,7°, 310/X0 +13,6°, 310/X1600 +2,4° – nevýznamné. Prepočet so sklonom top-nálepiek (viac uhlov, kap. 5.1) objektív prakticky nemení:
f 1467,2–1469,1 px, posun zobrazenia ≤ 0,45 px v páse vozíkov.

**Čo v hlavnom fite nesie ktorý parameter** (kap. 6, M6, 9.2): skreslenie nesie priamosť hrán. Hrany samé (úbežníky
+ priamosť, bez rozmerov) dávajú f = 1436 px; na 1469 px ho posunú strany čiastočne zakrytých nálepiek
(samostatne minimum pri ~1480 px), z toho +26 px len dve strany na konci 80/X0 (bez nich f = 1443),
kde podľa kap. 5 nesedí pomer rozostupov políc. Úbežníky sú nutné (bez nich f = 1405, cy = 427) a nesú cy;
62 rohov nálepiek má na f malý vplyv (bez nich f = 1471), lebo ich blok má kvôli nesúladu geometrie malú váhu
(σ ≈ 10,7 px na súradnicu). Preto je rozdelenie f z bootstrapu asymetrické (medián 1460 px, 68 % interval
1411–1493 px, kap. 10): neistota f (± 47 px) odráža hlavne to, či sa tieto dve strany započítajú.

**Porovnanie metód v skratke** (podrobne kap. 8):

| metóda | f [px] | pp [px] | poznámka |
|---|---|---|---|
| **hlavný: nálepky (výkres) + hrany** | **1469 ± 47** | (928, 503) | |
| hlavný model, samostatné váhy top / police | 1473 | (929, 504) | variant v systematike |
| hlavný model, stĺpiky ∥ zvislica scény | 1467 | (931, 507) | variant v systematike |
| hlavný model, podlahové čiary len na priamosť | 1469 | (928, 492) | variant v systematike |
| len hrany, spoločná zvislica (nezávislá impl.) | 1473 ± 22 | (931, 438 ± 46) | bez rozmerov |
| len hrany, modrá čiara ako jedna priamka | 1479 ± 21 | (939, 509) | bez rozmerov |
| len hrany, samostatné zvislice | 1425 ± 124 | (927, 503) | f nestabilné |
| úbežníky (+ plumb-line) | 1479 ± 48 | (929, 426) | |
| štúdia modelov, spoločné dáta | 1434 ± 43 | (927, 502) | |
| dva vozíky na jednej podlahe | 1378 ± 81 | – | konzistenčná kontrola |
| model vozíka ako kvádra (výkres) | 1359 ± 58 | stred obrazu | vychýlené geometriou |
| len nálepky (výkres) | 1300–1375 | cy 290–540 | vychýlené, RMS 5,0–5,9 px |

## 2. Dáta a predspracovanie

* **Vstup:** 7 fotiek 1920×1080 (`data/stills/`), statická scéna, pevná kamera. ChArUco fotky nie sú priložené
  (`data/charuco_photos/` je prázdny), takže kalibrácia stojí len na vozíkoch a scéne.
* **Chvenie medzi snímkami (rolling-shutter / vibrácie):** snímky sa líšia o zvislý posun a zvislú mierku,
  `dy = a_f + b_f·(y−540)` s |a_f| ≤ 1,0 px a |b_f| ≤ 6,3·10⁻⁴; `dx` je ~0,05 px (`work/02_initial_calib.py`).
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
  viditeľných bitoch; test proti všetkým 1000 kódom × 4 natočenia). Pri piatich nálepkách (80:8, 310:4, 310:5, 310:7, 310:8)
  existuje v slovníku aj iný kód so vzdialenosťou 0 na viditeľných bitoch, takže ich id je jednoznačné len medzi id
  použitými na vozíkoch (spolu s polohou na vozíku). Je to uvedené pri každej nálepke v `results/detections.json`.
* **Rohy:** každú stranu čierneho štvorca som sub-pixelovo dohľadal pozdĺž normály a rohy vypočítal ako priesečníky
  priamok strán (vzniká menší bias než pri cornerSubPix). Výsledok je priemer zo 7 zarovnaných snímok. Roh je platný len vtedy,
  keď sú obe jeho strany viditeľné a rovné. **Platných rohov je 62 z 80**, 18 je zakrytých (hranou dosky, susednou kartičkou,
  stĺpikom, lemom police). Pri čiastočne zakrytých nálepkách sa **viditeľné strany** (13 strán) používajú ako presné priamky
  modelu (priamka na nálepke podľa výkresu); tým sa využije aj nálepka, ktorej niektorý roh nie je vidieť.
* **Šum a systematika detekcie:**
  - rozptyl rohu medzi snímkami: medián 0,13 px (hranová metóda), 0,20 px (ArUco cornerSubPix);
  - hranové rohy vs. ArUco rohy: RMS 0,44 px, v priemere 0,16 px dovnútra po uhlopriečke;
  - **systematický posun hrán k čiernej strane o 0,66 ± 0,23 px** (tónová krivka alebo doostrenie kamery). Zmeral som ho na vnútornej
    mriežke kódu (hrany buniek musia ležať na mriežke 15 mm), vo všetkých snímkach je rovnaký. Opravené rohy sú v
    `corners_px_bias_corrected`. Vplyv na hlavný výsledok je v kap. 9.5 (rohy aj strany nálepiek).
* **Natočenie v rovine (rot_k):** top-nálepky majú 0, nálepky na policiach 2. Druhé najlepšie natočenie je horšie
  o ≥ 20 px RMS (najmenší rozdiel pri 80:8: 9,5 vs. 30,2 px), takže voľba je jednoznačná.
* Súbor `results/detections.json` obsahuje všetko na zopakovanie výpočtu: rohy (v spoločnom snímku aj v súradniciach
  jednotlivých fotiek), priradenie rohov k fyzickým rohom (`corners_3d_mm`, vzorec cez rot_k), platnosť, rozptyly,
  spôsob potvrdenia id a **strany čiastočne zakrytých nálepiek použité v hlavnom fite** (`sticker_sides_used`: body v px
  a presná 3D priamka). Obrázky: `results/markers_overlay.png`, `markers_crops.png`, `markers_rect_views.png`,
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
* **Lem E vozíka 80:** nezávislá implementácia čiarových metód ho vyradila ako mierne ohnutý (0,74–0,80 px). Hlavný fit
  však tri hrany tohto lemu používa (`cart80_x_E_front`, `cart80_x_E_front_in`, `cart80_x_A_front_in`, revízia ich
  prijala). Bez nich vychádza f = 1468 (zmena −1 px); variant je zahrnutý v systematike.
* **Hrany koncových dosiek sú rovné, ale nie sú rovnobežné s osami vozíka na presnosť úbežníkov** (napr. predná hrana dosky
  310/X0 je o 5,9° mimo úbežníka piatich lemov políc). Preto ich v skupinách úbežníkov nepoužívam, slúžia len na priamosť.
  Keďže všetky hrany v smere hĺbky vozíka (os Y) sú hrany koncových dosiek, **úbežník osi Y sa nepoužíva** (kap. 6, M3).
* Výstupy: `results/edges.json` (všetky hrany vrátane vyradených, s dôvodmi), `results/edges_cart310.png`,
  `edges_cart80.png`, `edges_scene.png` (s názvami hrán, konečný stav po revízii; `*` = len priamosť),
  `edges_all.png`; výrezy v `work/cache/edgecrops_*`, `review_*`.

## 5. Nesúlad nálepiek s výkresom – pozorovanie (`work/42_geomdiag_*`, `work/41_cuboid_*`)

Podľa zadania sú rozmery vozíka a polohy nálepiek presné. Hlavný výsledok ich preto nemení a všetko nižšie je
**len pozorovanie**. Zisťuje, kde a o koľko sa dáta s výkresom rozchádzajú; žiadny rozmer tým neupravujem a chybu
nalepenia nepredpokladám.

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
| sklon koncových dosiek pozdĺž X | 0° | zmerané z tvaru nálepiek, vonkajší koniec vyššie: 80/X0 +13,7°, 80/X1600 +5,7°, 310/X0 +13,6°, 310/X1600 +2,4° – nevýznamné; kap. 5.1 | 0,7–4 px na roh | horný a dolný okraj obrazu |
| bočný rozptyl nálepiek nad sebou | 0 mm | 1–6 mm | 0,2–3 px | všetky 4 stĺpce |
| zadná horná tyč vozíka 80 (model s výkresom) | Z = 0 | −56 mm | −17 px | pravý vozík |

* Najkompaktnejší opis je, že **top-nálepky (koncové dosky) vychádzajú o +60 až +80 mm vyššie voči nálepkám na policiach**,
  alebo ekvivalentne, že police vychádzajú nižšie. Samotný tento prvok zníži RMS z 5,3 na 2,9 px. Keď k tomu uvoľním aj
  sklony dosiek, klesne na 2,3 px (s voľnou veľkosťou kódu na 1,9 px). Žiadny kompaktný prvok nezníži RMS na úroveň šumu;
  časť nesúladu je nálepka po nálepke (pozorované lokálne nerovnosti podkladu; príčinu z fotiek určiť nemožno).
* **Top alebo police?** Z nálepiek samých je pozorovateľný len relatívny posun. Jediná nezávislá kontrola je zadná horná tyč
  vozíka 80 (Z = 0 podľa výkresu, v žiadnom fite nepoužitá, `work/cache/cuboid_diag.md`): sedí na +1,5 px, len ak sa
  zdvihnú top-dosky (+65 mm); pri hypotézach s nižšími policami zostáva na −17…−19 px. To **naznačuje skôr top-dosky
  nad úrovňou rámu** než nižšie police. Je to však jedna hrana, teda len náznak.
* Na výrezoch vidno, že koncové dosky sú tenké (~10 mm) biele dosky s potlačeným hárkom, nie krabice.
  60–90 mm vysoká bočná stena by mala 20–35 px a nie je viditeľná. Zadné rohy dosiek vozíka 80 sú zdvihnuté, doska
  310/X0 má previsnutú chlopňu. Fyzickú príčinu (rám, výšky políc, uchytenie) z fotiek určiť nemožno.
* **Dôsledok pre objektív:** nálepky s geometriou z výkresu vynútia f ≈ 1300–1375 px, hlavný bod podľa modelu
  (cy 290–540) a skreslenie, ktoré odporuje priamosti 61 hrán. Bez bariéry proti prehnutiu sú tieto fity prehnuté
  vo vnútri obrazu (fold margin −0,02 … −0,21); s bariérou buď sedia na bariére, alebo prejdú do iného minima
  (cy ≈ 290 px, k2 ≈ +0,37), ktoré síce nie je prehnuté, ale hrany narovná len na 1,6 px RMS (hlavný objektív 0,23 px).
  Keď sa uvoľní výška dosiek, nálepky samé dajú k1 = −0,295, k2 = 0,105 (pri 1300), teda prakticky rovnaké skreslenie
  ako plumb-line (−0,29; 0,07), a f ≈ 1430–1495.
  **Nálepky a hrany sa teda zhodnú na objektíve, len ak sa pripustí táto odchýlka geometrie.** Hlavný výsledok ju
  nepripúšťa (rozmery z výkresu), ale nálepky v ňom majú váhu podľa ich skutočného rozptylu voči modelu.
  Obrázky: `results/geomdiag_*.png`, `results/cuboid_overlay.png`, `results/cuboid_offsets.png`,
  `results/residuals_stickers_main.png`, `results/predicted_stickers_crops.png`; podrobne `work/cache/geometry_diagnosis.md`,
  `work/cache/cuboid_offsets.md`.

### 5.1 Sklon top-nálepiek: meranie a prepočet objektívu (`work/60_top_tilt_scan*.py`, `61_top_sticker_orientation*.py`, `62_top_tilt_methods.py`)

Na požiadanie som prepočítal hlavný odhad s top-nálepkami v miernom sklone a vyskúšal viac uhlov. Je to **what-if**
variant, lebo výkres predpisuje nálepky vodorovne; hlavný výsledok zostáva s geometriou z výkresu.

**Konvencie znamienok:** sklon okolo osi Y vozíka (pozdĺž X): + znamená, že vonkajší koniec dosky je vyššie (smerom k X = 0 pri
konci X0, k X = 1600 pri konci X1600). Sklon okolo osi X vozíka (pozdĺž Y): + znamená, že zadná strana (väčšie Y) je vyššie.

**1. Zmeraný sklon.** Póza každej nálepky sa určí len z jej štyroch rohov (hlavný objektív, bez polôh a výšok z výkresu) a jej
normála sa vyjadrí v súradniciach vozíka. Rotácia vozíka sa berie z nálepiek na policiach alebo z úbežníkov hrán; obe dávajú
zhodu na 0,2–0,6°. Neistota zahŕňa šum rohov, neistotu objektívu (celková kovariancia), zdroj rotácie vozíka a empirickú
systematiku 0,75° na nálepku (dve nálepky tej istej dosky sa líšia viac, než by dal šum).

| doska | nálepky | sklon okolo Y (+ = vonkajší koniec vyššie) | sklon okolo X (+ = zadná strana vyššie) | celkový uhol | významnosť |
|---|---|---|---|---|---|
| 80/X0 | 80:0 +12,6, 80:92 +14,1 | **+13,7 ± 1,3°** | −0,5 ± 1,3° | 13,7° | 10σ |
| 80/X1600 | 80:1 +6,6, 80:2 +5,1 | **+5,7 ± 0,9°** | −0,4 ± 0,9° | 5,9° | 6σ |
| 310/X0 | 310:0 +13,0, 310:322 +14,3 | **+13,6 ± 1,1°** | −0,4 ± 1,0° | 13,8° | 13σ |
| 310/X1600 | 310:1 +1,9, 310:2 +2,8 | **+2,4 ± 1,5°** | +0,5 ± 1,5° | 2,8° | 2σ (nevýznamné) |

* Kontrola: plne viditeľné nálepky na policiach vychádzajú vodorovne v rámci 1,5σ (sklon −1,3 až +2,9°).
* Sklon okolo osi X (roll) je pri všetkých doskách nulový v rámci neistoty. Predná a zadná nálepka tej istej dosky sa v ňom
  líšia o 2,5–4,3°; či ide o mierne prehnutie dosky alebo o anizotropnú chybu detekcie, sa rozlíšiť nedá.
* Nezávislá kontrola z priamej hrany čela každej koncovej dosky dáva zhodné hodnoty v rámci 1σ (táto kontrola je však
  menej presná).
* Pri zmene f v rozsahu 1420–1520 px sa sklon mení najviac o ~1,4°. Pri iných modeloch objektívu (17 variantov) vychádzajú
  dosky X0 na +12,3 až +14,8°.
* **Tri dosky sú merateľne naklonené, vonkajší koniec vyššie:** 80/X0 a 310/X0 o ~14°, 80/X1600 o ~6°. Doska 310/X1600
  naklonená preukazne nie je. To zodpovedá staršej diagnostike (kap. 5, geometry_diagnosis.md).

**2. Prepočet hlavného odhadu so sklonom.** Hlavný kombinovaný odhad (variančné komponenty do konvergencie) som prepočítal
pre rad uhlov a tri definície sklonu:

* **Y okolo stredu:** každá top-nálepka sa otočí okolo osi Y cez svoj stred; stred zostáva presne podľa výkresu.
* **X okolo stredu:** to isté okolo osi X.
* **Y s pántom:** celá doska sa otočí okolo pántu 55 mm dnu od stredu nálepky, takže stred nálepky stúpne o 55·sin(uhol) mm.

Ďalej sú prepočítané zmerané sklony dosiek a sklony dosiek nafitované na polohy rohov.

| konfigurácia | f [px] | cy [px] | k1 | RMS rohov všetky / top [px] | σ blok rohov [px] | f len nálepky | Δ zobrazenia pás / rohy [px] |
|---|---|---|---|---|---|---|---|
| bez sklonu (výkres = hlavný výsledok) | 1468,6 | 503,2 | −0,3516 | 7,89 / 6,58 | 10,72 | 1300 | 0,00 / 0,00 |
| Y okolo stredu −10° | 1468,7 | 503,2 | −0,3516 | 8,32 / 7,54 | 10,89 | 1313 | 0,01 / 0,02 |
| Y okolo stredu +4° | 1468,6 | 503,2 | −0,3516 | 7,81 / 6,42 | 10,69 | 1296 | 0,01 / 0,01 |
| Y okolo stredu +8° | 1468,6 | 503,2 | −0,3516 | 7,79 / 6,38 | 10,68 | 1293 | 0,01 / 0,02 |
| Y okolo stredu +10° | 1468,6 | 503,2 | −0,3516 | 7,80 / 6,40 | 10,68 | 1291 | 0,01 / 0,02 |
| Y okolo stredu +15° | 1468,6 | 503,2 | −0,3516 | 7,85 / 6,55 | 10,69 | 1288 | 0,00 / 0,01 |
| X okolo stredu −10° | 1468,7 | 503,2 | −0,3516 | 8,04 / 6,90 | 10,79 | 1303 | 0,01 / 0,02 |
| X okolo stredu +10° | 1468,7 | 503,2 | −0,3516 | 8,07 / 7,04 | 10,77 | 1300 | 0,02 / 0,03 |
| Y s pántom 55 mm −6° | 1469,0 | 503,2 | −0,3517 | 8,66 / 7,42 | 11,61 | 1293 | 0,10 / 0,15 |
| Y s pántom 55 mm +6° | 1468,3 | 503,2 | −0,3514 | 7,31 / 6,08 | 9,96 | 1306 | 0,12 / 0,16 |
| Y s pántom 55 mm +10° | 1468,0 | 503,2 | −0,3513 | 7,04 / 5,93 | 9,53 | 1309 | 0,20 / 0,27 |
| Y s pántom 55 mm +20° | 1467,4 | 503,1 | −0,3510 | 6,79 / 6,21 | 8,77 | 1314 | 0,38 / 0,52 |
| Y s pántom 55 mm +30° | 1467,2 | 503,0 | −0,3508 | 7,19 / 7,28 | 8,60 | 1313 | 0,45 / 0,60 |
| zmerané sklony dosiek, okolo stredu | 1468,6 | 503,2 | −0,3515 | 7,74 / 6,25 | 10,66 | 1292 | 0,02 / 0,02 |
| zmerané sklony dosiek, pánt 55 mm | 1468,1 | 503,2 | −0,3513 | 7,20 / 6,05 | 9,74 | 1304 | 0,18 / 0,24 |
| sklony dosiek nafitované na rohy (okolo stredu) | 1468,6 | 503,2 | −0,3515 | 7,72 / 6,20 | 10,65 | 1294 | 0,02 / 0,03 |

(„len nálepky“ = fit len z nálepiek s pp v strede obrazu; všetky tieto objektívy sú prehnuté vo vnútri obrazu.
„Δ zobrazenia“ je posun voči hlavnému výsledku, medián v páse vozíkov / v rohoch. Úplné tabuľky sú v
`work/cache/top_tilt_scan.md`, grafy v `results/top_tilt_scan.png` a `results/top_sticker_orientation.png`.)

**Záver prepočtu:**
* **Objektív sa sklonom top-nálepiek prakticky nemení.** Vo všetkých 58 konfiguráciách je f 1467,2–1469,1 px,
  cy 503,0–503,2 px a posun zobrazenia voči hlavnému výsledku nanajvýš 0,45 px v páse vozíkov a
  0,60 px v rohoch. To je hlboko pod neistotou 15,3 / 21,6 px. Dôvod: f a cy nesú úbežníky hrán a strany
  nálepiek na policiach, zatiaľ čo rohy nálepiek majú v spoločnom fite malú váhu (σ ≈ 10,7 px).
* **Sklon vysvetlí len malú časť nesúladu.** So zmeraným sklonom okolo stredu nálepky klesne RMS rohov zo 7,89 na
  7,74 px (top-rad 6,58 → 6,25 px). Pri doske otočenej okolo pántu stúpnu stredy nálepiek
  o ~13 mm a RMS klesne na 7,20 px. Zvyšok zodpovedá výškovému posunu top-nálepiek o +60–80 mm (kap. 5),
  ktorý sklon nevytvorí.
* **Nálepky samé zostávajú so sklonom aj bez neho nekonzistentné s hranami:** fity len z nálepiek dávajú f 1288–1331 px
  (hrany samé 1436, hlavný výsledok 1469) a objektív sa vo vnútri obrazu prehne.
* Ak by sa vynechali hrany, objektív na sklon reaguje (f 1352–1387 px, 30–45 px zobrazenia). Stabilitu hlavného výsledku
  teda dávajú hrany, nie nálepky.
* Nafitované sklony dosiek (polohy rohov, hlavný objektív) sú +12,3 ± 1,8 / +6,1 ± 2,1 / +15,7 ± 3,4 / −4,7 ± 2,8°
  (80/X0, 80/X1600, 310/X0, 310/X1600). Pri troch doskách sa zhodujú so zmeraným sklonom. Pri 310/X1600 majú opačné
  znamienko; tam je sklon z polohy rohov ovplyvnený výškovým nesúladom a priame meranie z tvaru je spoľahlivejšie.

## 6. Metódy (každá dotiahnutá do čísla s neistotou)

Všetky rezíduá hrán sa merajú **v skreslenom obraze**. Rezíduum je vzdialenosť pozorovaného bodu od dopredne
skreslenej priamky ("forward"). Meranie v neskreslenom obraze by šum systematicky ťahal k slabšiemu skresleniu
(syntetický test: pri σ = 0,2 px je k1 vychýlené o +0,01 až +0,04). Každý fit má navyše **bariéru proti prehnutiu**:
radiálne skreslenie musí byť monotónne až po rohy obrazu, inak nie je invertovateľné.

| # | metóda | čo používa | skript |
|---|---|---|---|
| M1 | nálepky (BA) | 62 rohov, rozmery z výkresu; 2 pózy vozíkov | `20_markers_ba.py` |
| M2 | plumb-line | priamosť 61 hrán (len skreslenie a jeho stred) | `30_lines_methods.py`, `40_lines_indep_a_plumb.py` |
| M3 | úbežníky | smery osí X a Z vozíkov, zvislice scény, rovnobežné čiary na podlahe; kolmosť → f, pp | `30_lines_methods.py`, `40_lines_indep_b_vp.py` |
| M4 | spoločný fit z čiar | M2 + M3 naraz (bez rozmerov) | `30_lines_methods.py`, `40_lines_indep_c_joint.py` |
| M5 | model vozíka (kváder s policami) | nálepky + presné hrany modelu (výkres) + kontrola predikcie všetkých hrán | `41_cuboid_*.py` |
| M6 | **kombinovaný (hlavný)** | všetky nálepky (výkres, jedna spoločná váha pre všetky rohy) + hrany (úbežníky vrátane podlahy, priamosť) + viditeľné strany čiastočne zakrytých nálepiek | `50_combined.py`, `combined.py` |
| M7 | dva vozíky na jednej podlahe | súhlas normál podlahy z póz oboch vozíkov ako funkcia f | `42_geomdiag_d_twocart.py` |
| M8 | tvar nálepiek | každá nálepka ako vodorovný štvorec (bez polôh a výšok) + nadir | `42_geomdiag_e_markershape.py` |
| M9 | iné modely objektívu | Brown (rôzne sady), racionálny, Kannala-Brandt, divízny; tie isté dáta | `43_altmodels_*.py` |

Poznámky k jednotlivým metódam:

* **M1 – nálepky:** výber modelu pomocou leave-one-out (najlepší model k1,k2 s pp v strede, LOO 8,4 px), profil f,
  klastrový bootstrap po nálepkách, vynechanie radu a vozíka. Všetky modely majú RMS 5,0–5,9 px (max 9,5–11,7 px).
  Bez bariéry sú prehnuté vo vnútri obrazu; s bariérou sedia na bariére, alebo (voľný pp) prejdú do minima s cy ≈ 290 px.
  Výsledok je **vychýlený** geometriou (kap. 5), nie zašumený.
* **M2 – plumb-line:** skreslenie je určené v pixelových jednotkách (k1/f², k2/f⁴), f z priamosti určiť nemožno.
  Model k1 samotný nestačí (0,51 px, naráža na bariéru). Pre k1, k2 (voľný stred) je rezíduum 0,217 px,
  čo zodpovedá vlastnej nerovnosti hrán. Stred skreslenia je v x určený (924–938), v y len voľne (423–516 podľa modelu).
* **M3 – úbežníky:** VP skupín s ML smerom; smerová chyba jedného člena ~0,16°, čo je ďaleko nad šumom bodov.
  f dáva kolmosť osi X (lemy políc) a osi Z (stĺpiky) vozíkov a kolmosť podlahových smerov na zvislicu.
  **Úbežník hĺbky (os Y) sa použiť nedal:** jediné hrany v smere Y sú hrany koncových dosiek, ktoré nie sú rovnobežné
  s osami vozíka (kap. 4). Hlavný bod zo samotných úbežníkov určiť nemožno (berie sa stred skreslenia). Zvislice scény sa
  od stĺpikov vozíkov líšia o 0,5–0,8° (podlaha nie je dokonale vodorovná / predmety nie sú presne zvislé), preto majú
  vlastný smer.
* **M4 – spoločný fit z čiar:** f závisí od toho, či majú oba vozíky spoločnú zvislicu: so spoločnou 1473 px,
  so samostatnými 1425–1446 px, keďže vozík 80 má len dve krátke hrany stĺpika. Vynechanie jednej krátkej hrany stĺpika
  vozíka 310 posunie f o ~100 px, takže **f zo samotných hrán je krehké** (bootstrap σ 114–124 px v prvej implementácii,
  jackknife σ 156 px v štúdii modelov).
* **M5 – kváder:** s rozmermi z výkresu f = 1359 ± 36 px (bootstrap; ± 58 vrátane geometrie), rezíduá 6 px, sedí na
  bariére proti prehnutiu. Lemy políc sedia na −2…+9 px. Mimo predikcie sú: zadná horná tyč vozíka 80 (−17 px, −56 mm;
  po zdvihnutí top-dosky o +65 mm +1,5 px), zadné pozdĺžniky `cart310_x_back_inner` a `cart80_x_back_low` (−75…−79 px),
  ktoré som priradil hornej zadnej hrane, hoci sú to zjavne nižšie rúrky rámu (výkres ich neobsahuje), `cart80_x_back_rail_in`
  (−29 px) a `cart310_x_B_inner_fall` (+13 px, vnútorná hrana ohybu lemu). Podrobne `work/cache/cuboid_offsets.md`.
* **M6 – kombinovaný (hlavný):** jeden fit všetkých blokov: rohy nálepiek (M), strany čiastočne zakrytých nálepiek (L),
  úbežníkové skupiny hrán (V) a priamosť hrán koncových dosiek a scény (S). Váhy blokov sú **variančné komponenty**
  iterované do konvergencie (relatívna zmena σ bloku < 0,5 %; každý blok má potom redukované χ² ≈ 1): σ rohov 10,7 px (na súradnicu),
  strán 1,37 px, úbežníkov 0,36 px, priamosti 0,23 px. Všetky nálepky majú v rámci bloku rovnakú váhu;
  rozmery z výkresu sa nemenia. Keďže nesúlad geometrie zväčší rozptyl rohov, dostanú rohy ako blok malú váhu.
  **Vplyv jednotlivých blokov na f** (fit bez bloku, váhy ostatných blokov ponechané):

  | vynechaný blok | f | cx | cy | k1 |
  |---|---|---|---|---|
  | – (hlavný) | 1468,6 | 928,2 | 503,2 | −0,352 |
  | rohy nálepiek (62) | 1471 | 928 | 503 | −0,352 |
  | strany nálepiek (13) | 1436 | 927 | 501 | −0,335 |
  | úbežníky | 1405 | 925 | 427 | −0,316 |
  | len strany 80:3/1 a 80:7/3 (znova vážené) | 1443 | 928 | 502 | −0,339 |

  f teda nesú hlavne úbežníky hrán; zo strán nálepiek prispievajú najmä dve strany na konci 80/X0 (+26 px),
  kde podľa kap. 5 nesedí pomer rozostupov políc. Preto sú v systematike aj tieto varianty: bez týchto dvoch strán,
  samostatné variančné komponenty pre top a pre police (RMS top-rohov 22,2 px na bod, polic 2,9 px → f 1473),
  robustné váhy po nálepkách (f 1476), stĺpiky oboch vozíkov rovnobežné so zvislicou scény (f 1467),
  čiary na podlahe len na priamosť (cy 492) a bez lemu E vozíka 80 (f 1468).
* **M7 – dva vozíky:** f, pri ktorom sú normály podlahy z póz oboch vozíkov rovnobežné: 1378 ± 81 px (len konzistencia;
  geometria z výkresu). **Výška kamery nad podlahou** z póz oboch vozíkov sa zhoduje na |Δh| < 5 mm pre každé f
  v rozsahu 1100–1800, je teda konzistentná, ale f neobmedzuje.
* **M8 – tvar nálepiek:** len konzistenčná kontrola; na f príliš slabá (± 170 px).
* **M9 – modely objektívu:** dáta podporujú práve 2 radiálne stupne voľnosti. k3 je len konzistentné, p1, p2 a fx ≠ fy nie sú
  určené (kap. 7).

## 7. Výber modelu (primerane dátam)

Zo štúdie 16 modelov na rovnakých dátach (`work/43_altmodels_*`, `results/altmodels_*.png`, `work/cache/altmodels_summary.md`):

* **Radiálne skreslenie potrebuje práve 2 stupne voľnosti.** k1 samotné hrany nenarovná (0,29–0,51 px, naráža
  na bariéru prehnutia, ΔQAIC +315, ΔQBIC +302). Všetky dvoj- a trojparametrové radiálne tvary (Brown k1k2, k1k2k3,
  divízny l1l2, racionálny k4, Kannala-Brandt k1k2) dosiahnu 0,228–0,229 px. To je hranica daná vlastnou nerovnosťou hrán.
  Ich zobrazenia sa od Brown k1,k2 líšia o 0,4 / 2,0 / 2,7 px (RMS cez modely; max 0,5 / 2,3 / 3,9 px; stred / pás / rohy).
* **k3** vychádza −0,03 ± 0,02 (štúdia modelov −0,031 ± 0,025; kombinovaný fit −0,022): je len konzistentné, preto ho
  fixujem na 0 a jeho vplyv je v systematike.
* **p1, p2:** na hranách vychádzajú ~0 (−0,001 ± 0,006; 0,0015 ± 0,003) a fit je nestabilný. S nálepkami s geometriou z výkresu
  vyskočí p1 na 0,009 a f klesne na 1398 (kombinovaný fit, cy 453; v štúdii modelov 1374), lebo tangenciálne
  členy len pohlcujú nesúlad geometrie. Preto p1 = p2 = 0.
* **fx ≠ fy** nie je určené (fx 1383 ± 91 vs. fy 1426 ± 34 z hrán; kombinovaný fit fx/fy − 1 = −2,2 %), preto fx = fy.
* **Hlavný bod voľný:** fixácia do stredu obrazu stojí ΔQAIC +21…+24 a zhoršuje krížovú predikciu. cx je určené,
  cy len konzistentné.
* Prevod na OpenCV Brown [k1, k2, p1, p2, k3]: Kannala-Brandt k1,k2, divízny l1,l2 a racionálny k4 sa prevedú s chybou
  ≤ 0,08 px (medián); pri racionálnom k1..k6 je to 0,2–0,45 px. Brown k1, k2 je teda primeraný model.

## 8. Porovnanie všetkých metód

Tabuľka je vygenerovaná zo súborov metód (`work/97_report_tables.py`). Všetky metódy sú aj v `results/lens_by_method.json`
(58 záznamov vrátane variantov, jednotný tvar a `method_id`) a na grafe `results/methods_comparison.png`.
Stĺpec „Δ zobrazenia“ je posun zobrazenia metódy voči hlavnému výsledku po kompenzácii rotácie (medián oblasti). Neistoty
metód sú ich vlastné (bootstrap alebo klastrové), okrem hlavného riadku, ktorý má celkovú neistotu.

| metóda | rozmery z výkresu | f [px] | cx | cy | k1 | k2 | k3 | prehnutie | Δ zobrazenia vs hlavný: stred / pás / rohy [px] | neistota zobrazenia metódy (stred / pás / rohy) | pozn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **M6 kombinovaný – HLAVNÝ** | áno (všetky nálepky) | 1469 ± 47 | 928 ± 19 | 503 ± 29 | −0,352 | +0,099 | +0,000 | nie | 0,0 / 0,0 / 0,0 | 4,0 / 15,3 / 21,6 | celková neistota |
| M6, samostatné váhy top / police | áno | 1473 | 929 | 504 | −0,354 | +0,101 | +0,000 | nie | 0,3 / 1,2 / 1,4 | – / – / – | variant |
| M6, robustné váhy nálepiek | áno (top-nálepky ~0 váha) | 1476 | 930 | 505 | −0,357 | +0,103 | +0,000 | nie | 0,5 / 2,2 / 2,7 | – / – / – | variant |
| M6, stĺpiky ∥ zvislica scény | áno | 1467 | 931 | 507 | −0,354 | +0,102 | +0,000 | nie | 0,3 / 0,9 / 2,1 | – / – / – | variant |
| M6, bez strán 80:3/1, 80:7/3 | áno | 1443 | 928 | 502 | −0,339 | +0,092 | +0,000 | nie | 1,8 / 8,1 / 11,0 | – / – / – | variant |
| M6, bez lemu E vozíka 80 | áno | 1468 | 931 | 505 | −0,352 | +0,100 | +0,000 | nie | 0,2 / 0,4 / 0,7 | – / – / – | variant |
| M6, podlahové čiary len na priamosť | áno | 1469 | 928 | 492 | −0,350 | +0,098 | +0,000 | nie | 0,8 / 0,5 / 1,6 | – / – / – | variant |
| M6, top-nálepky so zmeraným sklonom (what-if) | áno + sklon top | 1469 | 928 | 503 | −0,352 | +0,099 | +0,000 | nie | 0,0 / 0,0 / 0,0 | – / – / – | what-if, kap. 5 |
| M6, top-dosky so zmeraným sklonom, pánt 55 mm (what-if) | áno + sklon top | 1468 | 928 | 503 | −0,351 | +0,099 | +0,000 | nie | 0,0 / 0,2 / 0,2 | – / – / – | what-if, kap. 5 |
| M6 kombinovaný, k1,k2,k3 | áno | 1470 | 930 | 505 | −0,364 | +0,132 | −0,022 | nie | 0,1 / 0,2 / 0,4 | – / – / – | k3 len konzistentné |
| M6 kombinovaný, pp v strede obrazu | áno | 1467 | 960 | 540 | −0,372 | +0,116 | +0,000 | nie | 3,0 / 2,7 / 11,0 | – / – / – |  |
| M6 kombinovaný, fx≠fy | áno | 1431/1463 | 928 | 509 | −0,345 | +0,095 | +0,000 | nie | 1,6 / 10,5 / 20,3 | – / – / – | fx≠fy neurčené |
| M6 kombinovaný, +p1,p2 | áno | 1398 | 946 | 453 | −0,314 | +0,079 | +0,000 | nie | 4,9 / 20,6 / 24,5 | – / – / – | p1,p2 pohltia nesúlad geometrie |
| M4 spoločný fit z čiar (nezávislá impl., spoločná zvislica) | nie | 1473 ± 22 | 931 ± 8 | 438 ± 46 | −0,349 | +0,095 | +0,000 | nie | 4,5 / 3,1 / 9,3 | 3,3 / 7,0 / 11,8 |  |
| M4 spoločný fit z čiar, podlahová čiara ako 1 priamka | nie | 1479 ± 21 | 939 ± 7 | 509 ± 42 | −0,362 | +0,109 | +0,000 | nie | 0,9 / 2,9 / 3,9 | – / – / – |  |
| M4 spoločný fit z čiar (prvá implementácia, samostatné zvislice) | nie | 1425 ± 124 | 927 ± 6 | 503 ± 8 | −0,333 | +0,090 | +0,000 | nie | 3,0 / 14,0 / 19,6 | – / – / – | cy ± len štatistická |
| M3 úbežníky (nezávislá impl.) | nie | 1479 ± 48 | 929 ± 9 | 426 ± 49 | −0,349 | +0,094 | +0,000 | nie | 5,4 / 4,9 / 11,8 | 4,8 / 14,9 / 22,1 | pp = stred skreslenia |
| M3 úbežníky (prvá implementácia, `30_lines_methods.py`) | nie | 1419 ± 114 | 941 ± 7 | 513 ± 5 | −0,348 | +0,102 | +0,000 | nie | 3,5 / 16,9 / 28,6 | – / – / – | cy ± len štatistická |
| M2 plumb-line (skreslenie; f z M3) | nie | 1479 ± 48 | 929 ± 10 | 426 ± 50 | −0,349 | +0,094 | +0,000 | nie | 5,4 / 4,9 / 11,8 | 3,5 / 2,5 / 7,9 | len k1/f², k2/f⁴ a stred |
| M9 Brown k1,k2 (štúdia modelov, spoločné dáta) | áno | 1434 ± 43 | 927 ± 10 | 502 ± 43 | −0,336 | +0,091 | +0,000 | nie | 2,3 / 10,7 / 14,7 | 4,3 / 15,5 / 22,1 |  |
| M9 Brown k1,k2,k3 | áno | 1432 ± 44 | 931 ± 10 | 505 ± 35 | −0,353 | +0,137 | −0,031 | na bariére | 2,7 / 12,4 / 16,9 | 4,1 / 15,1 / 21,1 |  |
| M9 divízny l1,l2 | áno | 1431 | 935 | 508 | −0,356 | +0,138 | −0,028 | nie | 2,7 / 12,9 / 18,4 | 1,8 / 7,6 / 10,7 | prevedený na Brown |
| M9 Kannala-Brandt k1,k2 | áno | 1431 | 931 | 506 | −0,354 | +0,140 | −0,032 | na bariére | 2,7 / 12,8 / 17,3 | 1,8 / 7,7 / 10,8 | prevedený na Brown |
| M7 dva vozíky (normály podlahy) | áno | 1378 ± 81 | 955 | 522 | −0,328 | +0,090 | +0,000 | nie | 6,4 / 29,9 / 47,0 | 5,9 / 25,9 / 35,8 | len konzistencia |
| M8 tvar nálepiek | nie (len štvorce) | 1601 ± 174 | 979 ± 49 | 471 ± 26 | −0,443 | +0,165 | +0,000 | nie | 9,7 / 39,4 / 48,4 | 12,8 / 49,4 / 70,9 | príliš slabé |
| M5 kváder s policami (výkres) | áno | 1359 ± 58 | 960 | 540 | −0,210 | +0,000 | +0,000 | na bariére | 7,5 / 28,5 / 19,0 | 2,9 / 13,4 / 21,9 | vychýlené geometriou |
| M1 len nálepky (výkres) | áno | 1326 ± 35 | 960 | 540 | −0,246 | +0,025 | +0,000 | na bariére | 10,1 / 43,1 / 55,4 | 2,9 / 15,4 / 36,9 | vychýlené geometriou |

**Prečo sa metódy líšia – čo z toho vyplýva:**

1. **Nálepky s geometriou z výkresu verzus hrany (f ~1300–1375 vs. ~1430–1480).** Rozdiel nespôsobuje objektív, ale geometria.
   Nálepky nesedia so žiadnou dierkovou kamerou a samy vynútia skreslenie, ktoré odporuje priamosti hrán. Keď sa uvoľní výška
   top-dosiek (+60–80 mm), dajú nálepky rovnaké skreslenie ako hrany a f = 1430–1495. Syntetický test so zdvihnutými doskami
   (scenár T2_c) reprodukuje presne tento vzor: fity len z nálepiek sú vychýlené o −115 až −143 px vo f, kombinovaný odhad nie
   (−0,6 px).
2. **Hrany medzi sebou (f 1425 vs. 1473–1479).** Rozdiel spôsobuje hlavne predpoklad, či majú oba vozíky rovnakú zvislicu.
   Vozík 80 má len dve krátke, takmer kolineárne hrany stĺpika, takže jeho vlastná zvislica je slabo určená. Hrany samotné dávajú f
   krehko: vynechanie jednej krátkej hrany stĺpika posunie f o ~100 px. V hlavnom fite tento predpoklad mení f o
   −2 px (variant so spoločnou zvislicou).
3. **Hlavný fit verzus hrany samé (f 1469 vs. 1436).** Rozdiel spôsobujú strany čiastočne zakrytých nálepiek, hlavne
   dve na konci 80/X0 (kap. 6, M6). Bez nich je hlavný fit na 1443.
4. **cy (426–513).** Určujú ho hlavne úbežníky: bez nich vychádza v hlavnom fite cy = 427. V hlavnom fite
   posunie predpoklad o rovnobežnosti čiar na podlahe cy len o 11 px (čiary len na priamosť: 492).
   Nižšie hodnoty 426–438 vychádzajú z čiarových fitov bez nálepiek so spoločnou zvislicou; 509, ak sa obe časti modrej čiary
   berú ako jedna priamka. Nálepky s uvoľnenou geometriou dávajú cy ≈ 390. **cy je z dát len konzistentné.**
5. **Zvislice scény verzus stĺpiky vozíkov** sa líšia o 0,5–0,8°. Podlaha alebo predmety nie sú presne zvislé, preto
   majú zvislice scény vo všetkých fitoch vlastný smer. V hlavnom fite zvierajú osi Z oboch vozíkov 0,63°.

## 9. Overenie

### 9.1 Reziduá

* **Hlavný fit, bloky:** priamosť hrán 0,23 px, konzistencia s úbežníkmi 0,36 px, strany zakrytých nálepiek 1,4 px,
  rohy nálepiek 15,2 px na bod (v spoločnom fite, póza určená hlavne hranami).
* **Rohy nálepiek s hlavným objektívom a pózou vozíka fitovanou z jeho nálepiek (metóda najmenších štvorcov):**
  **7,9 px RMS (max 13,7 px)**. To je nesúlad geometrie, nie šum (šum ~0,15 px). Obrázky
  `results/residuals_stickers_main.png` (šípky ×10) a `results/predicted_stickers_crops.png` (výrezy: detegované zelené,
  predpovedané červené). Reziduá hrán: `results/residuals_edges_main.png`. Obraz po odstránení skreslenia je
  `results/undistorted_mean.png`: hrany políc, pásky aj modrá čiara sú vizuálne rovné.

Reziduá rohov nálepiek s hlavným objektívom (geometria z výkresu, póza vozíka metódou najmenších štvorcov):

| skupina | RMS [px] | max [px] | počet rohov |
|---|---|---|---|
| všetky | 7,9 | 13,7 | 62 |
| vozík 80 | 8,1 | 12,3 | 34 |
| vozík 310 | 7,6 | 13,7 | 28 |
| rad top | 6,6 | 10,6 | 32 |
| rad A | 11,0 | 13,7 | 10 |
| rad B | 8,3 | 10,6 | 13 |
| rad C | 7,0 | 9,7 | 7 |
| každá zo 7 fotiek zvlášť (póza pre fotku) | 7,78–7,90 | 13,5–13,8 | 61–62 |

Po nálepkách (rovnaká definícia):

| vozík:id | poloha | RMS [px] | max [px] | rohov |
|---|---|---|---|---|
| 80:0 | TOP-TL | 3,7 | 4,1 | 4 |
| 80:1 | TOP-TR | 6,7 | 8,0 | 4 |
| 80:2 | TOP-BR | 8,9 | 10,3 | 4 |
| 80:3 | A-LEFT | 7,1 | 8,0 | 2 |
| 80:4 | A-RIGHT | 11,9 | 12,3 | 4 |
| 80:5 | B-LEFT | 9,4 | 10,6 | 4 |
| 80:6 | B-RIGHT | 7,9 | 8,4 | 4 |
| 80:7 | C-LEFT | 9,3 | 9,7 | 2 |
| 80:8 | C-RIGHT | 5,7 | 5,8 | 2 |
| 80:92 | TOP-BL | 6,8 | 7,8 | 4 |
| 310:0 | TOP-TL | 6,6 | 8,0 | 4 |
| 310:1 | TOP-TR | 4,5 | 5,7 | 4 |
| 310:2 | TOP-BR | 9,2 | 10,6 | 4 |
| 310:3 | A-LEFT | 13,6 | 13,7 | 2 |
| 310:4 | A-RIGHT | 9,5 | 10,1 | 2 |
| 310:5 | B-LEFT | 10,4 | 10,4 | 1 |
| 310:6 | B-RIGHT | 6,9 | 7,9 | 4 |
| 310:7 | C-LEFT | 7,9 | 7,9 | 1 |
| 310:8 | C-RIGHT | 4,7 | 5,6 | 2 |
| 310:322 | TOP-BL | 3,8 | 5,2 | 4 |

* **Len nálepky, s geometriou z výkresu (M1):** RMS 5,0–5,9 px (max 9,5–11,7 px) pri šume ~0,13 px podľa modelu.
  Najlepší model (k1,k2, pp v strede): po vozíkoch 5,7 / 5,8 px, po radoch top 5,6 / A 8,6 / B 3,4 / C 4,6 px
  (`results/markers_residuals.png`).

### 9.2 Profil ohniska (`results/f_profiles.png`, `results/combined_f_profile.png`)

* **Len nálepky:** RMS má minimum pri f ≈ 1300–1375 a rastie o 0,5–1 px na každých 100 px posunu f. Minimum je plytké a
  vychýlené geometriou.
* **Len hrany:** RMS spoločného fitu z čiar (priamosť + úbežníky, samostatné zvislice; samotná priamosť od f nezávisí) má
  minimum pri f ≈ 1430–1440 a je veľmi plochý: +0,002 px pri ±30 px, +0,005 px pri ±50 px. f zo samotných hrán je slabé.
* **Kombinovaný (hlavný):** f je pevne nastavené a všetko ostatné sa refituje (váhy blokov hlavného fitu). Minimum je pri
  1469 px (mriežka po 31 px). Formálne Δχ² = 1 zodpovedá ±3,3 px (kovariancia). Pri f 1437 / 1500 px je Δχ²
  94 / 88; RMS blokov rohy 10,2 / 11,4 (v minime 10,7), strany
  1,95 / 1,37 (1,37), úbežníky 0,36 / 0,37 (0,36), priamosť
  0,2317 / 0,2312 (0,2314) px. Príspevky blokov k Δχ²: pri 1437 px strany +121, rohy −12, úbežníky −18;
  pri 1500 px strany +0, rohy +16, úbežníky +74. Nižšie f teda odmietajú hlavne strany
  nálepiek, vyššie f hlavne úbežníky a menej rohy (samotné úbežníky majú minimum okolo 1440 px, rohy okolo 1310 px, strany
  okolo 1480 px).
  Legenda obrázka: M = rohy nálepiek, L = strany čiastočne zakrytých nálepiek, V = úbežníky, S = priamosť.
  Formálna krivka je úzka; reálnu neistotu (±47 px) určuje bootstrap, rozptyl alternatív a jackknife krížovej validácie.

### 9.3 Krížová validácia

* **Len nálepky** (`20_markers_ba.py`): vynechanie jednej nálepky dáva predikčnú chybu 8,4–9,1 px (v rámci fitu 5,0–5,9).
  Vynechanie radu: top 16–22 px, A 10–11, B 4,7–6, C 8–11 px. Vynechanie vozíka (vnútorné parametre z jedného vozíka,
  póza druhého refitovaná): 5,8–6,0 px pre modely s pp v strede, 5,5–87 px pre modely s voľným pp (k1: 9,5 px).
* **Kombinovaný (hlavný)** (`51_combined_cv.py`): vynechá sa rad, vozík, oblasť hrán alebo jedna nálepka, fit sa zopakuje
  (vrátane váh blokov) a vynechané dáta sa predpovedajú. Pri vynechanom rade / nálepke sa použije póza redukovaného fitu,
  pri vynechanom vozíku sa jeho póza refituje z jeho nálepiek. Pre porovnanie je uvedené RMS tých istých bodov v hlavnom fite.

| vynechané | f | cx | cy | k1 | k2 | predikcia vynechaných [px] | tie isté body v hlavnom fite [px] | trénovacie rohy v spoločnom fite [px/bod] |
|---|---|---|---|---|---|---|---|---|
| rad top | 1474 | 929 | 504 | −0,355 | +0,102 | 22,6 | 20,1 | 2,9 |
| rad A | 1428 | 928 | 501 | −0,333 | +0,089 | 9,0 | 6,4 | 14,4 |
| rad B | 1475 | 931 | 506 | −0,357 | +0,104 | 9,9 | 8,0 | 17,3 |
| rad C | 1435 | 927 | 501 | −0,335 | +0,090 | 8,7 | 4,0 | 11,6 |
| vozík 80 | 1437 | 927 | 502 | −0,336 | +0,091 | 7,5 (póza refit.) | 8,1 (póza z nálepiek) | 12,9 |
| vozík 310 | 1484 | 927 | 503 | −0,358 | +0,103 | 7,8 (póza refit.) | 7,6 (póza z nálepiek) | 15,1 |
| hrany vozíka 310 | 1484 | 911 | 497 | −0,344 | +0,087 | priamosť 0,25 | 0,23 (všetky hrany) | 14,8 |
| hrany vozíka 80 | 1431 | 943 | 503 | −0,335 | +0,092 | priamosť 0,25 | 0,23 (všetky hrany) | 10,7 |
| hrany scény | 1474 | 935 | 534 | −0,365 | +0,117 | priamosť 0,22 | 0,23 (všetky hrany) | 15,2 |
| **jackknife σ: nálepky (20)** | 36 | 3 | 3 | 0,018 | 0,010 | | | |
| **jackknife σ: rady (4)** | 38 | 2 | 3 | 0,019 | 0,011 | | | |
| **jackknife σ: vozíky (2)** | 24 | 0 | 1 | 0,011 | 0,006 | | | |
| **jackknife σ: oblasti hrán (3)** | 33 | 19 | 23 | 0,018 | 0,018 | | | |

  Vynechanie jednej nálepky (20 fitov): f 1442–1475 px (jackknife σ_f = 36 px; leave-one-out odhady
  sú korelované, preto sa rozptyl násobí (n−1)), cy 502–506,
  zobrazenie sa zmení najviac o 1,8 / 8,3 / 11,2 px. f sa pohne o viac ako 5 px len pri
  vynechaní nálepiek 80:3 (−26,5 px), 310:4 (+5,2 px), 310:5 (+6,0 px), 80:7 (−24,2 px). Predikčná chyba vynechanej nálepky: police 0,8–12,8 px,
  top-nálepky 10–26 px; predikcia je v priemere o 6 % horšia než RMS tých istých bodov
  v hlavnom fite, takže fit nie je preučený na jednotlivé nálepky, ale geometria top-dosiek sa z ostatných dát predpovedať nedá.
  Najjasnejšie to ukazuje vynechanie celého top-radu: bez neho majú zvyšné rohy v spoločnom fite 2,9 px na bod
  a top-rad sa predpovedá s chybou 22,6 px.
* **Jackknife po hranách** (štúdia modelov, spoločné dáta): f 1413–1457 pri vynechaní jedného člena. Samotné hrany:
  vynechanie jednej krátkej hrany stĺpika vozíka 310 posunie f o ~+100 px.

### 9.4 Syntetický round-trip (`work/45_synth_main.py`, `work/44_synth_*`, `results/synthetic_*.png`)

Syntetické dáta majú presne reálnu geometriu a viditeľnosť (tie isté nálepky a rohy, tie isté hrany prevedené na ideálne priamky)
a vznikli zo známeho objektívu. Pravda T2: f = 1420, pp (935, 505), k1 −0,33, k2 0,09 (blízko odhadov zo samotných hrán;
  od hlavného výsledku sa f líši o 49 px); T1: f = 1250,
pp (955, 530), k1 −0,26, k2 0,06. Scenáre:

* **a** – len šum detekcie;
* **b** – šum detekcie + náhodná odchýlka 3D bodov od výkresu (σ 5 mm na nálepku) a výšok radov (σ 33 mm),
  nastavená tak, aby RMS nálepiek bolo ako v reálnych dátach (5,1–5,4 px). Je to štatistický model nesúladu neznámeho
  pôvodu, nie tvrdenie o nalepení. K tomu ohyby a smerové chyby hrán z reálnych reziduí;
* **c** – systematicky zdvihnuté top-dosky podľa diagnostiky (kap. 5).

**Hlavný odhad presne v tej podobe, v akej dal výsledok** (`45_synth_main.py`, váhy blokov iterované do konvergencie):

| scenár | pravda | replík | odchýlka f (± chyba priemeru) | rozptyl f | RMSE f | RMSE cx / cy | RMSE k1 / k2 | RMSE zobrazenia stred / pás / rohy [px] |
|---|---|---|---|---|---|---|---|---|
| a | T2 | 8 | +0,1 ± 0,1 | 0,3 | 0,3 | 0,5 / 0,3 | 0,000 / 0,001 | 0,0 / 0,1 / 0,3 |
| **b** | T2 | 30 | +0,3 ± 4,0 | 21,9 | 21,5 | 5,7 / 4,0 | 0,012 / 0,008 | 1,6 / 7,0 / 10,0 |
| c | T2 | 8 | −0,6 ± 0,3 | 0,9 | 1,0 | 1,8 / 1,5 | 0,001 / 0,002 | 0,1 / 0,4 / 1,1 |
| b | T1 | 30 | −7,7 ± 2,8 | 15,1 | 16,7 | 7,7 / 5,2 | 0,007 / 0,004 | 1,4 / 6,2 / 9,2 |

* **Postup vráti známy objektív.** V scenároch a a c je hlavný odhad nevychýlený (f +0,1, resp. −0,6 px).
  V scenári b je odchýlka f +0,3 ± 4,0 px (T2) a −7,7 ± 2,8 px (T1);
  pri T2 je odchýlka nevýznamná, pri T1 je malá, ale štatisticky významná (2,8σ, 0,6 % f); voči rozptylu je malá a RMSE ju obsahuje. Rozptyl f v scenári b (T2) je 22 px.
* **Scenár b nereprodukuje štruktúru reziduí hlavného fitu:** medián σ blokov rohy / strany / úbežníky / priamosť je
  5,5 / 2,62 / 0,34 / 0,22 px, reálne 10,7 / 1,37 / 0,36 / 0,23 px; scenár c
  neobsahuje nesúlad rozostupov políc na konci 80/X0, kde ležia dve strany nálepiek posúvajúce f o +26 px.
  RMSE zo syntetického testu (30 replík, samo s neistotou ~±13 %) je preto len dolná hranica.
  RMSE f rastie približne s f (T1: 17 px pri 1250, T2: 22 px pri 1420).
* **Výsledná neistota je najväčšia z troch hodnôt:** empirický rozpočet (bootstrap ⊕ rozptyl alternatív), RMSE zo scenára b
  (T2) a jackknife z krížovej validácie (kap. 9.3, 10).
* **Staršia štúdia** (`work/44_synth_*`, `work/cache/synthetic_report.md`, `results/synthetic_*.png`; pravda T2: f = 1400,
  pp (935, 410), k1 −0,32, k2 0,08; 9 replík) porovnala všetky metódy: fity len z nálepiek sú
  v scenári b rozptýlené o 65–100 px vo f (v rohoch 45–186 px zobrazenia) a v scenári c vychýlené o −115 až −143 px;
  spoločný fit z čiar má v scenári b RMSE f ~27 px. **Formálne kovariancie podhodnocujú chybu 9–15×**, bootstrap ju naopak
  skôr nadhodnocuje (pomer 0,4–1,0). Na reálnych dátach je bootstrapový rozptyl f hlavného odhadu 45 px,
  v scenári b je rozptyl f 22 px.
* Metóda plumb-line s pevným stredom (stred obrazu) je vychýlená, ak skutočný stred nie je v strede obrazu
  (T2: k1 o −0,018). Preto je stred skreslenia v hlavnom fite voľný (= hlavný bod).

### 9.5 Citlivosť na malé chyby detekcie rohov a hrán

Na reálnych dátach pre hlavný odhad (`work/46_sensitivity_main.py`): každá porucha sa aplikuje na pozorovania a celý fit
sa zopakuje (vrátane váh blokov).

| porucha | veľkosť | Δf [px] | Δcx, Δcy [px] | Δk1 | zmena zobrazenia stred / pás / rohy [px] |
|---|---|---|---|---|---|
| dilatácia čierneho štvorca | +0,3 px na stranu | −1,03 | +0,02; −0,01 | +0,0004 | 0,07 / 0,33 / 0,45 |
| erózia čierneho štvorca | −0,3 px na stranu | −0,28 | −0,11; −0,09 | +0,0002 | 0,02 / 0,08 / 0,09 |
| posun všetkých rohov a strán v x | 0,2 px | −0,08 | +0,01; +0,01 | +0,0000 | 0,01 / 0,02 / 0,04 |
| posun všetkých rohov a strán v y | 0,2 px | −0,29 | −0,00; −0,01 | +0,0001 | 0,02 / 0,09 / 0,12 |
| zvislá mierka rohov (rolling-shutter) | 5·10⁻⁴ | +0,46 | −0,01; +0,01 | −0,0002 | 0,03 / 0,15 / 0,20 |
| zvislá mierka celého obrazu | 5·10⁻⁴ | +0,87 | +0,00; −0,15 | −0,0001 | 0,07 / 0,29 / 0,54 |
| ArUco rohy namiesto hranových | 0,44 px RMS | +0,03 | −0,01; −0,01 | −0,0000 | 0,00 / 0,01 / 0,01 |
| rohy opravené o posun hrán k čiernej | 0,96 px RMS | +0,01 | −0,02; −0,01 | +0,0000 | 0,00 / 0,01 / 0,01 |
| strany nálepiek opravené o posun hrán k čiernej | 0,66 px von | −3,54 | −0,10; −0,18 | +0,0017 | 0,26 / 1,11 / 1,50 |
| rohy aj strany opravené | – | −3,53 | −0,11; −0,19 | +0,0017 | 0,25 / 1,10 / 1,49 |
| hrany posunuté k tmavej strane | +0,3 px | +0,28 | +0,05; +0,07 | −0,0002 | 0,02 / 0,08 / 0,06 |
| hrany posunuté k svetlej strane | 0,3 px | −0,27 | −0,06; −0,07 | +0,0002 | 0,02 / 0,08 / 0,05 |
| hrany posunuté von od stredu | 0,3 px | +0,13 | +0,02; +0,19 | +0,0001 | 0,02 / 0,05 / 0,16 |
| jednotlivé fotky namiesto priemeru 7 | 7 fotiek | −0,02 … +0,01 | | | max 0,00 / 0,01 / 0,01 |

**Chyby detekcie sú malé** voči modelovej a geometrickej neistote: poruchy 0,2–0,3 px menia zobrazenie v páse vozíkov
nanajvýš o 0,33 px. Najväčší je vplyv známeho posunu hrán k čiernej (0,66 px, kap. 3) na stranách
nálepiek: oprava by posunula f o −3,5 px a zobrazenie o ≤ 1,5 px. Hlavný výsledok používa rohy aj strany
bez tejto opravy (merané rovnako; pri rohoch oprava nemení nič), rozdiel je hlboko pod celkovou neistotou.
Pri samotných nálepkách sú citlivosti väčšie (až 2–3 px zobrazenia v páse, desiatky px v rohoch).

### 9.6 Vizuálna kontrola: mriežka lokácií, model vozíkov a rovné čiary (`work/70_render_grid.py`, `work/71_grid_checks.py`)

S hlavným objektívom a pózami vozíkov z hlavného fitu som do fotky vykreslil:
* **mriežku lokácií na podlahe** po 0,5 m v rovine podlahy vozíkov (`results/grid_overview.png`). Mriežka je zarovnaná na
  bielu pásku: stredové čiary oboch pások sú osi (červené) a začiatok je v ich krížení;
* **drôtený model oboch vozíkov podľa výkresu** (vrchná doska, police A–E, stĺpiky, pôdorys) a predpovedané nálepky;
* **predĺženia rovných hrán** (`results/grid_lines.png`): každou overenou hranou vedie priamka v 3D, ktorá je vykreslená
  cez objektív a predĺžená o 1,2-násobok dĺžky hrany na obe strany;
* **obraz po odstránení skreslenia** s tou istou mriežkou (`results/grid_undistorted.png`) a zväčšené výrezy
  (`results/grid_crops.png`).

**Čo ukazuje vizuálna kontrola:**
* Predĺžené rovné čiary ležia na hranách v celej dĺžke, aj pri silno zakrivených hranách na okrajoch obrazu. Príklady:
  tyč regálu na ľavom okraji (~600 px), koľajnice a stĺpik vpravo, pásky na podlahe, lemy políc. Tam, kde hrana
  pokračuje mimo sledovaného úseku (napríklad modrá čiara vľavo dole), predĺženie pokračuje presne po nej.
  Odchýlka bodov hrany od krivky je v mediáne 0,45 px, najviac 2,1 px (vzorkovanie krivky pridáva ~0,3 px).
* Os mriežky ide po bielej páske od kríženia až po spodný okraj obrazu. Priečna os ide po priečnej páske a vľavo
  pokračuje rovnobežne s modrou čiarou. Mriežka sa k okrajom zakrivuje presne podľa skreslenia.
* Model vozíkov z výkresu: lemy políc a stĺpiky sedia do ±5 px (lemy vozíka 80 v radoch B a E +9 px). Koncové dosky
  (vrchná doska) sú mimo o 17–28 px a predpovedané top-nálepky o 5–15 px. Je to nesúlad geometrie z kap. 5, nie chyba
  objektívu (rovné čiary v tých istých miestach sedia).

**Číselné kontroly, ktoré fit nevynucuje** (spätné premietnutie na podlahu, `work/cache/grid_checks.json`, citlivosť na f
v `work/cache/grid_fsens.json`):
* **Šírka bielej pásky** na podlahe vychádza 50,0 mm a je rovnaká po celej dĺžke 1356 mm
  (profil 49,9 / 49,9 / 50,1 / 50,0 / 50,0 / 50,0 mm). Priečna páska má 50,9 mm, modrá čiara 48,1 mm. Bežná páska
  má 50 mm. Absolútnu šírku fit nevynucuje, vychádza z objektívu a z mierky vozíkov podľa výkresu. Na f je však citlivá
  len slabo: pri pevnom f 1400 / 1540 px (ostatné dofitované) vychádza 51,3 / 48,8 mm, teda
  ~1,8 mm na 100 px. Nominálnu šírku pásky nepoznám, preto je to len kontrola konzistencie, nie meranie f.
* **Uhol medzi dvoma smermi bielej pásky** je 89,8°. Oba smery majú vo fite vlastné parametre, takže kolmosť
  nie je vynútená. Pásky tak potvrdzujú konzistenciu objektívu a roviny podlahy, na f je však uhol takmer necitlivý
  (89,70° pri f 1400 px, 89,91° pri 1540 px).
* **Oba vozíky na jednej podlahe:** pôdorys vozíka 80 vychádza v súradniciach vozíka 310 o 15 až 33 mm
  nižšie a osi Z vozíkov zvierajú 0,63°. Výška kamery nad podlahou je 3,91 m (z vozíka 310) a 3,95 m
  (z vozíka 80).

**Záver:** objektív so zobrazenými parametrami k obrazu vizuálne aj číselne sedí. Rovné čiary sú rovné v celom obraze
vrátane okrajov, čo potvrdzuje skreslenie. Mriežka na podlahe drží tvar aj mierku pásky a pásky sú na seba kolmé.
Kontroly na podlahe však f nespresnia; ohnisko ostáva určené na ± 47 px (kap. 1). Viditeľné odchýlky sú na
koncových doskách vozíkov a zodpovedajú známemu nesúladu geometrie s výkresom.

## 10. Neistota zobrazenia

`results/mapping_uncertainty.png` ukazuje mapu 1σ posunu priemetu po celom obraze v troch zložkách: štatistická
(bootstrap: prevzorkovanie nálepiek spolu s ich rohmi aj stranami a hrán v rámci skupín úbežníkov, 100 replík, každá
s celým fitom vrátane váh), systematická (rozptyl 15 prijateľných alternatívnych metód, modelov a modelových rozhodnutí
voči hlavnému výsledku) a rozpočet spolu (bez syntetického testu a jackknife). Výsledok je najväčšia z hodnôt: rozpočet,
syntetický test (scenár b), jackknife z krížovej validácie (najväčší cez rozdelenia nálepky / rady / vozíky / oblasti hrán).

| oblasť | štatistická | systematická | rozpočet spolu | syntetický test (scenár b, T2) | jackknife CV | **výsledok (1σ)** | max v oblasti |
|---|---|---|---|---|---|---|---|
| stred | 3,1 | 2,5 | 4,0 | 1,6 | 2,9 | **4,0 px** | 5,4 |
| pás vozíkov | 13,6 | 7,1 | 15,3 | 7,0 | 11,7 | **15,3 px** | 22,8 |
| rohy | 18,6 | 10,7 | 21,6 | 10,0 | 16,6 | **21,6 px** | 23,6 |

* **Bootstrap f je asymetrický:** medián 1460 px, 68 % interval 1411–1493 px. V 68 % replík
  chýba strana 80:3/1 alebo 80:7/3 a vtedy je f ≈ 1438 px, s oboma ≈ 1473 px. Hlavná hodnota 1469 px leží na hornej
  strane rozdelenia; ± 47 px je symetrizovaná neistota.
* Definícia: posun priemetu pevného lúča po kompenzácii rotácie kamery. Čistú rotáciu pohltí póza, takže
  ide o chybu, ktorá sa prejaví pri súčasnom odhade pózy. **Bez kompenzácie rotácie** je rozpočet
  32 / 36 / 40 px (stred / pás / rohy). Dominujú ho alternatívy s iným hlavným bodom (cy 426–438,
  pp v strede obrazu); týka sa to len porovnania lúčov bez nového odhadu pózy.
* Dominuje neistota f: 1 % vo f ≈ 4,6 px v páse vozíkov (medián, max 6,4) a ≈ 6,2 px
  v rohoch (pri pevnom skreslení v pixelových jednotkách). Samotné skreslenie (pri pevnom f) je určené na ~3,5 / 2,5 / 7,9 px
  (plumb-line, štatistická ⊕ systematická zložka; len štatistická 1,8 / 1,3 / 5,0 px).
* Výsledná neistota zobrazenia je v každej oblasti najväčšia z troch stĺpcov (rozpočet, syntetický test, jackknife).
* Hrany siahajú do vzdialenosti ~1056 px od hlavného bodu, rohy obrazu sú 1056–1146 px, takže rohy sú len
  mierna extrapolácia.
  Modely vyhodnotené ako prehnuté (modely len z nálepiek bez bariéry) nemajú v rohoch zmysluplné zobrazenie a v štatistike nie sú.

## 11. Čo je určené, čo len konzistentné, čo sa určiť nedá

**Určené z dát:**
* Radiálny profil skreslenia (k1 a k2 spolu; v pixelových jednotkách hlavného výsledku k1/f² = −1,63·10⁻⁷, k2/f⁴ = 2,14·10⁻¹⁴):
  z priamosti 61 hrán s rezíduom 0,22 px, zhodne v dvoch nezávislých implementáciách (samotné hrany: −1,60·10⁻⁷, 1,96·10⁻¹⁴).
* cx = 928 ± 19 px (neistotu určuje jackknife cez oblasti hrán).
* f = 1469 ± 47 px, ale len spojením hrán a nálepiek: úbežníky hrán + strany nálepiek. Hrany samé dávajú f s neistotou
  ±21–48 px (bootstrap, nezávislá implementácia) až ±114–156 px (bootstrap prvej implementácie, jackknife štúdie modelov);
  nálepky samé sú vychýlené.
* Nesúlad geometrie výkresu s obrazom: nezávisí od objektívu (dvojpomer, všeobecná projektívna kamera).

**Len konzistentné s dátami:**
* cy = 503 ± 29 px: nesú ho úbežníky a predpoklad o rovnobežnosti čiar na podlahe; rôzne rozumné predpoklady dávajú
  426–513 px.
* k3 ≈ −0,03 (nevýznamné zlepšenie).
* Opis nesúladu geometrie: top-nálepky vyššie o 60–80 mm voči policiam, sklony dosiek 4–13°. Jediná nezávislá kontrola
  (zadná horná tyč vozíka 80) naznačuje skôr zdvihnuté top-dosky než nižšie police (kap. 5); je to len náznak.

**Z dát sa určiť nedá:**
* p1, p2 (tangenciálne) a fx ≠ fy (pomer strán pixla); predpokladá sa 0 a 1.
* Príčina nesúladu geometrie (rám, výšky políc, uchytenie dosiek).
* Skreslenie za rohmi obrazu (extrapolácia), ani to, či má objektív skutočne tvar Brown k1, k2, alebo iný
  dvojparametrový radiálny tvar. V obraze sa líšia o ≤ 2,7 px (RMS; max 3,9 px) a tento rozdiel je v systematike.

## 12. Čo by meranie najviac spresnilo

Poradie podľa očakávaného prínosu:

1. **Fotky kalibračnej dosky ChArUco** (`spec/charuco-board`) touto kamerou, 15–30 záberov. Doska by mala pokryť celé pole,
   hlavne rohy a okraje, a byť v rôznych sklonoch (±30–45°) a vzdialenostiach. To je jediná cesta, ako určiť f a hlavný bod
   na ±1–2 px a skreslenie až do rohov, a nezávisí od geometrie vozíkov. Pri webkamere treba zapnúť pevné zaostrenie
   a snímať v tom istom režime (1920×1080, rovnaký softvér) ako stanica.
2. **Premerať výšku a sklon povrchov s nálepkami** (meter alebo laserový diaľkomer): výšku povrchu koncových dosiek
   voči policiam A–C na všetkých 4 koncoch a sklon a rovinnosť dosiek. Dáta sa s výkresom rozchádzajú o +60–80 mm
   a 4–13° (kap. 5, pozorovanie); ak sa tento nesúlad vysvetlí, nálepky samy dajú f s presnosťou ~±15 px aj hlavný bod.
3. **Pridať priame 3D referencie so známymi rozmermi v rohoch obrazu:** napríklad dlhé rovné latky alebo pásky na podlahe
   v dvoch kolmých smeroch cez celé zorné pole (priamosť + kolmosť = úbežníky) a zvislé tyče pri okrajoch.
   Dnes sú rohy obrazu pokryté slabo (vľavo hore regál, hore v strede pohyblivá osoba), hlavný bod v y určujú
   najmä úbežníky a jedna skupina čiar na podlahe. Chýba aj úbežník hĺbky vozíka (os Y): dlhá rovná hrana pozdĺž boku vozíka
   (napr. páska na podlahe rovnobežne s vozíkom) by pridala tretí kolmý smer.
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
| 3 | `11_markers_a_guided.py`, `11_markers_b_corners.py`, `11_markers_c_fit.py`, `11_markers_d_diag.py`, `12_review_markers_check.py`, `12_review_markers.py`, `12_review_markers_plot.py`, `93_detections_sides.py` | `results/detections.json`, `markers_*.png` |
| 4 | `10_edges_cart310_trace.py` (+ `_summary`, `_boardcheck`), `10_edges_cart80_trace.py`, `10_edges_scene_trace.py` / `_final.py`, `12_review_edges_{cart310,cart80,scene}.py`, `13_edges_dedupe.py`, `91_merge_edges.py` | `results/edges.json`, `edges_*.png` |
| 5 | `20_markers_ba.py` | metóda M1 |
| 6 | `30_lines_methods.py 40` (40 = počet bootstrap replík; prvá implementácia) a `40_lines_indep_{a,b,c,d,e,g,h,i,j,f}_*.py` (nezávislá implementácia) | M2–M4 |
| 7 | `41_cuboid_{fit,predict,diag}.py` | M5 |
| 8 | `42_geomdiag_{a,b,c,f,d,e,g,h}_*.py` | diagnostika geometrie, M7, M8 |
| 9 | `43_altmodels_{fit,diaggeom,report,review}.py` | M9 |
| 10 | `run_combined.sh` (= `50_combined.py fit`, 4× `boot` po 25 replík, `finalize`; `run_combined_boot.sh` len bootstrap), `51_combined_cv.py all` | hlavná metóda M6, varianty, vplyv blokov, krížová validácia |
| 11 | `SYNTH_T2=1400,935,410,-0.32,0.08` + `44_synth_{a_setup,b_roundtrip,c_sensitivity,d_report}.py` (štúdia všetkých metód); `SYNTH_SETUP=cache/synthetic_setup_main.json python3 44_synth_a_setup.py`, potom `45_synth_main.py 30 4` a `46_sensitivity_main.py` (hlavný odhad) | syntetický test a citlivosti |
| 11b | `60_top_tilt_scan_fixed.py all`, `61_top_sticker_orientation_fixed.py`, `62_top_tilt_methods.py` (what-if so sklonom top-nálepiek; `*_fixed.py` sú skontrolované a opravené kópie, ktoré vytvorili výstupy) | kap. 5.1, `top_tilt_*`, `top_sticker_orientation.*` |
| 11c | `70_render_grid.py`, `71_grid_checks.py`, `72_grid_fsens.py` | vizuálna kontrola: `grid_*.png`, `cache/grid_checks.json` |
| 12 | `92_extract_alternatives.py`, `95_final.py`, `96_plots.py`, `98_finalize_json.py`, `90_compare_methods.py`, `97_report_tables.py`, `99_assemble_report.py` | `results/lens_result.json`, `lens_alternatives.json`, `lens_by_method.json`, grafy, `REPORT.md` |

Zdieľané moduly: `common.py` (model kamery, špecifikácia, kontrola prehnutia), `calib.py` (BA), `edgelib.py` (sub-pixelové hrany),
`lineselfcal.py` (plumb-line, úbežníky, spoločný fit z čiar), `combined.py` (spoločný odhad), `evaltools.py`
(neistota zobrazenia), `markerdata.py`, `linedata.py`, `altmodels.py`, `synthlines.py`, `resultnorm.py` (jednotný tvar výsledkov).

**Rozhodnutia a predpoklady (zapísané podľa zadania):**
* **Rozmery a polohy nálepiek** presne podľa výkresu vo všetkých hlavných výpočtoch; nesúlad je opísaný ako pozorovanie (kap. 5).
* **Chvenie medzi snímkami:** modelujem ho ako zvislý posun a mierku pre každú snímku a pracujem s priemerom 7 snímok.
* **Rezíduá hrán:** merajú sa v skreslenom obraze (forward). Model nesmie byť prehnutý vo vnútri obrazu (bariéra).
* **Hrany koncových dosiek** sa nepoužívajú v úbežníkoch (nie sú rovnobežné s osami vozíka), len na priamosť; preto nie je
  k dispozícii úbežník osi Y.
* **Čiary na podlahe:** biela páska (obe hrany) tvorí jednu skupinu. Modrá čiara vľavo, biela priečna páska
  a modrá čiara nad ňou sa považujú za rovnobežné (variant bez tohto predpokladu je v systematike).
* **Zvislice scény** majú vlastný smer, nie sú viazané na stĺpiky vozíkov (variant so spoločnou zvislicou je v systematike).
* **Váhy v hlavnom fite:** blokové variančné komponenty iterované do konvergencie; všetky rohy nálepiek majú spoločnú váhu.
  Iné rozumné váženia (samostatne top / police, robustné po nálepkách) sú alternatívy v systematike.
* **Neistoty:** konzervatívne najväčšia z hodnôt: rozpočet (bootstrap ⊕ alternatívy), syntetický round-trip hlavného odhadu
  (scenár b), jackknife z krížovej validácie.

## 14. Súbory vo `results/`

* **Výsledky:** `lens_result.json` (hlavný; `uncertainty_details` s kovarianciou, `mapping_uncertainty_details`, `rms_details`
  s reziduami po nálepkách / radoch / vozíkoch / fotkách, `determination`), `lens_alternatives.json` (7 rovnocenne
  podporených variantov v rovnakom tvare), `lens_by_method.json` (každá metóda a variant zvlášť, pole `method` a `method_id`,
  jednotné kľúče neistôt a rovnaká definícia `rms_reprojection_error_px`).
* **Detekcie a hrany:** `detections.json` (rohy nálepiek s priradením k fyzickým rohom, strany čiastočne zakrytých nálepiek),
  `edges.json` (všetky hrany s bodmi a overením).
* **Obrázky:**
  - hrany: `edges_all.png`, `edges_cart310.png`, `edges_cart80.png`, `edges_scene.png`;
  - nálepky: `markers_overlay.png`, `markers_crops.png`, `markers_rect_views.png`, `markers_edge_bias.png`;
  - reziduá a predikcia: `residuals_stickers_main.png`, `predicted_stickers_crops.png`, `residuals_edges_main.png`,
    `markers_residuals.png`, `markers_complete_residuals.png`;
  - profil ohniska: `f_profiles.png`, `combined_f_profile.png`, `markers_f_profile.png`, `cuboid_f_profile.png`;
  - neistota a porovnanie: `mapping_uncertainty.png`, `methods_comparison.png`, `undistorted_mean.png`;
  - diagnostika geometrie: `geomdiag_*.png`, `cuboid_*.png` (`cuboid_overlay.png` je s objektívom metódy M5, nie hlavným);
  - sklon top-nálepiek (kap. 5.1): `top_sticker_orientation.png`, `top_tilt_scan.png`;
  - vizuálna kontrola objektívu (kap. 9.6): `grid_overview.png`, `grid_lines.png`, `grid_undistorted.png`, `grid_crops.png`;
  - čiarové metódy (nezávislá implementácia): `40_lines_indep_*.png`;
  - modely objektívu: `altmodels_*.png`;
  - syntetický test a citlivosti (štúdia všetkých metód): `synthetic_*.png`.

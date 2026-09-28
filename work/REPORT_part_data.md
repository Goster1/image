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
  prijala). Bez nich vychádza f = {{v_nobent_f}} (zmena {{v_nobent_df}} px); variant je zahrnutý v systematike.
* **Hrany koncových dosiek sú rovné, ale nie sú rovnobežné s osami vozíka na presnosť úbežníkov** (napr. predná hrana dosky
  310/X0 je o 5,9° mimo úbežníka piatich lemov políc). Preto ich v skupinách úbežníkov nepoužívam, slúžia len na priamosť.
  Keďže všetky hrany v smere hĺbky vozíka (os Y) sú hrany koncových dosiek, **úbežník osi Y sa nepoužíva** (kap. 6, M3).
* Výstupy: `results/edges.json` (všetky hrany vrátane vyradených, s dôvodmi), `results/edges_cart310.png`,
  `edges_cart80.png`, `edges_scene.png` (s názvami hrán, konečný stav po revízii; `*` = len priamosť),
  `edges_all.png`; výrezy v `work/cache/edgecrops_*`, `review_*`.

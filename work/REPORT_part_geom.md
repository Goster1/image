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
| sklon koncových dosiek pozdĺž X | 0° | zmerané z tvaru nálepiek, vonkajší koniec vyššie: {{tilt_meas_list}}; kap. 5.1 | 0,7–4 px na roh | horný a dolný okraj obrazu |
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
  (cy ≈ 290 px, k2 ≈ +0,37), ktoré síce nie je prehnuté, ale hrany narovná len na {{mk_ppfree_straight}} px RMS (hlavný objektív {{bS}} px).
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

TABLE_TILT_MEAS

* Kontrola: plne viditeľné nálepky na policiach vychádzajú vodorovne v rámci 1,5σ (sklon {{tilt_shelf_range}}°).
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

TABLE_TILT_SCAN

(„len nálepky“ = fit len z nálepiek s pp v strede obrazu; všetky tieto objektívy sú prehnuté vo vnútri obrazu.
„Δ zobrazenia“ je posun voči hlavnému výsledku, medián v páse vozíkov / v rohoch. Úplné tabuľky sú v
`work/cache/top_tilt_scan.md`, grafy v `results/top_tilt_scan.png` a `results/top_sticker_orientation.png`.)

**Záver prepočtu:**
* **Objektív sa sklonom top-nálepiek prakticky nemení.** Vo všetkých {{tilt_n_cfg}} konfiguráciách je f {{tilt_f_min}}–{{tilt_f_max}} px,
  cy {{tilt_cy_min}}–{{tilt_cy_max}} px a posun zobrazenia voči hlavnému výsledku nanajvýš {{tilt_map_band_max}} px v páse vozíkov a
  {{tilt_map_corner_max}} px v rohoch. To je hlboko pod neistotou 15,3 / 21,6 px. Dôvod: f a cy nesú úbežníky hrán a strany
  nálepiek na policiach, zatiaľ čo rohy nálepiek majú v spoločnom fite malú váhu (σ ≈ 10,7 px).
* **Sklon vysvetlí len malú časť nesúladu.** So zmeraným sklonom okolo stredu nálepky klesne RMS rohov zo {{tilt_rms0}} na
  {{tilt_rms_meas_c}} px (top-rad {{tilt_top0}} → {{tilt_top_meas_c}} px). Pri doske otočenej okolo pántu stúpnu stredy nálepiek
  o ~13 mm a RMS klesne na {{tilt_rms_meas_h}} px. Zvyšok zodpovedá výškovému posunu top-nálepiek o +60–80 mm (kap. 5),
  ktorý sklon nevytvorí.
* **Nálepky samé zostávajú so sklonom aj bez neho nekonzistentné s hranami:** fity len z nálepiek dávajú f {{tilt_only_min}}–{{tilt_only_max}} px
  (hrany samé 1436, hlavný výsledok 1469) a objektív sa vo vnútri obrazu prehne.
* Ak by sa vynechali hrany, objektív na sklon reaguje (f 1352–1387 px, 30–45 px zobrazenia). Stabilitu hlavného výsledku
  teda dávajú hrany, nie nálepky.
* Nafitované sklony dosiek (polohy rohov, hlavný objektív) sú +12,3 ± 1,8 / +6,1 ± 2,1 / +15,7 ± 3,4 / −4,7 ± 2,8°
  (80/X0, 80/X1600, 310/X0, 310/X1600). Pri troch doskách sa zhodujú so zmeraným sklonom. Pri 310/X1600 majú opačné
  znamienko; tam je sklon z polohy rohov ovplyvnený výškovým nesúladom a priame meranie z tvaru je spoľahlivejšie.

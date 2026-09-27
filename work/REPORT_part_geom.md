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

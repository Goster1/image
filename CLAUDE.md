# Úloha: nezávislé určenie parametrov objektívu kamery zo staničných fotiek

Toto je **slepé, nezávislé meranie**. Nemáš k dispozícii žiadny predchádzajúci výsledok a ani ho nehľadaj
(nepoužívaj webové vyhľadávanie čísel tejto konkrétnej inštalácie, nehádaj „očakávanú“ hodnotu).
Všetko odvoď z dát v tomto priečinku. Výsledok sa neskôr porovná s iným, nezávislým meraním,
preto je dôležitejšia **poctivo vyčíslená neistota** než „pekné“ číslo.

## Cieľ

1. Čo najpresnejšie určiť vnútorné parametre kamery pre obraz 1920×1080:
   ohnisko (fx, fy v px), hlavný bod (cx, cy), radiálne skreslenie (k1, k2, k3), prípadne tangenciálne (p1, p2)
   – v konvencii OpenCV (pinhole + Brown-Conrady).
2. **Zvoliť model primerane dátam** (napr. k1 len / k1+k2 / voľný hlavný bod ...) a zdôvodniť to –
   parametre, ktoré dáta neurčia, fixuj a povedz prečo.
3. **Overiť výsledok**: neistota každého parametra a hlavne neistota samotného zobrazenia
   (o koľko px sa môže mýliť projekcia v rôznych častiach obrazu – stred, pás kde sú vozíky, rohy).
   Minimálne:
   - reziduá (RMS, max) po nálepkách / fotkách / vozíkoch,
   - profil alebo pásmo ohniska (ako rýchlo rastie chyba pri posune f),
   - krížová validácia: vynechať nálepku / riadok / vozík a predpovedať ju z ostatných,
   - syntetický round-trip test tvojho postupu (vyrob dáta zo známeho objektívu so šumom porovnateľným
     s reálnym, over, že postup ho vráti a s akou chybou),
   - ako citlivo výsledok reaguje na malé chyby v detekcii rohov a hrán (geometria vozíka je daná presne, viď nižšie).
4. Jasne oddeliť: čo je z dát **určené**, čo je len **konzistentné** s dátami a čo sa z nich určiť **nedá**.

## Nezávislé metódy – buď kreatívny a vyskúšaj všetko

Nálepky nie sú jediný zdroj informácie. Každú metódu, ktorá dá odhad, dotiahni do čísla
s neistotou a výsledky všetkých metód na konci porovnaj v jednej tabuľke. Ak sa nezhodujú, hľadaj príčinu.

- **Hrany políc a vozíka nájdi sám, vizuálne v obraze.** Zdetekuj ich (napr. Canny, LSD, Hough,
  subpixelové dohľadanie hrany pozdĺž normály) a prekontroluj na výrezoch fotiek, že sú to naozaj
  hrany konštrukcie, a nie tovar, tieň alebo odlesk. Potom ich použi:
  - **priamosť čiar (plumb-line)**: skutočne rovné hrany musia byť po odstránení skreslenia rovné,
    z toho vychádza skreslenie (k1, k2, stred skreslenia) bez potreby rozmerov;
  - **úbežníky** troch navzájom kolmých smerov vozíka (šírka, hĺbka, zvislica): z nich ohnisko
    a hlavný bod;
  - **model vozíka ako kvádra s policami**: hrany políc a rámu napasuj spolu s nálepkami, pričom
    rozmermi presne podľa výkresu (rozmery NIE sú voľné parametre); skontroluj, či hrany políc padnú
    tam, kam ich model predpovedá.
- Ďalšie nápady, ktoré môžeš skúsiť: iné rovné prvky scény (podlahové čiary, regály, steny), obrysy
  nálepiek ako malé štvorce (tvar, nielen stred), porovnanie oboch vozíkov (ten istý vozík,
  iná póza), zhoda výšky kamery a sklonu podlahy medzi vozíkmi, rôzne modely objektívu,
  robustné odhady (RANSAC, vynechávanie odľahlých bodov), bootstrap cez nálepky/hrany, a všetko,
  čo ťa napadne.
- Nálepky a hrany sú nezávislé zdroje. Ak sa výsledok z hrán a z nálepiek líši, je to cenná informácia
  o geometrii alebo o modeli objektívu. Napíš, čo z toho vyplýva.

## Dáta

- `data/stills/` – 7 fotiek 1920×1080 zo staničnej kamery (Windows aplikácia Kamera, 2026-08-11 08:58),
  pevná kamera nad pracoviskom, v zábere dva zaparkované vozíky (č. 80 a č. 310) s ArUco nálepkami.
  Scéna je statická, fotky sú v rozpätí ~34 s.
- `spec/cart-marker-layout.json` – rozmiestnenie nálepiek na vozíku (súradnice stredov v mm v súradnicovom
  systéme vozíka), slovník, veľkosti. **Vozík aj police sú na 100 % presne podľa výkresu** (šírka,
  hĺbka, výška hornej dosky aj výšky všetkých políc). Prevádzkovateľ to potvrdil. Rozmery ber ako
  presné a neodhaduj ich. Ak by sa dáta s nejakým rozmerom zdanlivo nezhodovali, hlavný výsledok
  aj tak počítaj s rozmermi z výkresu a nezhodu len opíš v REPORT.md ako pozorovanie (kde, koľko px).
  **Nálepky sú na 100 % nalepené presne na opísaných pozíciách, na opísanom povrchu (horná doska
  alebo daná polica).** Prevádzkovateľ to overil a potvrdil. Chybu nalepenia nepredpokladaj a nepoužívaj
  ju ako vysvetlenie rozdielov. Jediné, čo pri nálepke nie je predpísané, je jej natočenie v rovine
  (môže byť otočená o 90/180/270°).
- `spec/charuco-board/` – definícia tlačenej kalibračnej dosky ChArUco (A4, 10×7 polí po 27,686 mm,
  DICT_5X5_50). `data/charuco_photos/` – fotky tejto dosky zo staničnej kamery, **ak sú priložené**
  (ak je priečinok prázdny, doska zatiaľ nebola odfotená; pracuj len s vozíkmi).
- Kamera: Tracer WEB007 (webkamera 2 MP). Údaj výrobcu o zornom uhle je marketingový a **neoverený** –
  nepoužívaj ho ako fakt, len ako nanajvýš veľmi voľný štartovací odhad.

## Prostredie

Python 3 + `pip install -r requirements.txt` (opencv-contrib-python, numpy, scipy, Pillow, matplotlib).
Kód si napíš sám do `work/`. Nič zo `data/` a `spec/` neprepisuj.

## Výstup (povinný)

Do `results/`:

1. `lens_result.json` – presne v tomto tvare (OpenCV poradie koeficientov):
   ```json
   {
     "image_width": 1920, "image_height": 1080,
     "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
     "dist_coeffs": [k1, k2, p1, p2, k3],
     "model": "napr. k1 only, pp fixed at centre",
     "rms_reprojection_error_px": 0.0,
     "uncertainty": {"fx_px_1sigma": 0.0, "k1_1sigma": 0.0, "...": "..."},
     "mapping_uncertainty_px": {"centre": 0.0, "cart_band": 0.0, "corners": 0.0},
     "data_used": "ktoré fotky / nálepky / body",
     "geometry_assumptions": "potvrď, že si použil rozmery z výkresu; prípadné nezhody opíš v REPORT.md"
   }
   ```
   Ak dáta podporujú viac rovnocenných modelov, daj hlavný do `lens_result.json` a ďalšie do
   `lens_alternatives.json` (zoznam objektov rovnakého tvaru).
2. `REPORT.md` – postup, všetky kontroly a ich čísla, grafy (`results/*.png`: reziduá v obraze,
   profil ohniska, prekrytie predpovedaných nálepok na fotke), závery a zoznam, čo by meranie
   najviac spresnilo (aké ďalšie zábery / merania).
3. `detections.json` – všetky použité detegované rohy (id nálepky, vozík, fotka, 4 rohy v px,
   priradenie rohov k fyzickým rohom), aby sa dal výpočet zopakovať.
4. `edges.json` – nájdené hrany (fotka, čo je to za hranu, body v px) a `results/edges_*.png`
   s hranami vykreslenými na fotke na vizuálnu kontrolu.
5. `lens_by_method.json` – výsledok každej metódy zvlášť (nálepky, plumb-line, úbežníky,
   kombinovaný model, ...) v rovnakom tvare ako `lens_result.json`, s poľom `"method"`.

Pracuj samostatne až do konca; ak narazíš na nejednoznačnosť, zvoľ rozumný predpoklad,
zapíš ho do REPORT.md a pokračuj.

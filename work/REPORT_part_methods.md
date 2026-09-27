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

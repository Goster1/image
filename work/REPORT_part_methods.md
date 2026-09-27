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
  vozíka 310 posunie f o ~100 px, takže **f zo samotných hrán je krehké** (jackknife σ ≈ 110–150 px).
* **M5 – kváder:** s rozmermi z výkresu f = 1359 ± 36 px (bootstrap; ± 58 vrátane geometrie), rezíduá 6 px, sedí na
  bariére proti prehnutiu. Lemy políc sedia na −2…+9 px. Mimo predikcie sú: zadná horná tyč vozíka 80 (−17 px, −56 mm;
  po zdvihnutí top-dosky o +65 mm +1,5 px), zadné pozdĺžniky `cart310_x_back_inner` a `cart80_x_back_low` (−75…−79 px),
  ktoré som priradil hornej zadnej hrane, hoci sú to zjavne nižšie rúrky rámu (výkres ich neobsahuje), `cart80_x_back_rail_in`
  (−29 px) a `cart310_x_B_inner_fall` (+13 px, vnútorná hrana ohybu lemu). Podrobne `work/cache/cuboid_offsets.md`.
* **M6 – kombinovaný (hlavný):** jeden fit všetkých blokov: rohy nálepiek (M), strany čiastočne zakrytých nálepiek (L),
  úbežníkové skupiny hrán (V) a priamosť hrán koncových dosiek a scény (S). Váhy blokov sú **variančné komponenty**
  iterované do konvergencie (relatívna zmena σ bloku < 0,5 %; každý blok má potom redukované χ² ≈ 1): σ rohov {{sigM}} px (na súradnicu),
  strán {{sigL}} px, úbežníkov {{sigV}} px, priamosti {{sigS}} px. Všetky nálepky majú v rámci bloku rovnakú váhu;
  rozmery z výkresu sa nemenia. Keďže nesúlad geometrie zväčší rozptyl rohov, dostanú rohy ako blok malú váhu.
  **Vplyv jednotlivých blokov na f** (fit bez bloku, váhy ostatných blokov ponechané):

  | vynechaný blok | f | cx | cy | k1 |
  |---|---|---|---|---|
  | – (hlavný) | {{f1}} | {{cx1}} | {{cy1}} | {{k1_3}} |
  | rohy nálepiek (62) | {{inf_nocorners_f}} | {{inf_nocorners_cx}} | {{inf_nocorners_cy}} | {{inf_nocorners_k1}} |
  | strany nálepiek (13) | {{inf_nosides_f}} | {{inf_nosides_cx}} | {{inf_nosides_cy}} | {{inf_nosides_k1}} |
  | úbežníky | {{inf_novp_f}} | {{inf_novp_cx}} | {{inf_novp_cy}} | {{inf_novp_k1}} |
  | len strany 80:3/1 a 80:7/3 (znova vážené) | {{v_nosides_f}} | {{v_nosides_cx}} | {{v_nosides_cy}} | {{v_nosides_k1}} |

  f teda nesú hlavne úbežníky hrán; zo strán nálepiek prispievajú najmä dve strany na konci 80/X0 (+{{v_nosides_df}} px),
  kde podľa kap. 5 nesedí pomer rozostupov políc. Preto sú v systematike aj tieto varianty: bez týchto dvoch strán,
  samostatné variančné komponenty pre top a pre police (RMS top-rohov {{rb_top}} px na bod, polic {{rb_shelf}} px → f {{v_rowblocks_f}}),
  robustné váhy po nálepkách (f {{v_robust_f}}), stĺpiky oboch vozíkov rovnobežné so zvislicou scény (f {{v_commonv_f}}),
  čiary na podlahe len na priamosť (cy {{v_floor_cy}}) a bez lemu E vozíka 80 (f {{v_nobent_f}}).
* **M7 – dva vozíky:** f, pri ktorom sú normály podlahy z póz oboch vozíkov rovnobežné: 1378 ± 81 px (len konzistencia;
  geometria z výkresu). **Výška kamery nad podlahou** z póz oboch vozíkov sa zhoduje na |Δh| < 5 mm pre každé f
  v rozsahu 1100–1800, je teda konzistentná, ale f neobmedzuje.
* **M8 – tvar nálepiek:** len konzistenčná kontrola; na f príliš slabá (± 170 px).
* **M9 – modely objektívu:** dáta podporujú práve 2 radiálne stupne voľnosti. k3 je len konzistentné, p1, p2 a fx ≠ fy nie sú
  určené (kap. 7).

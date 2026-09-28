## 7. Výber modelu (primerane dátam)

Zo štúdie 16 modelov na rovnakých dátach (`work/43_altmodels_*`, `results/altmodels_*.png`, `work/cache/altmodels_summary.md`):

* **Radiálne skreslenie potrebuje práve 2 stupne voľnosti.** k1 samotné hrany nenarovná (0,29–0,51 px, naráža
  na bariéru prehnutia, ΔQAIC +315, ΔQBIC +302). Všetky dvoj- a trojparametrové radiálne tvary (Brown k1k2, k1k2k3,
  divízny l1l2, racionálny k4, Kannala-Brandt k1k2) dosiahnu 0,228–0,229 px. To je hranica daná vlastnou nerovnosťou hrán.
  Ich zobrazenia sa od Brown k1,k2 líšia o 0,4 / 2,0 / 2,7 px (RMS cez modely; max 0,5 / 2,3 / 3,9 px; stred / pás / rohy).
* **k3** vychádza −0,03 ± 0,02 (štúdia modelov −0,031 ± 0,025; kombinovaný fit {{v_k3}}): je len konzistentné, preto ho
  fixujem na 0 a jeho vplyv je v systematike.
* **p1, p2:** na hranách vychádzajú ~0 (−0,001 ± 0,006; 0,0015 ± 0,003) a fit je nestabilný. S nálepkami s geometriou z výkresu
  vyskočí p1 na 0,009 a f klesne na {{v_p1p2_f}} (kombinovaný fit, cy {{v_p1p2_cy}}; v štúdii modelov 1374), lebo tangenciálne
  členy len pohlcujú nesúlad geometrie. Preto p1 = p2 = 0.
* **fx ≠ fy** nie je určené (fx 1383 ± 91 vs. fy 1426 ± 34 z hrán; kombinovaný fit fx/fy − 1 = {{v_fxfy}}), preto fx = fy.
* **Hlavný bod voľný:** fixácia do stredu obrazu stojí ΔQAIC +21…+24 a zhoršuje krížovú predikciu. cx je určené,
  cy len konzistentné.
* Prevod na OpenCV Brown [k1, k2, p1, p2, k3]: Kannala-Brandt k1,k2, divízny l1,l2 a racionálny k4 sa prevedú s chybou
  ≤ 0,08 px (medián); pri racionálnom k1..k6 je to 0,2–0,45 px. Brown k1, k2 je teda primeraný model.

## 8. Porovnanie všetkých metód

Tabuľka je vygenerovaná zo súborov metód (`work/97_report_tables.py`). Všetky metódy sú aj v `results/lens_by_method.json`
({{n_methods}} záznamov vrátane variantov, jednotný tvar a `method_id`) a na grafe `results/methods_comparison.png`.
Stĺpec „Δ zobrazenia“ je posun zobrazenia metódy voči hlavnému výsledku po kompenzácii rotácie (medián oblasti). Neistoty
metód sú ich vlastné (bootstrap alebo klastrové), okrem hlavného riadku, ktorý má celkovú neistotu.

TABLE_METHODS

**Prečo sa metódy líšia – čo z toho vyplýva:**

1. **Nálepky s geometriou z výkresu verzus hrany (f ~1300–1375 vs. ~1430–1480).** Rozdiel nespôsobuje objektív, ale geometria.
   Nálepky nesedia so žiadnou dierkovou kamerou a samy vynútia skreslenie, ktoré odporuje priamosti hrán. Keď sa uvoľní výška
   top-dosiek (+60–80 mm), dajú nálepky rovnaké skreslenie ako hrany a f = 1430–1495. Syntetický test so zdvihnutými doskami
   (scenár T2_c) reprodukuje presne tento vzor: fity len z nálepiek sú vychýlené o −115 až −143 px vo f, kombinovaný odhad nie
   ({{syn_c_bias_f}} px).
2. **Hrany medzi sebou (f 1425 vs. 1473–1479).** Rozdiel spôsobuje hlavne predpoklad, či majú oba vozíky rovnakú zvislicu.
   Vozík 80 má len dve krátke, takmer kolineárne hrany stĺpika, takže jeho vlastná zvislica je slabo určená. Hrany samotné dávajú f
   krehko: vynechanie jednej krátkej hrany stĺpika posunie f o ~100 px. V hlavnom fite tento predpoklad mení f o
   {{v_commonv_df}} px (variant so spoločnou zvislicou).
3. **Hlavný fit verzus hrany samé (f {{f0}} vs. {{inf_edges_f}}).** Rozdiel spôsobujú strany čiastočne zakrytých nálepiek, hlavne
   dve na konci 80/X0 (kap. 6, M6). Bez nich je hlavný fit na {{v_nosides_f}}.
4. **cy (426–513).** Určujú ho hlavne úbežníky: bez nich vychádza v hlavnom fite cy = {{inf_novp_cy}}. V hlavnom fite
   posunie predpoklad o rovnobežnosti čiar na podlahe cy len o {{v_floor_dcy}} px (čiary len na priamosť: {{v_floor_cy}}).
   Nižšie hodnoty 426–438 vychádzajú z čiarových fitov bez nálepiek so spoločnou zvislicou; 509, ak sa obe časti modrej čiary
   berú ako jedna priamka. Nálepky s uvoľnenou geometriou dávajú cy ≈ 390. **cy je z dát len konzistentné.**
5. **Zvislice scény verzus stĺpiky vozíkov** sa líšia o 0,5–0,8°. Podlaha alebo predmety nie sú presne zvislé, preto
   majú zvislice scény vo všetkých fitoch vlastný smer. V hlavnom fite zvierajú osi Z oboch vozíkov {{zz_angle}}°.

## 9. Overenie

### 9.1 Reziduá

* **Hlavný fit, bloky:** priamosť hrán {{bS}} px, konzistencia s úbežníkmi {{bV}} px, strany zakrytých nálepiek {{bL}} px,
  rohy nálepiek {{bM_pt}} px na bod (v spoločnom fite, póza určená hlavne hranami).
* **Rohy nálepiek s hlavným objektívom a pózou vozíka fitovanou z jeho nálepiek (metóda najmenších štvorcov):**
  **{{rms}} px RMS (max {{rms_max}} px)**. To je nesúlad geometrie, nie šum (šum ~0,15 px). Obrázky
  `results/residuals_stickers_main.png` (šípky ×10) a `results/predicted_stickers_crops.png` (výrezy: detegované zelené,
  predpovedané červené). Reziduá hrán: `results/residuals_edges_main.png`. Obraz po odstránení skreslenia je
  `results/undistorted_mean.png`: hrany políc, pásky aj modrá čiara sú vizuálne rovné.

TABLE_RESID

TABLE_STICKERS

* **Len nálepky, s geometriou z výkresu (M1):** RMS 5,0–5,9 px (max 9,5–11,7 px) pri šume ~0,13 px podľa modelu.
  Najlepší model (k1,k2, pp v strede): po vozíkoch 5,7 / 5,8 px, po radoch top 5,6 / A 8,6 / B 3,4 / C 4,6 px
  (`results/markers_residuals.png`).

### 9.2 Profil ohniska (`results/f_profiles.png`, `results/combined_f_profile.png`)

* **Len nálepky:** RMS má minimum pri f ≈ 1300–1375 a rastie o 0,5–1 px na každých 100 px posunu f. Minimum je plytké a
  vychýlené geometriou.
* **Len hrany:** RMS spoločného fitu z čiar (priamosť + úbežníky, samostatné zvislice; samotná priamosť od f nezávisí) má
  minimum pri f ≈ 1430–1440 a je veľmi plochý: +0,002 px pri ±30 px, +0,005 px pri ±50 px. f zo samotných hrán je slabé.
* **Kombinovaný (hlavný):** f je pevne nastavené a všetko ostatné sa refituje (váhy blokov hlavného fitu). Minimum je pri
  {{prof_min}}. Formálne Δχ² = 1 zodpovedá ±{{sig_cov_f}} px (kovariancia). Pri f {{prof_lo}} / {{prof_hi}} px je Δχ²
  {{prof_dchi_lo}} / {{prof_dchi_hi}}; RMS blokov rohy {{prof_M_lo}} / {{prof_M_hi}} (v minime {{prof_M_0}}), strany
  {{prof_L_lo}} / {{prof_L_hi}} ({{prof_L_0}}), úbežníky {{prof_V_lo}} / {{prof_V_hi}} ({{prof_V_0}}), priamosť
  {{prof_S_lo}} / {{prof_S_hi}} ({{prof_S_0}}) px. Príspevky blokov k Δχ²: pri {{prof_lo}} px strany {{dchi_L_lo}}, rohy {{dchi_M_lo}}, úbežníky {{dchi_V_lo}};
  pri {{prof_hi}} px strany {{dchi_L_hi}}, rohy {{dchi_M_hi}}, úbežníky {{dchi_V_hi}}. Nižšie f teda odmietajú hlavne strany
  nálepiek, vyššie f hlavne úbežníky a menej rohy (samotné úbežníky majú minimum okolo 1440 px, rohy okolo 1310 px, strany
  okolo 1480 px).
  Legenda obrázka: M = rohy nálepiek, L = strany čiastočne zakrytých nálepiek, V = úbežníky, S = priamosť.
  Formálna krivka je úzka; reálnu neistotu (±{{sf0}} px) určuje bootstrap, rozptyl alternatív a jackknife krížovej validácie.

### 9.3 Krížová validácia

* **Len nálepky** (`20_markers_ba.py`): vynechanie jednej nálepky dáva predikčnú chybu 8,4–9,1 px (v rámci fitu 5,0–5,9).
  Vynechanie radu: top 16–22 px, A 10–11, B 4,7–6, C 8–11 px. Vynechanie vozíka (vnútorné parametre z jedného vozíka,
  póza druhého refitovaná): 5,8–6,0 px pre modely s pp v strede, 5,5–87 px pre modely s voľným pp (k1: 9,5 px).
* **Kombinovaný (hlavný)** (`51_combined_cv.py`): vynechá sa rad, vozík, oblasť hrán alebo jedna nálepka, fit sa zopakuje
  (vrátane váh blokov) a vynechané dáta sa predpovedajú. Pri vynechanom rade / nálepke sa použije póza redukovaného fitu,
  pri vynechanom vozíku sa jeho póza refituje z jeho nálepiek. Pre porovnanie je uvedené RMS tých istých bodov v hlavnom fite.

TABLE_CV

  Vynechanie jednej nálepky (20 fitov): f {{loo_f_min}}–{{loo_f_max}} px (jackknife σ_f = {{jk_st_f}} px; leave-one-out odhady
  sú korelované, preto sa rozptyl násobí (n−1)), cy {{loo_cy_min}}–{{loo_cy_max}},
  zobrazenie sa zmení najviac o {{loo_map_c}} / {{loo_map_b}} / {{loo_map_k}} px. f sa pohne o viac ako 5 px len pri
  vynechaní nálepiek {{loo_big}}. Predikčná chyba vynechanej nálepky: police {{loo_shelf_lo}}–{{loo_shelf_hi}} px,
  top-nálepky {{loo_top_lo}}–{{loo_top_hi}} px; predikcia je v priemere o {{loo_infl}} % horšia než RMS tých istých bodov
  v hlavnom fite, takže fit nie je preučený na jednotlivé nálepky, ale geometria top-dosiek sa z ostatných dát predpovedať nedá.
  Najjasnejšie to ukazuje vynechanie celého top-radu: bez neho majú zvyšné rohy v spoločnom fite {{cv_top_train}} px na bod
  a top-rad sa predpovedá s chybou {{cv_top_held}} px.
* **Jackknife po hranách** (štúdia modelov, spoločné dáta): f 1413–1457 pri vynechaní jedného člena. Samotné hrany:
  vynechanie jednej krátkej hrany stĺpika vozíka 310 posunie f o ~+100 px.

### 9.4 Syntetický round-trip (`work/45_synth_main.py`, `work/44_synth_*`, `results/synthetic_*.png`)

Syntetické dáta majú presne reálnu geometriu a viditeľnosť (tie isté nálepky a rohy, tie isté hrany prevedené na ideálne priamky)
a vznikli zo známeho objektívu. Pravda T2: f = 1420, pp (935, 505), k1 −0,33, k2 0,09 (blízko odhadov zo samotných hrán;
  od hlavného výsledku sa f líši o {{t2_diff}} px); T1: f = 1250,
pp (955, 530), k1 −0,26, k2 0,06. Scenáre:

* **a** – len šum detekcie;
* **b** – šum detekcie + náhodná odchýlka 3D bodov od výkresu (σ 5 mm na nálepku) a výšok radov (σ 33 mm),
  nastavená tak, aby RMS nálepiek bolo ako v reálnych dátach (5,1–5,4 px). Je to štatistický model nesúladu neznámeho
  pôvodu, nie tvrdenie o nalepení. K tomu ohyby a smerové chyby hrán z reálnych reziduí;
* **c** – systematicky zdvihnuté top-dosky podľa diagnostiky (kap. 5).

**Hlavný odhad presne v tej podobe, v akej dal výsledok** (`45_synth_main.py`, váhy blokov iterované do konvergencie):

TABLE_SYNTH

* **Postup vráti známy objektív.** V scenároch a a c je hlavný odhad nevychýlený (f {{syn_a_bias_f}}, resp. {{syn_c_bias_f}} px).
  V scenári b je odchýlka f {{syn_b_bias_f}} ± {{syn_b_bias_se}} px (T2) a {{syn_t1_bias_f}} ± {{syn_t1_bias_se}} px (T1);
  {{syn_bias_comment}} Rozptyl f v scenári b (T2) je {{syn_b_std_f}} px.
* **Scenár b nereprodukuje štruktúru reziduí hlavného fitu:** medián σ blokov rohy / strany / úbežníky / priamosť je
  {{syn_sig_M}} / {{syn_sig_L}} / {{syn_sig_V}} / {{syn_sig_S}} px, reálne {{sigM}} / {{sigL}} / {{sigV}} / {{sigS}} px; scenár c
  neobsahuje nesúlad rozostupov políc na konci 80/X0, kde ležia dve strany nálepiek posúvajúce f o +{{v_nosides_df}} px.
  RMSE zo syntetického testu ({{syn_b_n}} replík, samo s neistotou ~±{{syn_rmse_relerr}} %) je preto len dolná hranica.
  RMSE f rastie približne s f (T1: {{syn_t1_rmse_f}} px pri 1250, T2: {{syn_t2_rmse_f}} px pri 1420).
* **Výsledná neistota je najväčšia z troch hodnôt:** empirický rozpočet (bootstrap ⊕ rozptyl alternatív), RMSE zo scenára b
  (T2) a jackknife z krížovej validácie (kap. 9.3, 10).
* **Staršia štúdia** (`work/44_synth_*`, `work/cache/synthetic_report.md`, `results/synthetic_*.png`; pravda T2: f = 1400,
  pp (935, 410), k1 −0,32, k2 0,08; 9 replík) porovnala všetky metódy: fity len z nálepiek sú
  v scenári b rozptýlené o 65–100 px vo f (v rohoch 45–186 px zobrazenia) a v scenári c vychýlené o −115 až −143 px;
  spoločný fit z čiar má v scenári b RMSE f ~27 px. **Formálne kovariancie podhodnocujú chybu 9–15×**, bootstrap ju naopak
  skôr nadhodnocuje (pomer 0,4–1,0). Na reálnych dátach je bootstrapový rozptyl f hlavného odhadu {{boot_std_f}} px,
  {{boot_vs_syn}}
* Metóda plumb-line s pevným stredom (stred obrazu) je vychýlená, ak skutočný stred nie je v strede obrazu
  (T2: k1 o −0,018). Preto je stred skreslenia v hlavnom fite voľný (= hlavný bod).

### 9.5 Citlivosť na malé chyby detekcie rohov a hrán

Na reálnych dátach pre hlavný odhad (`work/46_sensitivity_main.py`): každá porucha sa aplikuje na pozorovania a celý fit
sa zopakuje (vrátane váh blokov).

TABLE_SENS

**Chyby detekcie sú malé** voči modelovej a geometrickej neistote: poruchy 0,2–0,3 px menia zobrazenie v páse vozíkov
nanajvýš o {{sens_max_band}} px. Najväčší je vplyv známeho posunu hrán k čiernej (0,66 px, kap. 3) na stranách
nálepiek: oprava by posunula f o {{sens_sides_df}} px a zobrazenie o ≤ 1,5 px. Hlavný výsledok používa rohy aj strany
bez tejto opravy (merané rovnako; pri rohoch oprava nemení nič), rozdiel je hlboko pod celkovou neistotou.
Pri samotných nálepkách sú citlivosti väčšie (až 2–3 px zobrazenia v páse, desiatky px v rohoch).

### 9.6 Vizuálna kontrola: mriežka lokácií, model vozíkov a rovné čiary (`work/70_render_grid.py`, `work/71_grid_checks.py`)

S hlavným objektívom a pózami vozíkov z hlavného fitu som do fotky vykreslil:
* **mriežku lokácií na podlahe** po 0,5 m v rovine podlahy vozíkov (`results/grid_overview.png`). Mriežka je zarovnaná na
  bielu pásku: stredové čiary oboch pások sú osi (červené) a začiatok je v ich krížení;
* **drôtený model oboch vozíkov podľa výkresu** (vrchná doska, police A–E, stĺpiky, pôdorys) a predpovedané nálepky;
* **predĺženia rovných hrán** (`results/grid_lines.png`): každou overenou hranou vedie priamka v 3D, ktorá je vykreslená
  cez objektív a predĺžená o 1,2-násobok dĺžky hrany na obe strany;
* **obraz po odstránení skreslenia** v celom rozsahu snímky s tou istou mriežkou (`results/grid_undistorted.png`)
  a zväčšené výrezy (`results/grid_crops.png`).

**Čo ukazuje vizuálna kontrola** (vlastná kontrola výrezov a dve nezávislé kontroly celého obrazu po dlaždiciach;
odchýlky merané kolmo na priamku, nie odhadom okom):
* **Rovné čiary:** predĺžené priamky ležia na hranách v celej dĺžke, aj pri silno zakrivených hranách na okrajoch obrazu:
  tyč regálu na ľavom okraji, koľajnice a stĺpik vpravo, pásky na podlahe, lemy políc. Odchýlka bodov hrany od krivky
  je v mediáne {{grid_line_med}} px, najviac {{grid_line_max}} px. Zvyšok zakrivenia je pri 54 zo 61 hrán pod 0,5 px, najviac 0,8 px,
  hoci objektív tie hrany ohýba až o 16 px (tyč regálu 16,3 px, zvyšok −0,02 px).
* **Hrany, ktoré sa pri fite nepoužili:** ~150 ďalších rovných úsekov nájdených v obraze bez skreslenia, vrátane hrán pri
  okraji obrazu (vzdialenosť 700–1050 px od hlavného bodu), sedí s predpovedaným ohybom do ±0,35 px.
* **Predĺženie cez medzeru:** priamka vedená modrou čiarou vľavo (x 172–434) a predĺžená cez objektív o 450–700 px dopadne
  na pokračovanie tej istej čiary za vozíkom 310 s odchýlkou 0,6–1,8 px. Bez korekcie skreslenia by minula o 15–18 px.
  Objektívy nafitované len na nálepky (f ≈ 1330–1360, k1 ≈ −0,21) minú o 3–8 px a sú tým vylúčené. Rovnaký test so škárou
  v podlahe je citlivý na cy: nula vychádza pri cy ≈ 510–525, hlavná hodnota 503 je v rámci neistoty ±29 px.
* **Mriežka na podlahe:** os mriežky ide po stredovej čiare bielej pásky od rohu (L) po spodný okraj obrazu s odchýlkou
  do 0,2 px. Priečna os ide po priečnej páske do 1,2 px. Hrany modrej čiary vľavo sú s mriežkou rovnobežné do 1 px na
  240 px. Pomer bunky mriežky k šírke pásky je pozdĺž pásky stály (10,3–10,6).
* **Model vozíkov z výkresu:**
  - Lemy políc vozíka 310 sedia do 0–5 px.
  - Vozík 80 je ako celok posunutý o 3–10 px: lemy aj nálepky na policiach vychádzajú posunuté rovnako, bez zvyšku
    zakrivenia. Je to kompromis pózy tohto vozíka, ktorú ťahajú top-nálepky, nie chyba objektívu.
  - Koncové dosky (vrchná doska) sú mimo o 15–30 px, rovnomerne po celej dĺžke.
  - Predpovedané top-nálepky sú mimo o 9–26 px, všetky smerom od úbežníka stĺpikov. Susedné nálepky na tom istom mieste
    obrazu sa líšia o 15–20 px, čo objektív spôsobiť nemôže.
  - Toto je nesúlad geometrie z kap. 5, nie chyba objektívu.
* **Čo sa overiť nedá:**
  - Úplné rohy obrazu ďalej ako ~1050 px od hlavného bodu, kde nie je žiadna rovná štruktúra s merateľným ohybom.
  - Model s k3 sedí na všetky hrany rovnako dobre a od hlavného sa tam líši o 6 px (r = 1100 px) až 17 px (krajný roh).
  - Zobrazenie v krajných rohoch je teda extrapolácia.

**Číselné kontroly, ktoré fit nevynucuje** (spätné premietnutie na podlahu, `work/cache/grid_checks.json`, citlivosť na f
v `work/cache/grid_fsens.json`):
* **Šírka bielej pásky** na podlahe vychádza {{grid_tape_w_V}} mm a je rovnaká po celej dĺžke {{grid_tape_len_V}} mm
  (profil {{grid_tape_prof_V}} mm). Priečna páska má {{grid_tape_w_H}} mm, modrá čiara {{grid_blue_w}} mm. Bežná páska
  má 50 mm. Absolútnu šírku fit nevynucuje, vychádza z objektívu a z mierky vozíkov podľa výkresu. Na f je však citlivá
  len slabo: pri pevnom f {{fs_f_lo}} / {{fs_f_hi}} px (ostatné dofitované) vychádza {{fs_w_lo}} / {{fs_w_hi}} mm, teda
  ~{{fs_w_slope}} mm na 100 px. Nominálnu šírku pásky nepoznám, preto je to len kontrola konzistencie, nie meranie f.
* **Uhol medzi dvoma smermi bielej pásky** je {{grid_angle}}°. Oba smery majú vo fite vlastné parametre, takže kolmosť
  nie je vynútená. Pásky tak potvrdzujú konzistenciu objektívu a roviny podlahy, na f je však uhol takmer necitlivý
  ({{fs_a_lo}}° pri f {{fs_f_lo}} px, {{fs_a_hi}}° pri {{fs_f_hi}} px).
* **Oba vozíky na jednej podlahe:** pôdorys vozíka 80 vychádza v súradniciach vozíka 310 o {{grid_z80_min}} až {{grid_z80_max}} mm
  nižšie a osi Z vozíkov zvierajú {{grid_zz}}°. Výška kamery nad podlahou je {{grid_h310}} m (z vozíka 310) a {{grid_h80}} m
  (z vozíka 80).

**Záver:** objektív so zobrazenými parametrami k obrazu vizuálne aj číselne sedí. Rovné čiary sú rovné v celom obraze
až po ~1050 px od hlavného bodu, čo potvrdzuje skreslenie. Mriežka na podlahe drží tvar aj mierku pásky a pásky sú na seba kolmé.
Kontroly na podlahe však f nespresnia; ohnisko ostáva určené na ± {{sf0}} px (kap. 1). Viditeľné odchýlky sú na
koncových doskách vozíkov a zodpovedajú známemu nesúladu geometrie s výkresom.

## 10. Neistota zobrazenia

`results/mapping_uncertainty.png` ukazuje mapu 1σ posunu priemetu po celom obraze v troch zložkách: štatistická
(bootstrap: prevzorkovanie nálepiek spolu s ich rohmi aj stranami a hrán v rámci skupín úbežníkov, {{n_boot}} replík, každá
s celým fitom vrátane váh), systematická (rozptyl {{n_alts}} prijateľných alternatívnych metód, modelov a modelových rozhodnutí
voči hlavnému výsledku) a rozpočet spolu (bez syntetického testu a jackknife). Výsledok je najväčšia z hodnôt: rozpočet,
syntetický test (scenár b), jackknife z krížovej validácie (najväčší cez rozdelenia nálepky / rady / vozíky / oblasti hrán).

TABLE_BUDGET

* **Bootstrap f je asymetrický:** medián {{boot_med_f}} px, 68 % interval {{boot_lo_f}}–{{boot_hi_f}} px. V {{boot_miss_pct}} % replík
  chýba strana 80:3/1 alebo 80:7/3 a vtedy je f ≈ {{boot_miss_f}} px, s oboma ≈ {{boot_both_f}} px. Hlavná hodnota {{f0}} px leží na hornej
  strane rozdelenia; ± {{sf0}} px je symetrizovaná neistota.
* Definícia: posun priemetu pevného lúča po kompenzácii rotácie kamery. Čistú rotáciu pohltí póza, takže
  ide o chybu, ktorá sa prejaví pri súčasnom odhade pózy. **Bez kompenzácie rotácie** je rozpočet
  {{raw_c}} / {{raw_b}} / {{raw_k}} px (stred / pás / rohy). Dominujú ho alternatívy s iným hlavným bodom (cy 426–438,
  pp v strede obrazu); týka sa to len porovnania lúčov bez nového odhadu pózy.
* Dominuje neistota f: 1 % vo f ≈ {{f1pct_band}} px v páse vozíkov (medián, max {{f1pct_band_max}}) a ≈ {{f1pct_corners}} px
  v rohoch (pri pevnom skreslení v pixelových jednotkách). Samotné skreslenie (pri pevnom f) je určené na ~3,5 / 2,5 / 7,9 px
  (plumb-line, štatistická ⊕ systematická zložka; len štatistická 1,8 / 1,3 / 5,0 px).
* {{syn_vs_budget}}
* Hrany siahajú do vzdialenosti ~{{edge_rmax}} px od hlavného bodu, rohy obrazu sú {{corner_rmin}}–{{corner_rmax}} px, takže rohy sú len
  mierna extrapolácia.
  Modely vyhodnotené ako prehnuté (modely len z nálepiek bez bariéry) nemajú v rohoch zmysluplné zobrazenie a v štatistike nie sú.

## 11. Čo je určené, čo len konzistentné, čo sa určiť nedá

**Určené z dát:**
* Radiálny profil skreslenia (k1 a k2 spolu; v pixelových jednotkách hlavného výsledku k1/f² = {{k1f2}}, k2/f⁴ = {{k2f4}}):
  z priamosti 61 hrán s rezíduom 0,22 px, zhodne v dvoch nezávislých implementáciách (samotné hrany: −1,60·10⁻⁷, 1,96·10⁻¹⁴).
* cx = {{cx0}} ± {{scx0}} px (neistotu určuje jackknife cez oblasti hrán).
* f = {{f0}} ± {{sf0}} px, ale len spojením hrán a nálepiek: úbežníky hrán + strany nálepiek. Hrany samé dávajú f s neistotou
  ±21–48 px (bootstrap, nezávislá implementácia) až ±114–156 px (bootstrap prvej implementácie, jackknife štúdie modelov);
  nálepky samé sú vychýlené.
* Nesúlad geometrie výkresu s obrazom: nezávisí od objektívu (dvojpomer, všeobecná projektívna kamera).

**Len konzistentné s dátami:**
* cy = {{cy0}} ± {{scy0}} px: nesú ho úbežníky a predpoklad o rovnobežnosti čiar na podlahe; rôzne rozumné predpoklady dávajú
  426–513 px.
* k3 ≈ −0,03 (nevýznamné zlepšenie).
* Opis nesúladu geometrie: top-nálepky vyššie o 60–80 mm voči policiam, sklony dosiek 4–13°. Jediná nezávislá kontrola
  (zadná horná tyč vozíka 80) naznačuje skôr zdvihnuté top-dosky než nižšie police (kap. 5); je to len náznak.

**Z dát sa určiť nedá:**
* p1, p2 (tangenciálne) a fx ≠ fy (pomer strán pixla); predpokladá sa 0 a 1.
* Príčina nesúladu geometrie (rám, výšky políc, uchytenie dosiek).
* Skreslenie za rohmi obrazu (extrapolácia), ani to, či má objektív skutočne tvar Brown k1, k2, alebo iný
  dvojparametrový radiálny tvar. V obraze sa líšia o ≤ 2,7 px (RMS; max 3,9 px) a tento rozdiel je v systematike.

# Nezávislé (slepé) určenie parametrov objektívu staničnej kamery

Kamera Tracer WEB007, obraz 1920×1080, 7 fotiek z 2026-08-11 08:58. Všetko je odvodené len z dát v tomto repozitári.
Údaj výrobcu ani žiadne iné čísla zvonka som nepoužil. Kód je v `work/`, výstupy v `results/`.

## 1. Výsledok

**Hlavný výsledok** (`results/lens_result.json`), konvencia OpenCV (pinhole + Brown-Conrady):

| parameter | hodnota | 1σ | stav |
|---|---|---|---|
| fx = fy | **{{f1}} px** | ± {{sf0}} px ({{sfpct}} %) | určené, ale len spojením hrán a nálepiek (kap. 6, M6); samotné hrany dávajú f krehko |
| cx | **{{cx1}} px** | ± {{scx0}} px | určené |
| cy | **{{cy1}} px** | ± {{scy0}} px | len konzistentné (nesú ho úbežníky; rozumné alternatívne predpoklady o čiarach na podlahe a zvisliciach dávajú 426–513) |
| k1 | **{{k1_3}}** | ± {{sk1_3}} | určené spolu s k2 (korelácia {{rho_k1k2}}) |
| k2 | **{{k2_3}}** | ± {{sk2_3}} | určené spolu s k1 |
| p1, p2 | 0 (fixované) | – | dáta ich neurčia (voľné iba pohltia nesúlad geometrie) |
| k3 | 0 (fixované) | – | len konzistentné (−0,03 ± 0,025), zlepšenie nevýznamné; vplyv je v systematike |

`K = [[{{f2}}, 0, {{cx2}}], [0, {{f2}}, {{cy2}}], [0, 0, 1]]`, `dist = [{{k1_4}}, {{k2_4}}, 0, 0, 0]`.
V pixelových jednotkách, ktoré nezávisia od f: k1/f² = {{k1f2}} px⁻², k2/f⁴ = {{k2f4}} px⁻⁴.
Model je invertovateľný až po rohy obrazu (radiálna funkcia nie je prehnutá). k1 a k2 sú silno korelované: na ďalšie výpočty
treba použiť kovarianciu (`uncertainty_details.covariance` v `lens_result.json`) alebo priamo neistotu zobrazenia nižšie,
nie jednotlivé σ ako nezávislé.

**Neistota zobrazenia** (1σ posun priemetu pevného lúča po kompenzácii rotácie kamery; medián oblasti / max):

| oblasť | 1σ [px] | max v oblasti [px] |
|---|---|---|
| stred (r < 150 px) | **{{mc}}** | {{mc_max}} |
| pás vozíkov | **{{mb}}** | {{mb_max}} |
| rohy (200×150 px) | **{{mk}}** | {{mk_max}} |

Pri použití objektívu so súčasne odhadovanou pózou určuje veľkosť chyby hlavne neistota f: 1 % vo f posunie zobrazenie
o ~{{f1pct_band}} px v páse vozíkov (medián, max {{f1pct_band_max}}) a ~{{f1pct_corners}} px v rohoch. Ak by sa lúče porovnávali
bez nového odhadu pózy (bez kompenzácie rotácie), je neistota {{raw_c}} / {{raw_b}} / {{raw_k}} px (stred / pás / rohy),
lebo vtedy sa naplno prejaví neistota hlavného bodu (kap. 10).

**Najdôležitejšie zistenie:** geometria nálepiek podľa výkresu **nie je konzistentná so žiadnou dierkovou kamerou**
(RMS 4,9–5,1 px aj pri úplne voľnej projektívnej kamere 3×4, šum detekcie je ~0,15 px). Top-nálepky vychádzajú voči
nálepkám na policiach o ~60–80 mm vyššie a koncové dosky sú naklonené (kap. 5, diagnostické fity; dvojpomer dáva +50…+85 mm
na troch zo štyroch koncov, na konci 80/X0 0 mm, kde však nesedí pomer rozostupov políc). Podľa zadania rozmery neupravujem:
hlavný výsledok používa **všetkých 20 nálepiek s geometriou z výkresu spolu so 61 overenými hranami konštrukcie**
(tie nepoužívajú žiadne rozmery), váhy blokov dát sú určené z ich vlastného rozptylu (variančné komponenty).
Nálepky samy by s výkresom dali f ≈ 1300–1375 a cy 290–540 podľa modelu, so skreslením, ktoré odporuje priamosti hrán.
Tento výsledok je vychýlený (kap. 5, 9.4) a uvádzam ho len v porovnaní.

**Čo v hlavnom fite nesie ktorý parameter** (kap. 6, M6): skreslenie nesie priamosť hrán; f a cy hlavne úbežníky hrán
(bez nich f = {{inf_novp_f}}, cy = {{inf_novp_cy}}); zo strán nálepiek prispievajú k f hlavne dve strany na konci 80/X0
(bez nich f = {{v_nosides_f}}); 62 rohov nálepiek má na f malý vplyv (bez nich f = {{inf_nocorners_f}}), lebo ich blok má
kvôli nesúladu geometrie malú váhu (σ ≈ {{sigM}} px na súradnicu).

**Porovnanie metód v skratke** (podrobne kap. 8):

| metóda | f [px] | pp [px] | poznámka |
|---|---|---|---|
| **hlavný: nálepky (výkres) + hrany** | **{{f0}} ± {{sf0}}** | ({{cx0}}, {{cy0}}) | |
| hlavný model, samostatné váhy top / police | {{v_rowblocks_f}} | ({{v_rowblocks_cx}}, {{v_rowblocks_cy}}) | variant v systematike |
| hlavný model, stĺpiky ∥ zvislica scény | {{v_commonv_f}} | ({{v_commonv_cx}}, {{v_commonv_cy}}) | variant v systematike |
| hlavný model, podlahové čiary len na priamosť | {{v_floor_f}} | ({{v_floor_cx}}, {{v_floor_cy}}) | variant v systematike |
| len hrany, spoločná zvislica (nezávislá impl.) | 1473 ± 22 | (931, 438 ± 46) | bez rozmerov |
| len hrany, modrá čiara ako jedna priamka | 1479 ± 21 | (939, 509) | bez rozmerov |
| len hrany, samostatné zvislice | 1425 ± 124 | (927, 503) | f nestabilné |
| úbežníky (+ plumb-line) | 1479 ± 48 | (929, 426) | |
| štúdia modelov, spoločné dáta | 1434 ± 43 | (927, 502) | |
| dva vozíky na jednej podlahe | 1378 ± 81 | – | konzistenčná kontrola |
| model vozíka ako kvádra (výkres) | 1359 ± 58 | stred obrazu | vychýlené geometriou |
| len nálepky (výkres) | 1300–1375 | cy 290–540 | vychýlené, RMS 5,0–5,9 px |

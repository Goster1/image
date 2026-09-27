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

# Testovacie video pre falošnú kameru

Chrome vie nahradiť webkameru video súborom — bez hardvéru a bez povoľovacieho
dialógu:

```bash
google-chrome \
  --use-fake-device-for-media-stream \
  --use-file-for-fake-video-capture=/absolutna/cesta/klip.y4m
```

Súbor musí byť vo formáte Y4M (nekomprimovaný YUV4MPEG2). Z bežného videa ho
vyrobíš cez ffmpeg:

```bash
ffmpeg -i klip.mp4 -pix_fmt yuv420p klip.y4m
```

Odporúčaný obsah klipu: pohľad zhora na pracovníka, ktorý po „skene“ položí
predmet najprv do ľavej, potom do pravej lokácie — na klipe si potom v monitore
nakalibruješ zóny a klávesami F1/F2 na demo stránke vynútiš očakávanú stranu,
aby si videl MATCH aj MISMATCH.

Alternatíva bez spúšťacích prepínačov: na karte **Živý náhľad** v monitore
načítaj video súbor priamo (input „testovacie video“) — ide cez tú istú
detekčnú pipeline.

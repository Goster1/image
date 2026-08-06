# Video kontrola pracovníkov

Chrome rozšírenie (Manifest V3), ktoré cez USB webkameru kontroluje, či skladový
pracovník položil naskenovaný produkt na správnu stranu. Kamera je umiestnená nad
obrazovkou a smeruje na pracovníka; lokácie môžu byť **vľavo**, **vľavo + vpravo**
alebo **vľavo + vpravo + za sebou**. Ak pracovník položí produkt na inú stranu,
na obrazovke sa zobrazí červený banner so zvukovým upozornením.

Všetko beží lokálne v prehliadači — sledovanie rúk cez
[MediaPipe HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker),
žiadne cloudové volania.

## Ako to funguje

```
picking stránka (sken) ──vk-scan──▶ content script ──SCAN_EVENT──▶ monitor stránka
                                                                       │ kamera + MediaPipe
                                                                       │ zóny + stavový automat
picking stránka (banner) ◀──vk-result── content script ◀──VERIFICATION_RESULT──┘
```

1. Pracovník naskenuje kód (USB skener píše ako klávesnica — znaky + Enter).
2. Picking stránka zobrazí smer a vyšle udalosť `vk-scan`.
3. Monitor stránka rozšírenia sleduje kamerou ruky pracovníka. Položenie = ruka
   zostane v zóne aspoň `dwellMs` (400 ms) a potom sa stiahne. Zóna „za sebou“
   je heuristika: ruka smerovala do okrajového pásu obrazu a potom ruky zmizli
   z obrazu.
4. Pri nesúlade sa na picking stránke zobrazí banner **„ZLÁ STRANA!“** so zvukom;
   všetko sa zapisuje do logu (export CSV).

## Inštalácia

```bash
npm install
npm run build        # stiahne aj model (~7,5 MB) a vytvorí dist/
```

V Chrome: `chrome://extensions` → zapnúť **Režim pre vývojárov** → **Načítať
rozbalené** → vybrať priečinok `dist/`.

## Spustenie demo picking stránky

```bash
npm run demo         # http://localhost:5173
```

Na stránke:
- naskenuj kód USB skenerom, alebo napíš kód a stlač Enter,
- **F1 / F2 / F3** vynúti stranu vľavo / vpravo / za seba (na testovanie).

## Monitor a kalibrácia

1. Klikni na ikonu rozšírenia — otvorí sa okno **Monitor** (musí zostať otvorené
   počas smeny; stav vidno na odznaku ikony: zelené ON / červené OFF).
2. Na karte **Živý náhľad** vyber kameru a klikni **Spustiť kameru**.
3. Na karte **Kalibrácia**:
   - vyber rozloženie lokácií (len vľavo / vľavo+vpravo / vľavo+vpravo+za sebou),
   - zapni **Upravovať zóny myšou** a potiahni polygóny/vrcholy tak, aby zodpovedali
     skutočným lokáciám v obraze kamery,
   - zamávaj rukou — zelená bodka ukazuje, čo kamera vidí, a zóna sa rozsvieti,
     keď do nej ruka vstúpi,
   - **Dôležité:** zóna VĽAVO má zodpovedať ľavej strane *z pohľadu pracovníka*.
     Ak je obraz opačne, zapni **Zrkadliť náhľad** (zrkadlí sa len náhľad, zóny
     ostávajú v priestore obrazu).
4. Nastavenia (časy, režim detekcie) sa ukladajú per stanica; staníc môže byť viac.

## Testovanie bez hardvéru

- **Falošná kamera:** spusti Chrome s
  `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=/cesta/klip.y4m`
  (pozri `test-assets/README.md`, klip vyrobíš cez `ffmpeg`).
- **Video súbor:** na karte Živý náhľad načítaj ľubovoľné video — beží cez tú istú
  pipeline ako kamera.
- **Unit testy:** `npm test` (stavový automat verifikácie + geometria).

## Integrácia s reálnou WMS aplikáciou

Rozšírenie nie je viazané na demo stránku. Kontrakt:

```js
// stránka vyšle po skene:
window.dispatchEvent(new CustomEvent('vk-scan', {
  detail: { scanId: crypto.randomUUID(), sku: 'ABC123', expectedSide: 'LEFT' } // LEFT | RIGHT | BEHIND
}));

// a môže počúvať výsledok kontroly:
window.addEventListener('vk-result', (ev) => {
  // ev.detail = { scanId, sku, outcome: 'MATCH'|'MISMATCH'|'TIMEOUT'|'SUPERSEDED', expectedSide, detectedSide, ... }
});
```

Ak WMS stránku nemožno upraviť, dá sa rozšíriť content script tak, aby čítal smer
priamo z DOM stránky. V oboch prípadoch treba pridať URL reálnej aplikácie do
`content_scripts.matches` v `extension/manifest.json` a znova spustiť build.

## Známe obmedzenia

- Zóna **„za sebou“** je heuristika (ruky mimo obraz + pohyb v okrajovom páse) —
  môže hlásiť falošné zhody, keď pracovník od stola odíde počas čakania na položenie.
- Ruka držiaca produkt je čiastočne zakrytá; prahy detekcie sú preto nastavené
  voľnejšie a členstvo v zóne sa vyhladzuje cez 3 snímky.
- Ak sa MediaPipe nepodarí načítať, rozšírenie automaticky prejde do režimu
  **detekcie pohybu v zónach** (menej presné, ale funkčné).

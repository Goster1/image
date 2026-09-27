## 12. Čo by meranie najviac spresnilo

Poradie podľa očakávaného prínosu:

1. **Fotky kalibračnej dosky ChArUco** (`spec/charuco-board`) touto kamerou, 15–30 záberov. Doska by mala pokryť celé pole,
   hlavne rohy a okraje, a byť v rôznych sklonoch (±30–45°) a vzdialenostiach. To je jediná cesta, ako určiť f a hlavný bod
   na ±1–2 px a skreslenie až do rohov, a nezávisí od geometrie vozíkov. Pri webkamere treba zapnúť pevné zaostrenie
   a snímať v tom istom režime (1920×1080, rovnaký softvér) ako stanica.
2. **Zmerať skutočnú geometriu nálepiek** (meter alebo laserový diaľkomer): výšku povrchu koncových dosiek
   voči policiam A–C na všetkých 4 koncoch vozíkov, sklon a rovinnosť dosiek, polohu kartičiek na policiach.
   Dáta ukazujú odchýlku +60–80 mm a sklony 4–13° (kap. 5). S opravenou geometriou by nálepky samy dali f
   s presnosťou ~±15 px a hlavný bod.
3. **Pridať priame 3D referencie so známymi rozmermi v rohoch obrazu:** napríklad dlhé rovné latky alebo pásky na podlahe
   v dvoch kolmých smeroch cez celé zorné pole (priamosť + kolmosť = úbežníky) a zvislé tyče pri okrajoch.
   Dnes sú rohy obrazu pokryté slabo (vľavo hore regál, hore v strede pohyblivá osoba), hlavný bod v y určujú
   najmä úbežníky a jedna skupina čiar na podlahe. Chýba aj úbežník hĺbky vozíka (os Y): dlhá rovná hrana pozdĺž boku vozíka
   (napr. páska na podlahe rovnobežne s vozíkom) by pridala tretí kolmý smer.
4. **Nálepky na pevných, rovných podložkách** namiesto papiera na doskách, viac nálepiek na rôznych výškach a
   ďalej od stredu (rohy obrazu); ideálne aj zvisle, aby sa zlepšila kondícia f.
5. **Viac záberov s pohybom vozíka** (ten istý vozík v rôznych polohách a natočeniach). Každá póza pridá nezávislú
   perspektívu a zmenšia sa korelácie f–pp–pózy.
6. **Stabilizácia kamery / snímanie videa bez chvenia:** snímky sa líšia o ±1 px vo zvislom smere (rolling-shutter
   a vibrácie). Priemerovanie viacerých snímok alebo pevnejšie uchytenie znížia šum hrán aj rohov.
7. **Vypnúť doostrenie alebo tónové úpravy vo webkamere, ak sa dá:** hrany sú systematicky posunuté o ~0,66 px k tmavej strane.
   Na priamosť to vplyv nemá, ale na absolútne polohy rohov áno.

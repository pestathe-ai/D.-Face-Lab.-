# D. Face Lab

App Android per sostituire un volto in **foto e video brevi**, ispirata a
[FaceFusion](https://github.com/facefusion/facefusion). Tutto avviene **sul telefono**:
le immagini non vengono inviate a nessun server.

## Come si usa

1. Al primo avvio accetta le regole d'uso e scarica i modelli (~700 MB, una volta sola, meglio col Wi‑Fi).
2. **Il volto da usare**: scegli una foto con il volto ben visibile.
3. **Dove metterlo**: scegli una foto o un video dalla galleria.
4. Premi **Avvia**, poi **Salva in galleria**.

## Cosa fa la v0.1

- Rilevamento volti (SCRFD 2.5g), riconoscimento (ArcFace w600k r50) e scambio (inswapper 128),
  gli stessi modelli usati da FaceFusion, eseguiti con ONNX Runtime sulla CPU del telefono.
- Sostituzione di tutti i volti o solo del più grande.
- Video: fino a 10 secondi (i video più lunghi vengono tagliati), massimo 30 fotogrammi al secondo,
  lato lungo ridotto a 1280 px; l'audio originale viene mantenuto.
- Download dei modelli con ripresa se si interrompe.

## Limiti noti

- Qualità: il modello lavora a 128×128 pixel, quindi su volti grandi il risultato è un po' morbido
  (FaceFusion su PC aggiunge un "migliora volto" che qui non c'è ancora).
- Velocità: indicativamente da mezzo secondo a qualche secondo per volto a seconda del telefono;
  un video di 10 secondi può richiedere diversi minuti. Tieni l'app aperta durante l'elaborazione.
- Memoria: servono circa 1–1,5 GB di RAM libera; su telefoni con poca memoria l'app potrebbe chiudersi.
- Foto in formato HEIC potrebbero non essere lette: usa JPG o PNG.

## Uso responsabile

Usa solo volti di persone che ti hanno dato il consenso. Non creare contenuti per ingannare,
diffamare o molestare, né contenuti sessuali o con minori. Se condividi un risultato, indica che è
un'immagine modificata.

## Licenze

- Codice dell'app: MIT.
- Modelli scaricati da [facefusion-assets](https://github.com/facefusion/facefusion-assets):
  inswapper 128 di InsightFace è **solo per uso non commerciale**.

## Sviluppo

Le cartelle `android/` e `ios/` vengono generate in CI da `tool/setup_platforms.sh`.
Ogni push su `main` compila, testa e pubblica l'APK nella sezione *Releases*.
I test (`test/pipeline_replay_test.dart`) confrontano la pipeline Dart con un'implementazione di
riferimento in Python eseguita con i modelli reali (`tool/fixtures/gen_fixtures.py`).

La chiave di firma dell'APK non è nel repository: viene creata alla prima compilazione e conservata
nella cache privata di GitHub Actions. Se la cache scade (dopo 7 giorni senza compilazioni) viene
creata una chiave nuova e, solo quella volta, bisogna disinstallare la versione precedente.

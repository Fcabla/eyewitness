# EYEWITNESS — cómo funciona (mapa del sistema)

## El principio de diseño

**Author the ground truth, don't detect it.** El motor compone cada cara desde un
vocabulario cerrado de 12 atributos (`game/face.py: VOCAB`), así que siempre sabe la
verdad exacta. Los modelos solo *traducen* — nunca tienen que reconocer nada.

## Flujo de una partida

```
intro ──ROLL──▶ glimpse (3s, CSS animation-delay oculta la cara)
      ──I SAW HIM──▶ testimony (texto libre EN/ES)
      ──SEND──▶ [PARSER] testimonio → JSON de atributos
                   ├── Tier B: MiniCPM5-1B fine-tuneado (ZeroGPU, frases sucias)
                   └── Tier A: matcher de sinónimos (backstop literal, siempre gana donde matchea)
              ──▶ sketch (dibuja SOLO lo que dijiste) + lineup (gr.Gallery PNG)
                   └── lineup.py: distractores desde TUS errores
                       · lo que dijiste MAL → plantado en distractores
                       · lo que NO mencionaste → ejes de confusión
                       · lo que acertaste → poda la rueda (p=0.72, afinado por simulación)
      ──click──▶ verdict: sello + cita con voz (VoxCPM2 pre-render, en memoria) +
                 tabla transparente dijiste/verdad + póster SE BUSCA (PIL) ──▶ next case
```

Rangos: ROOKIE (3s, rueda 4) → DETECTIVE (2s, 6) → INSPECTOR (1.5s, 6 + disfraz) → CHIEF (1s, 8 + disfraz).
"Disfraz" = el culpable cambia UN atributo saliente entre el crimen y la rueda.

## Ficheros

| Fichero | Qué hace |
|---|---|
| `app.py` | UI Gradio: una sola etapa `@gr.render` dirigida por `gr.State` dict (screen=intro/glimpse/testimony/lineup/verdict) |
| `game/face.py` | VOCAB + FaceSpec + renderer SVG paramétrico (estilo retrato robot) |
| `game/render.py` | SVG→PIL compartido (Gallery y póster) |
| `game/casegen.py` | Casos: crimen + culpable + parámetros por rango; carga banco Modal si existe |
| `game/parser.py` | Tier A: sinónimos EN+ES, matching longest-first |
| `game/model.py` | Tier B: MiniCPM5-1B slot-filling (gated: solo ZeroGPU/CUDA/env), merge con prioridad Tier A |
| `game/lineup.py` | Distractores derivados de errores (la mecánica firma) |
| `game/scoring.py` | Nota ponderada por saliencia + rating con humor |
| `game/poster.py` | Póster SE BUSCA compartible (PIL) |
| `modal_factory.py` | Build-time en Modal: banco de casos (llama.cpp+GGUF) + banco de voces (VoxCPM2, anchor a 48kHz) |
| `train/gen_dataset.py` | Dataset sintético: spec→testimonio ruidoso bilingüe (verdad por construcción) |
| `train/train_modal.py` | LoRA en Modal → publica `Fcabla/MiniCPM5-1B-eyewitness` |
| `tests/test_engine.py` | QA: testimonios límite, equidad por simulación (400 partidas), bounds |
| `deploy.sh` | Sube el Space a la org (idempotente) |

## Modelos en runtime (~3.4B total)

- `Fcabla/MiniCPM5-1B-eyewitness` (1.08B, LoRA del MiniCPM5-1B) — parser de testimonios.
  Variable de Space `EYEWITNESS_MODEL_ID` lo selecciona; sin ella usa el base.
- VoxCPM2 (2.29B) — SOLO build-time hoy (banco pre-renderizado); el wav viaja en memoria
  como (48000, int16 array) porque gr.Audio con rutas revienta en el Space.

## Infra

- **Space**: `build-small-hackathon/eyewitness`, ZeroGPU (H200 slice), gating por env.
- **GitHub**: `fcabla/eyewitness` (público).
- **Dataset**: `build-small-hackathon/eyewitness-testimonies`.
- **Modal**: apps `eyewitness-factory` (casos+voces) y `eyewitness-train` (LoRA, A10G).

## Limitaciones conocidas (candidatas a iteración)

1. La voz del veredicto es de un banco de 6 líneas fijas — no referencia TUS errores,
   y el timbre no siempre casa con la cara del sospechoso.
2. El 1B trabaja una sola vez por ronda (parse) — su papel es invisible para el jurado.
3. El glimpse es vulnerable a inspeccionar el DOM (la cara sigue en el HTML borroso).
4. El parser puede alucinar atributos no mencionados (mitigado: Tier A manda donde habló).

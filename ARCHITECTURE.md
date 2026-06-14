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

## Modelos en runtime (~5.7B total)

- `Fcabla/MiniCPM5-1B-eyewitness` (1.08B, LoRA) — parser de testimonios (SOLO sabe slot-filling:
  olvido catastrófico verificado — balbucea dataset en prompts abiertos). OJO: en inferencia
  exige el formato EXACTO de entrenamiento (SYSTEM corto + `Witness testimony: "..."`); con el
  prompt few-shot del base parrotea los valores del ejemplo (fedora fantasma, 12 jun).
- `openbmb/MiniCPM5-1B` base (1.08B) — escribe la COLETILLA personalizada del veredicto desde el
  diff dijiste/verdad (`culprit_taunt`); solo entra si pasa validación + actitud. El chiste
  principal del veredicto es AUTORADO por crimen (`CRIME_TAUNTS` en casegen.py — el lab demostró
  ~3% de acierto en comedia libre del 1B). `enable_thinking=False` obligatorio (modelo razonador).
- `CohereLabs/cohere-transcribe-03-2026` (2B) — ASR del testimonio hablado, EN EXCLUSIVA (sponsor).
  `language` es obligatorio: se transcribe con `en` Y `es` y gana el de mayor log-prob medio.
- VoxCPM2 (2.29B) — voz del veredicto EN VIVO clonando un anchor (`game/voice.py`); banco
  pre-renderizado como fallback. Audio siempre en memoria (48000, int16) — gr.Audio con rutas
  fuera de allowed dirs revienta el render.

**Veredicto**: punchline autorado del crimen + coletilla de memoria (1B vivo si pasa el listón,
autorada si no) → voz viva → banco enlatado → solo texto. Cada `except` loguea su causa
(`[taunt]`/`[voice]`/`[asr]` en logs del Space).

**Patrón ZeroGPU**: en el arranque se precargan a CPU el parser+pulla (model.py), la voz
(voice.py) y el ASR (asr.py) — así el primer testimonio hablado no paga la carga del 2B de
Cohere. Las funciones `@spaces.GPU` (20s parser/pulla/ASR, 30s voz) solo transfieren y generan.
La cuota ZeroGPU admite por duración SOLICITADA — pedir de más = llamadas rechazadas.

## Infra

- **Space**: `build-small-hackathon/eyewitness`, ZeroGPU (H200 slice), gating por env.
- **GitHub**: `fcabla/eyewitness` (público).
- **Dataset**: `build-small-hackathon/eyewitness-testimonies`.
- **Modal**: apps `eyewitness-factory` (casos+voces) y `eyewitness-train` (LoRA, A10G).

## Limitaciones conocidas (candidatas a iteración)

1. La pulla del veredicto es chiste sobre el CRIMEN (banco autorado de 64 líneas + 1B base
   improvisando por encima cuando pasa validación). Riesgo: cuando entra el modelo, la
   calidad es variable (legible pero no siempre tan buena como el banco curado).
2. El glimpse es vulnerable a inspeccionar el DOM (la cara sigue en el HTML borroso).
3. El parser puede alucinar atributos no mencionados (mitigado: Tier A manda donde habló).
4. Crash SIGSEGV espontáneo visto 1 vez en local (C-level, en el writer de stdout durante
   render de voz concurrente) — no reproducido en el Space; observar tras deploy.

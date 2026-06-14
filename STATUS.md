# EYEWITNESS — Estado y backlog de iteración (12 jun 2026, mañana)

## Estado del sistema

| Componente | Estado | Evidencia |
|---|---|---|
| Juego core completo en el Space (ZeroGPU) | ✅ estable | múltiples rondas verificadas en prod |
| Parser fine-tune (`Fcabla/MiniCPM5-1B-eyewitness`) | ✅ en vivo | "caterpillar eyebrows"→retrato correcto |
| Pulla viva (1B base) + voz viva (VoxCPM2 anchor) | ✅ mecánicamente / ⚠️ calidad | logs `[taunt] ok`+`[voice] ok 3-4s`; pero ver backlog F2/F3 |
| Preload canónico ZeroGPU + duraciones 20/30s | ✅ | `[startup] models preloaded`; quota-admisible |
| Cascada de fallbacks | ✅ probada | nunca rompió prod en ningún fallo |
| Banco casos/voces/anchors (Modal) | ✅ | verificados por transcripción whisper |

Latencia percibida: aceptable (palabra de Fernando). GitHub/dataset/model publicados.

## BACKLOG DE FEEDBACK (Fernando, 12 jun) — por orden de ataque

- **F1 · UI: textos invisibles sobre fondo blanco (light mode)** — bug de contraste; además pase
  general /frontend-design ("hay que pulir más"). *Primera tarea: reproducir en light mode.*
- **F2 · Calidad de las pullas**: el 1B confunde la voz narrativa (habla como testigo, no como
  ladrón) y no se percibe la personalización. Diagnóstico en logs: `"A headache! I never noticed
  your fancy hat!"`. Candidatos: prompt más restringido, temperatura menor, o plantilla
  determinista desde el diff + el modelo solo adorna (honest fit). **Probar en local con la 4090.**
- **F3 · Voz: siempre suena la misma (femenina), alta y regular.** Los 3 anchors no se
  diferenciaron — VoxCPM2 LEYÓ las descripciones de estilo pero no las OBEDECIÓ (gamble fallido).
  Plan: con la 4090 local, generar ~20 candidatos de anchor, medir pitch (f0) y elegir 3
  genuinamente distintos (grave/medio/agudo) + normalizar volumen (está clipeando o muy alto).
- **F4 · "No queda claro qué modelo hace qué en cada momento"** — UX de visibilidad del modelo:
  badges por pantalla ("MiniCPM5-1B parseando tu testimonio…"), el cuaderno del retratista
  (mapeo palabra→atributo visible), retrato construyéndose rasgo a rasgo.
- **F5 · Testimonio HABLADO** (idea Fernando): ASR para hablar en vez de escribir.
  Decisión (12 jun): **Cohere Transcribe 2B en exclusiva** (sponsor award; sube runtime
  a ~5.7B, sigue ≤32B sobrado). Sin fallback whisper — no puntúa para ningún premio.
  Idioma del testigo (EN/ES) elegido por log-prob entre ambos decoder prompts.
- **F6 · Imagen realista del sospechoso** (idea Fernando): generar una "foto policial" realista
  desde el retrato/atributos, al final o en el loop. Opciones: img2img/ControlNet sobre el SVG,
  SDXL-Turbo (~3.5B) o FLUX (12B, rompe presupuesto Tiny Titan si computa runtime). Pesada:
  evaluar coste/beneficio DESPUÉS de F1-F4. Posible modo "ficha policial final" como recompensa.

## Recursos nuevos

- **RTX 4090 local disponible** — probar modelos/voces/prompts en local, sin quemar cuota
  ZeroGPU ni ciclos de deploy. (Falta instalar torch CUDA en el venv; los drivers ya están.)

## Lecciones ZeroGPU (no repetir)

1. La cuota ADMITE por duración SOLICITADA vs restante — pedir 90s "por seguridad" hace
   inadmisibles las llamadas. Pedir pequeño (20-30s).
2. Patrón canónico: cargar modelos a CPU en el arranque (`preload()` en `__main__`), `.to(cuda)`
   dentro de la función `@spaces.GPU` — las llamadas pasan de 30-60s a 3-8s.
3. Los usuarios anónimos pueden tener 0s de cuota → la cascada de fallbacks no es opcional.
4. El bloque `__main__` no se ejercita con `from app import demo` — usar `compile()` del fichero.

## Lo que NO está en discusión (congelado)

Mecánica del juego, motor de caras, rueda-desde-errores, scoring transparente, póster,
deploy pipeline. La iteración es: UI/contraste, calidad pulla+voz, visibilidad del modelo,
y (si hay margen) ASR y/o imagen realista. **El vídeo de submission sigue siendo el único
entregable duro pendiente — deadline 15 jun.**

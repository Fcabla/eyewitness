# EYEWITNESS — Estado final del build autónomo (11 jun 2026, noche)

## ✅ HECHO (verificado, commiteado)

| Pieza | Estado |
|---|---|
| Motor del juego (caras SVG paramétricas, casos, rueda desde errores, scoring transparente) | ✅ + QA suite |
| App Gradio completa (5 pantallas, 4 rangos con escalada y disfraces, estética expediente) | ✅ jugada entera en navegador real, 2 veces |
| Parser de testimonios bilingüe — Tier A (sinónimos, instantáneo) | ✅ verificado con español e inglés mezclados |
| Parser Tier B — MiniCPM5-1B slot-filling (ZeroGPU-ready, validado contra vocabulario, backstop Tier A) | ✅ 10/13 en testimonios adversarialmente sucios; e2e por el flujo real de la app; gateado a CUDA (local usa Tier A) |
| Equidad de la rueda afinada por simulación (400 partidas) | ✅ curva 37%→8% por rango (antes: 2% — injugable) |
| Póster SE BUSCA compartible (PIL) | ✅ |
| Audio del veredicto (hook plug-and-play para el banco de voz) | ✅ cableado; render pendiente de Modal |
| Fábrica Modal (banco de casos llama.cpp+GGUF, banco de voz VoxCPM2) | ✅ script listo |
| README del Space (award tags), deploy.sh, requirements | ✅ |
| Pack de submission (guión vídeo 60s, post social, Field Notes completo, checklist) | ✅ SUBMISSION.md + FIELD_NOTES.md |

9 commits. Gotchas de Gradio 6 documentados en FIELD_NOTES (visibility desync → @gr.render; gr.HTML sin scripts → CSS delays; Gallery sin data-URIs → PIL).

## 🔑 TUS 10 MINUTOS (lo único que no puedo hacer yo)

```bash
cd ~/Documents/projects/fcabla/build_small_hackathon/eyewitness

# 1. (2 min) Login HF con token de WRITE:
#    en el prompt de Claude Code:  ! hf auth login

# 2. (1 min) Deploy del Space a la org:
bash deploy.sh
#    luego en Settings del Space: Hardware -> ZeroGPU

# 3. (2 min) Tu usuario de HF en README.md (busca TODO-FERNANDO) y re-run deploy.sh

# 4. (3 min) Modal (banco de casos + voces):
#    ! modal token new
modal run modal_factory.py
bash deploy.sh   # re-sube con assets

# 5. (después) Vídeo según el guión de SUBMISSION.md + post social + pegar FIELD_NOTES.md como blog de la org
```

## Premios que reclama (mapa completo en SUBMISSION.md)
Wood podium · OpenBMB (MiniCPM5-1B core + GGUF/llama.cpp + VoxCPM2) · Modal · Tiny Titan (~3.4B runtime) · Off-Brand · Off the Grid · Llama Champion · Field Notes · Sharing is Caring · Best Demo.

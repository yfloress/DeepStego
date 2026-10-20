# A6 · Lectura de QR (M3)

**Responsable:** IF (70 %) · **Participa:** JM (30 %) · **Objetivo:** O3 · **Requisito:** RF4
**Branch:** `feature/a6-lectura-qr` · **Código:** `src/deepstego/qr.py`

## Qué hace

Recibe el QR revelado S′ que entrega RevealNet (128×128, 1 canal, grises con ruido)
y devuelve el texto del mensaje o el aviso «QR no legible», junto con la imagen del
QR limpio para mostrar en la interfaz.

```python
from deepstego.qr import read_qr

res = read_qr(qr_revelado)   # ndarray o tensor: (128,128), (1,128,128), (1,1,128,128)
res.text      # "Hola TEL-328" o None
res.ok        # True / False
res.message   # el texto o "QR no legible"  → para Gradio (M5)
res.clean     # QR limpio uint8 (640×640)    → para Gradio (M5)
res.threshold # umbral que eligió Otsu       → para la bitácora
```

Para M4, la tasa de lectura es simplemente:

```python
ok = sum(read_qr(s_rev).text == texto for s_rev, texto in pares)
tasa = ok / len(pares)
```

## Pipeline

| Paso | Operación | Por qué |
|---|---|---|
| 1 | Ampliación ×4 (bilineal) | Con 37 módulos en 128 px cada módulo mide ~3,5 px. A esa escala una mediana de 5×5 ya borra módulos sueltos. |
| 2 | Filtro de mediana 7×7 (en la imagen ampliada) | Quita ruido puntual y sal y pimienta sin redondear los bordes como un promedio. |
| 3 | Umbral global de Otsu | Elige solo el corte entre módulos negros y blancos, aunque la red no llegue a 0 y 1. |
| 4 | Margen blanco de 16 px (64 px ampliado) | Zona de silencio que necesita el detector. |
| 5 | `cv2.QRCodeDetector().detectAndDecode` | Lector estándar. |
| 6 | Reintentos si falla: sin mediana, luego mediana 11 | Sin mediana sirve cuando S′ está borroso pero no ruidoso; mediana 11 cuando hay mucho ruido. |

### Cambio respecto de la propuesta

La propuesta (sección 4.2, M3) indica *mediana → Otsu → margen → ×4*. La implementación
usa los **mismos pasos en otro orden** (ampliar primero) y agrega reintentos, porque
las pruebas mostraron una mejora clara (tabla abajo). El orden original sigue
disponible con `read_qr(s, median_ksize=3, upscale_first=False, fallback=False)` para
poder comparar en la bitácora. Conviene mencionarlo en la Bitácora 1.

## Resultados sin red (QR degradados artificialmente)

`python scripts/bench_qr.py --n 100` — 100 QR con textos aleatorios de 1 a 20
caracteres por condición, semilla 0. `c` es el contraste (0,6 = la red no llega a
negro ni blanco puros).

| Condición | Propuesta | A6 | A6 + reintentos |
|---|---:|---:|---:|
| limpio | 100 % | 100 % | 100 % |
| grises (c=0,6) | 100 % | 100 % | 100 % |
| ruido σ=0,10 | 100 % | 100 % | 100 % |
| ruido σ=0,20 | 82 % | 97 % | **100 %** |
| ruido σ=0,30 | 23 % | 45 % | **76 %** |
| blur σ=0,8 | 100 % | 99 % | 100 % |
| blur σ=1,2 | 77 % | 83 % | **98 %** |
| sal y pimienta 2 % | 98 % | 99 % | 100 % |
| sal y pimienta 5 % | 88 % | 79 % | **98 %** |
| mixto (σ=0,15 + blur 0,8 + s/p 2 %) | 47 % | 71 % | **83 %** |

**Lectura:** M3 supera la meta de 90 % de RF4 mientras el ruido de S′ no pase de
σ≈0,2 o un desenfoque de σ≈1,2 px. Con eso se le puede decir al equipo de M2 (YF/DA)
cuánto error puede tener RevealNet: **si el QR revelado se ve como la fila "ruido
σ=0,20", la lectura no será el cuello de botella.**

Limitación: la degradación es sintética. Hay que repetir la medición con los S′
reales apenas exista el primer checkpoint (A9) y ajustar en A11.

## Cómo correrlo

```bash
pip install -e ".[dev]"
pytest tests/test_qr.py -v
python scripts/bench_qr.py --n 200 --csv outputs/a6/bench.csv --save-examples outputs/a6
```

## Pendiente / siguiente

- [ ] Medir con S′ reales del primer modelo entrenado (A9).
- [ ] Si la red produce QR invertidos o con contraste muy bajo, revisar el paso 3.
- [ ] Probar `pyzbar` como lector de respaldo si `cv2.QRCodeDetector` falla en casos que
      se ven bien a simple vista.
- [ ] Coordinar con DA (A4): `make_qr` quedó aquí para poder probar M3; M1 debe usarla
      (o reemplazarla manteniendo la misma convención: blanco = 1, negro = 0, borde 4).

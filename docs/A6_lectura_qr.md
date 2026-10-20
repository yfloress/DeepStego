# A6 · Lectura de QR (M3)

**Responsable:** IF (70 %) · **Participa:** JM (30 %) · **Objetivo:** O3 · **Requisito:** RF4
**Branch:** `m3` · **Código:** `src/deepstego/qr.py`

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

## Cómo comprobar la A6 (Windows, macOS y Linux)

Son cinco pasos. Los pasos 3 a 5 son la comprobación; los dos primeros solo se hacen
una vez por computador.

> **Regla general:** usen siempre `python -m pip` y `python -m pytest`, nunca `pip` ni
> `pytest` solos. Así se usa el Python correcto en cualquier sistema y se evitan los
> errores de "comando no reconocido" o de pip de Homebrew.
>
> En macOS y Linux, si `python` no existe, usen `python3` (solo al crear el entorno;
> una vez activado, `python` funciona).

### Paso 1 · Obtener el código en la branch correcta

Igual en los tres sistemas. Si ya tienen el repo clonado, salten el `git clone`.

```bash
git clone https://github.com/yfloress/DeepStego.git
cd DeepStego
git fetch
git checkout feature/a6-lectura-qr     # después del merge del PR: git checkout main
git pull
git branch --show-current               # debe decir feature/a6-lectura-qr
```

Todos los comandos siguientes se ejecutan **dentro de la carpeta `DeepStego`** (la que
tiene `pyproject.toml`).

### Paso 2 · Crear el entorno e instalar (una sola vez)

**Windows (PowerShell)**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned #en caso que les tire que la ejecución de scripts está deshabilitada en este sistema
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-python numpy "qrcode[pil]" pytest
python -m pip install -e . --no-deps
```

- En `cmd` (no PowerShell) se activa con `.venv\Scripts\activate.bat`.
- Se instala sin PyTorch a propósito: la A6 no lo necesita, y en Windows torch suele fallar
  por rutas largas (ver *Problemas frecuentes*). Para instalar todo, primero habiliten las
  rutas largas y luego usen `python -m pip install -e ".[dev]"`.

**macOS (Terminal)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

**Linux (terminal)**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

En Debian/Ubuntu, si `python3 -m venv` falla, instalen antes `sudo apt install python3-venv`.
En servidores sin pantalla, si `import cv2` reclama por `libGL`, cambien OpenCV por la versión
sin interfaz: `python -m pip install opencv-python-headless`.

**Comprobar la instalación (los tres sistemas):**

```bash
python -c "import deepstego.qr, cv2; print('OK, OpenCV', cv2.__version__)"
```

Debe imprimir `OK, OpenCV x.y.z`.

> Cada vez que abran una terminal nueva, activen el entorno antes de seguir
> (`.venv\Scripts\Activate.ps1` en Windows, `source .venv/bin/activate` en macOS/Linux).
> Se reconoce porque el prompt muestra `(.venv)`.

### Paso 3 · Pruebas automáticas

Igual en los tres sistemas:

```bash
python -m pytest tests/test_qr.py -v
```

**Resultado esperado:** una línea `PASSED` por prueba y al final:

| Instalación | Resumen esperado |
|---|---|
| Con PyTorch (macOS/Linux con `.[dev]`) | `23 passed` |
| Sin PyTorch (Windows rápido) | `22 passed, 1 skipped` |

La prueba saltada (`test_acepta_tensor_torch`) solo verifica tensores de PyTorch; que se
salte no es un error. **Cualquier `FAILED` sí lo es**: copien el mensaje y avisen en el PR.

### Paso 4 · Prueba visual de un QR

Un solo comando, válido en PowerShell, cmd, Terminal de macOS y Linux. Genera un QR,
lo ensucia como lo haría la red, lo limpia y lo lee:

```bash
python -c "import os, cv2, numpy as np; from deepstego.qr import make_qr, simulate_revealed, read_qr; os.makedirs('outputs/demo', exist_ok=True); q = make_qr('Hola TEL-328'); r = simulate_revealed(q, noise_std=0.2, contrast=0.6, blur_sigma=0.6, seed=1); res = read_qr(r); print('Leido:', res.message, '| umbral Otsu:', res.threshold); cv2.imwrite('outputs/demo/1_original.png', (q*255).astype(np.uint8)); cv2.imwrite('outputs/demo/2_revelado.png', (r*255).astype(np.uint8)); cv2.imwrite('outputs/demo/3_limpio.png', res.clean)"
```

**Resultado esperado en la terminal:** `Leido: Hola TEL-328 | umbral Otsu: ...` (umbral cerca
de 128).

**Abrir las imágenes:**

| Sistema | Comando |
|---|---|
| Windows (PowerShell) | `start outputs\demo\1_original.png; start outputs\demo\2_revelado.png; start outputs\demo\3_limpio.png` |
| macOS | `open outputs/demo/*.png` |
| Linux | `xdg-open outputs/demo/3_limpio.png` (una por vez) |

| Imagen | Qué se debe ver |
|---|---|
| `1_original.png` | QR negro sobre blanco, nítido, 128×128. |
| `2_revelado.png` | El mismo QR gris, granulado y algo borroso (como saldría de RevealNet). |
| `3_limpio.png` | QR en blanco y negro puro, 640×640, con borde blanco ancho. |

**Comprobación externa:** abran `3_limpio.png` en pantalla y escanéenlo con la cámara de un
celular. Debe mostrar `Hola TEL-328`.

**Caso de falla esperado:** cambien `noise_std=0.2` por `noise_std=0.5` y repitan. Debe
imprimir `Leido: QR no legible`.

### Paso 5 · Benchmark (tasa de lectura)

Igual en los tres sistemas (tarda 1 a 3 minutos):

```bash
python scripts/bench_qr.py --n 100 --csv outputs/a6/bench.csv --save-examples outputs/a6
```

| Opción | Qué hace |
|---|---|
| `--n 100` | QR por condición de daño (más = más confiable y más lento; 200 por defecto). |
| `--seed 0` | Semilla; con la misma semilla los resultados se repiten. |
| `--csv RUTA` | Guarda la tabla en CSV. **Sin esta opción no se crea ningún archivo.** |
| `--save-examples CARPETA` | Guarda un par de imágenes (revelado y limpio) por condición. |

**Resultado esperado:** la tabla de la sección *Resultados* (pueden variar 1 o 2 puntos
según la versión de OpenCV). Para dar la comprobación por buena:

- `limpio`, `grises (c=0.6)` y `ruido σ=0.10` → **100 %** en las tres columnas.
- `ruido σ=0.20` en `A6+fallback` → **≥ 90 %**.
- `A6+fallback` es la columna más alta (o empata) en cada fila.

**Archivos generados** (en `outputs/a6/`):

- `bench.csv`: columnas `condicion, config, leidos, total, tasa` (se abre en Excel o Sheets).
- 20 imágenes: un par `*_revelado.png` / `*_limpio.png` por cada condición.

Para revisarlos: `explorer outputs\a6` (Windows), `open outputs/a6` (macOS),
`xdg-open outputs/a6` (Linux).

> `outputs/` y `.venv/` están en `.gitignore`: **no se suben a GitHub**. Cada persona genera
> sus propios resultados. Antes de hacer `git push`, `git status` no debe mostrarlos.

### Resumen: la A6 está comprobada si…

1. `python -m pytest tests/test_qr.py -v` termina sin ningún `FAILED`.
2. La prueba visual imprime `Leido: Hola TEL-328` y el celular lee `3_limpio.png`.
3. El benchmark da 100 % en las condiciones fáciles y ≥ 90 % en `ruido σ=0.20` con `A6+fallback`.

### En Google Colab o Kaggle (sin instalar nada local)

```python
!git clone -b feature/a6-lectura-qr https://github.com/yfloress/DeepStego.git
%cd DeepStego
!pip install -q -e ".[dev]"
!python -m pytest tests/test_qr.py -v
!python scripts/bench_qr.py --n 100 --csv outputs/a6/bench.csv --save-examples outputs/a6
```

Las imágenes quedan en el panel de archivos de la izquierda (`DeepStego/outputs/a6`).



## Pendiente / siguiente

- [ ] Medir con S′ reales del primer modelo entrenado (A9).
- [ ] Si la red produce QR invertidos o con contraste muy bajo, revisar el paso 3.
- [ ] Probar `pyzbar` como lector de respaldo si `cv2.QRCodeDetector` falla en casos que
      se ven bien a simple vista.
- [ ] Coordinar con DA (A4): `make_qr` quedó aquí para poder probar M3; M1 debe usarla
      (o reemplazarla manteniendo la misma convención: blanco = 1, negro = 0, borde 4).

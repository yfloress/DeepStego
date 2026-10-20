# DeepStego
Mensajes ocultos en imágenes con redes neuronales

## Entorno con uv

Con Python 3.10 o superior y [uv](https://docs.astral.sh/uv/) instalado:

```bash
uv sync --extra dev
```

## Generador de QR — A4 / M1

Genera códigos QR de **128×128 píxeles**, en escala de grises, con corrección
de errores **H** y margen blanco de al menos cuatro módulos. Admite mensajes de **1 a
20 caracteres**, incluidos espacios y acentos. El escalado por vecino más
cercano usa un factor entero para conservar módulos uniformes y valores binarios,
sin introducir tonos intermedios. El QR se centra en un lienzo blanco de 128×128.

Para guardar un mensaje como PNG:

```bash
uv run python -m deepstego.qr "Hola DeepStego" --output outputs/qr.png
```

Para usarlo desde Python:

```python
from deepstego.qr import generate_qr, generate_qr_array, iter_qr_samples

image = generate_qr("Hola")           # PIL, modo L, valores 0 y 255
secret = generate_qr_array("Hola")    # NumPy float32, (1, 128, 128), rango [0, 1]

# Conjunto de 1.000 QR reproducibles para prueba; conserva el texto esperado.
test_samples = list(iter_qr_samples(1000, seed=42))

# Generación sobre la marcha para entrenamiento, sin almacenar imágenes.
training_samples = iter_qr_samples()
text, secret = next(training_samples)
```

Los mensajes aleatorios usan letras ASCII y dígitos, con longitudes de 1 a 20.
La semilla es local y no modifica el estado aleatorio del resto del programa.
Para reproducir entrenamiento, pasa también una semilla; si usas varios workers,
asigna una distinta a cada worker. Usa semillas diferentes para entrenamiento,
validación y prueba; esto no garantiza que los mensajes sean disjuntos.
Los arreglos se pueden convertir con `torch.from_numpy(secret)` o agrupar con
el `DataLoader` de PyTorch para obtener lotes `(B, 1, 128, 128)`.

Pruebas de generación y lectura de los PNG con OpenCV (detector clásico y
detector ArUco como alternativa cuando el primero no localiza el QR):

```bash
uv run pytest
uv run ruff check .
```

El generador cubre la parte de QR sintéticos de A4. La carga de DIV2K/COCO,
el entrenamiento y la limpieza/lectura de QR revelados siguen pendientes.

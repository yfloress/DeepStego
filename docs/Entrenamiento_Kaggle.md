# Entrenamiento de M2 en Kaggle

## Preparación

1. Importa `notebooks/train_kaggle.ipynb` en un notebook de Kaggle.
2. Activa un acelerador GPU y el acceso a Internet.
3. Adjunta el dataset `DeepStego-DIV2K`, que contiene entrenamiento y validación.
4. Crea un dataset privado con `outputs/deepstego-kaggle.zip` y adjúntalo también.
   El ZIP contiene el código preparado en esta copia del repositorio.
5. Ejecuta las celdas en orden.

La primera celda instala `uv` si falta y utiliza `uv` para instalar `qrcode[pil]`.
Usa PyTorch y CUDA del entorno de Kaggle; no ejecutes `uv sync` en ese notebook.
El código se copia a `/kaggle/working/deepstego-code` antes de importarlo.

El notebook busca las carpetas `DIV2K_train_HR` y `DIV2K_valid_HR` con imágenes
dentro de `/kaggle/input`, sin depender del nombre del dataset o las carpetas
intermedias y sin distinguir mayúsculas de minúsculas.
También acepta el ZIP de código con mayúsculas o minúsculas en su nombre.

Si aparece «Falta el código», adjunta el dataset creado con
`outputs/deepstego-kaggle.zip`: el dataset de imágenes y el código son dos
entradas distintas. Subir únicamente el notebook y DIV2K no es suficiente.

Si hay varias copias de una partición, indica su ruta en `TRAIN_DIR` o `VAL_DIR`.
Si falta una partición, la ejecución se detiene; no se divide validación para entrenar.
Comprueba los recuentos: el DIV2K completo tiene 800 imágenes de entrenamiento
y 100 de validación. No apuntes ambas variables a la raíz común del dataset.

## Primera ejecución

El notebook empieza con **una época** para verificar el recorrido completo.
Muestra fotografías y QR antes de entrenar, y ejemplos reconstruidos al terminar.

La configuración inicial utiliza:

| Parámetro | Valor |
| --- | --- |
| Tamaño | 128×128 |
| Batch | 16 |
| Recortes por imagen y época | 4 |
| Canales internos | 32 |
| Optimizador | Adam |
| Learning rate | 0,001 |
| β | 1 |
| Workers | 2 |

Son valores iniciales de experimentación, no parámetros ya validados con DIV2K.
Si falta memoria GPU, reduce el batch antes de iniciar una ejecución nueva.
Si hay problemas con los workers, usa `num_workers=0`.
Se utiliza una sola GPU aunque el entorno ofrezca más de una.

Después de revisar la primera época, cambia `EPOCHS` a `20` y `RESUME` a
`"/kaggle/working/m2/last.pt"`. Vuelve a ejecutar la celda de entrenamiento
y las siguientes. `EPOCHS` indica el total deseado, no las épocas adicionales.

## Resultados y reanudación

En `/kaggle/working/m2` se guardan:

- `last.pt`: estado al terminar la última época.
- `best.pt`: estado con menor pérdida total de validación.
- `history.json`: pérdidas de portadora y QR, pérdida total y PSNR por época.
- `curves.png` y `examples.png`: gráficos creados por el notebook.
- `container.png`: ejemplo guardado y recargado como PNG antes de revelar.

Los checkpoints contienen ambas redes, el optimizador, la configuración
y el historial. Guarda una versión del notebook con sus outputs o descarga
estos archivos antes de cerrar la sesión. Solo se recuperan épocas completas.
`Save & Run All` ejecuta el notebook desde el principio en otra sesión; revisa
las rutas y `RESUME` antes de usarlo.

Para continuar en otra sesión:

1. Adjunta los outputs anteriores como entrada del nuevo notebook.
2. Mantén `last.pt` y `best.pt` juntos.
3. Configura `RESUME` con la ruta del `last.pt` adjunto.
4. Conserva los mismos datos, batch, β, learning rate, canales, recortes y semilla.
5. Usa `/kaggle/working/m2` como salida y aumenta `EPOCHS`.

Para experimentar con otros hiperparámetros, usa `RESUME=None` y una carpeta
de salida nueva. El programa evita sobrescribir checkpoints de un entrenamiento
existente cuando no se solicita reanudar.

## Qué implementa esta versión

`HideNet` concatena RGB y QR y aplica cuatro convoluciones; `RevealNet` recibe
la contenedora y reconstruye un canal. Es una CNN inicial simplificada,
no una reproducción de la arquitectura multiescala de Baluja.

La pérdida es `MSE(portadora, contenedora) + β * MSE(QR, QR_revelado)`.
La contenedora se cuantiza a 8 bits antes de calcular la pérdida y revelar;
el estimador STE permite propagar gradientes a través del redondeo.

Entrenamiento usa recortes y volteos variables por época. Validación usa
recortes centrales y mensajes fijos. Cada muestra depende de la semilla,
la época y el índice, para permitir reanudar sin repetir la primera época.
No se garantiza igualdad bit a bit entre distintos dispositivos o versiones.

El PSNR registrado es el promedio por imagen, con un máximo numérico de 120 dB.
La selección de `best.pt` usa la pérdida, no la tasa de lectura de QR.
Faltan la evaluación completa de M4 (SSIM y lectura del mensaje), la prueba
independiente con COCO y la validación de los objetivos del proyecto.
Una pérdida baja no demuestra por sí sola que los QR sean legibles.

## Ejecución local

```bash
uv sync --extra dev
uv run python -m deepstego.train \
  --train-dir /ruta/DIV2K_train_HR \
  --val-dir /ruta/DIV2K_valid_HR \
  --output-dir outputs/m2 \
  --epochs 1
uv run pytest
```

Referencias de entorno: [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)
y [entornos de uv](https://docs.astral.sh/uv/pip/environments/).

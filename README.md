# DeepStego

Mensajes ocultos en imágenes con redes neuronales.

[Propuesta del proyecto](docs/Propuesta_Proyecto_DeepStego.md).

## Iniciar la web

Con [uv](https://docs.astral.sh/uv/) instalado, ejecuta desde la raíz del proyecto:

```bash
uv run --extra app python -m deepstego.app
```

Abre http://127.0.0.1:7860. Usa el modelo `checkpoints/best.pt`.

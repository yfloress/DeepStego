"""Interfaz Gradio para probar el checkpoint de DeepStego."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from deepstego.metrics import image_quality
from deepstego.models import HideNet, RevealNet, quantize_8bit
from deepstego.qr import QR_SIZE, generate_qr, read_qr

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "best.pt"


class ModelDemo:
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Falta el modelo: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.hide = HideNet(checkpoint["config"]["width"])
        self.reveal = RevealNet(checkpoint["config"]["width"])
        self.hide.load_state_dict(checkpoint["hide"])
        self.reveal.load_state_dict(checkpoint["reveal"])
        self.hide.eval()
        self.reveal.eval()
        self.epoch = checkpoint["epoch"]
        self.files = TemporaryDirectory(prefix="deepstego-web-")

    @staticmethod
    def tensor(image):
        pixels = np.array(image, dtype=np.float32) / 255
        if pixels.ndim == 2:
            pixels = pixels[:, :, None]
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)

    def hide_message(self, photo, text):
        if photo is None:
            raise ValueError("Carga una fotografía para ocultar el mensaje.")
        secret = generate_qr(text)
        cover = ImageOps.fit(
            ImageOps.exif_transpose(photo).convert("RGB"),
            (QR_SIZE, QR_SIZE),
            method=Image.Resampling.LANCZOS,
        )
        with torch.inference_mode():
            container = quantize_8bit(self.hide(self.tensor(cover), self.tensor(secret)))
        pixels = container[0].permute(1, 2, 0).mul(255).round().byte().numpy()
        path = Path(self.files.name) / f"deepstego-{uuid4().hex}.png"
        Image.fromarray(pixels).save(path, format="PNG")
        with Image.open(path) as png:
            loaded = png.convert("RGB")
        quality = image_quality(np.array(cover) / 255, np.array(loaded) / 255)
        _, _, decoded = self.reveal_image(path)
        status = (
            "Mensaje recuperado correctamente."
            if decoded == text
            else "El modelo no recuperó el mensaje exacto en esta imagen. Prueba otra fotografía."
        )
        metrics = (
            f"**PSNR:** {quality['psnr']:.2f} dB · **SSIM:** {quality['ssim']:.3f}\n\n"
            f"{status}\n\nMetas del proyecto: PSNR ≥ 30 dB y SSIM ≥ 0,90."
        )
        return cover, secret, str(path), str(path), metrics

    def reveal_image(self, path):
        if not path:
            raise ValueError("Carga la imagen PNG que descargaste al ocultar el mensaje.")
        try:
            with Image.open(path) as png:
                if png.format != "PNG" or png.size != (QR_SIZE, QR_SIZE):
                    raise ValueError("Usa una contenedora PNG de 128×128, sin redimensionarla.")
                image = png.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("No se pudo abrir la imagen PNG.") from exc
        with torch.inference_mode():
            recovered = self.reveal(self.tensor(image))[0, 0]
        result = read_qr(recovered)
        raw = recovered.mul(255).round().byte().numpy()
        return raw, result.clean, result.text


def build_app(model):
    import gradio as gr

    def hide(photo, text):
        try:
            return model.hide_message(photo, text)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def reveal(path):
        try:
            raw, clean, text = model.reveal_image(path)
            return raw, clean, text if text is not None else "QR no legible"
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="DeepStego", analytics_enabled=False) as demo:
        gr.Markdown(
            "# DeepStego\n"
            "### Un mensaje dentro de una fotografía\n"
            "Oculta un texto corto, descarga la imagen y vuelve a cargarla para recuperarlo."
        )
        gr.Markdown(
            f"Modelo de prueba · época {model.epoch} · imágenes de 128×128\n\n"
            "La calidad visual y la lectura dependen de la fotografía. "
            "Este modelo todavía está en desarrollo."
        )
        with gr.Tab("Ocultar"):
            with gr.Row():
                with gr.Column():
                    photo = gr.Image(label="1. Carga tu fotografía", type="pil", sources=["upload"])
                    text = gr.Textbox(
                        label="2. Escribe tu mensaje", placeholder="Hasta 20 caracteres"
                    )
                    gr.Markdown("La fotografía se recorta al centro y se ajusta a 128×128 píxeles.")
                    hide_button = gr.Button("Ocultar mensaje", variant="primary")
                with gr.Column():
                    container = gr.Image(
                        label="Imagen con el mensaje oculto", format="png", interactive=False
                    )
                    download = gr.File(label="Descargar imagen PNG", interactive=False)
                    metrics = gr.Markdown()
            with gr.Accordion("Ver fotografía preparada y QR original", open=False), gr.Row():
                cover = gr.Image(label="Fotografía preparada", format="png", interactive=False)
                secret = gr.Image(label="QR original", format="png", interactive=False)
            hide_button.click(
                hide, [photo, text], [cover, secret, container, download, metrics], api_name="hide"
            )
        with gr.Tab("Revelar"):
            gr.Markdown("Carga el PNG descargado, conservando su tamaño y formato originales.")
            source = gr.File(label="Imagen contenedora", file_types=[".png"], type="filepath")
            reveal_button = gr.Button("Revelar mensaje", variant="primary")
            message = gr.Textbox(label="Mensaje recuperado", interactive=False)
            with gr.Row():
                raw = gr.Image(label="QR revelado por el modelo", format="png", interactive=False)
                clean = gr.Image(label="QR limpio", format="png", interactive=False)
            reveal_button.click(
                reveal, source, [raw, clean, message], api_name="reveal", concurrency_id="model"
            )
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    model = ModelDemo(args.checkpoint)
    demo = build_app(model)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        allowed_paths=[model.files.name],
    )


if __name__ == "__main__":
    main()

"""Descarga COCO val2017 y extrae 1000 imágenes para evaluar DeepStego."""

import argparse
import random
import re
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image

COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGE_COUNT = 1000
MIN_SIDE = 128  #Tamaño del recorte en DIV2KDataset.


def download_archive(archive: Path) -> None:
    if archive.is_file():
        return

    print("Descargando COCO val2017 (816 MB)...", flush=True)
    with TemporaryDirectory(dir=archive.parent) as staging:
        partial = Path(staging) / archive.name
        with urlopen(COCO_URL, timeout=60) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        with ZipFile(partial):
            pass
        partial.replace(archive)


def select_images(archive: ZipFile, seed: int) -> dict[str, ZipInfo]:
    candidates = sorted(
        (
            entry
            for entry in archive.infolist()
            if re.fullmatch(r"val2017/[0-9]{12}\.jpg", entry.filename)
        ),
        key=lambda entry: entry.filename,
    )
    random.Random(seed).shuffle(candidates)

    selected = {}
    for entry in candidates:
        with archive.open(entry) as source, Image.open(source) as image:
            if min(image.size) < MIN_SIDE:
                continue
        selected[PurePosixPath(entry.filename).name] = entry
        if len(selected) == IMAGE_COUNT:
            return selected
    raise ValueError(f"Se encontraron {len(selected)} imágenes aptas; se necesitan {IMAGE_COUNT}.")


def prepare_coco(data_dir: Path, seed: int) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "val2017.zip"
    output = data_dir / "coco_val2017_1000"
    download_archive(archive_path)

    with ZipFile(archive_path) as archive:
        images = select_images(archive, seed)
        if output.exists():
            if (
                output.is_dir()
                and {path.name for path in output.iterdir()} == images.keys()
                and all(
                    (output / name).is_file() and (output / name).stat().st_size == entry.file_size
                    for name, entry in images.items()
                )
            ):
                return output
            raise ValueError(f"{output} ya existe con otro contenido. Usa otro --data-dir.")

        print(f"Extrayendo {IMAGE_COUNT} imágenes (semilla {seed})...", flush=True)
        with TemporaryDirectory(dir=data_dir) as staging:
            subset = Path(staging) / output.name
            subset.mkdir()
            for name, entry in images.items():
                with archive.open(entry) as source, (subset / name).open("wb") as target:
                    shutil.copyfileobj(source, target)
            subset.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=202611)
    args = parser.parse_args()
    try:
        output = prepare_coco(args.data_dir, args.seed)
    except (OSError, BadZipFile, ValueError) as error:
        parser.exit(1, f"No se pudo preparar COCO val2017. {error}\n")
    print(f"{IMAGE_COUNT} imágenes listas en {output.resolve()}")


if __name__ == "__main__":
    main()

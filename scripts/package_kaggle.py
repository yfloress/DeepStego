from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs" / "deepstego-kaggle.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        root / "pyproject.toml",
        root / "README.md",
        root / "LICENSE",
        root / "notebooks" / "train_kaggle.ipynb",
        *sorted((root / "src").rglob("*.py")),
    ]
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("deepstego") / path.relative_to(root))
    print(output)


if __name__ == "__main__":
    main()

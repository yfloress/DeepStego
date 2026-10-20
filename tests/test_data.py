import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from deepstego.data import DIV2KDataset, find_div2k_split


@pytest.fixture
def image_dir(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(42)
    for index in range(2):
        Image.fromarray(rng.integers(0, 256, (160, 180, 3), dtype=np.uint8)).save(
            root / f"{index}.png"
        )
    return root


def test_validation_is_fixed_and_normalized(image_dir):
    dataset = DIV2KDataset(image_dir, training=False)
    before = dataset[0]
    dataset.set_epoch(10)
    after = dataset[0]
    assert before["text"] == after["text"]
    for key, shape in (("cover", (3, 128, 128)), ("secret", (1, 128, 128))):
        assert before[key].shape == shape
        assert before[key].dtype == torch.float32
        assert before[key].min() >= 0 and before[key].max() <= 1
        torch.testing.assert_close(before[key], after[key])
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["cover"].shape == (2, 3, 128, 128)


def test_training_changes_by_epoch_reproducibly(image_dir):
    first = DIV2KDataset(image_dir, training=True, crops_per_image=2)
    second = DIV2KDataset(image_dir, training=True, crops_per_image=2)
    assert len(first) == 4
    torch.testing.assert_close(first[0]["cover"], second[0]["cover"])
    old = first[0]
    first.set_epoch(1)
    second.set_epoch(1)
    assert not torch.equal(old["cover"], first[0]["cover"])
    assert old["text"] != first[0]["text"]
    assert first[0]["text"] == second[0]["text"]
    torch.testing.assert_close(first[0]["cover"], second[0]["cover"])


def test_missing_images_and_too_small_images(tmp_path):
    with pytest.raises(ValueError, match="No se encontraron"):
        DIV2KDataset(tmp_path, training=True)
    Image.new("RGB", (64, 64)).save(tmp_path / "small.png")
    dataset = DIV2KDataset(tmp_path, training=True)
    with pytest.raises(ValueError, match="menor a 128"):
        dataset[0]


@pytest.mark.parametrize("folder", ["DIV2K_train_HR", "div2k_TRAIN_hr"])
def test_find_split_ignores_mount_name_and_empty_wrapper(tmp_path, folder):
    leaf = tmp_path / "datasets" / "owner" / "another-name" / folder / folder
    leaf.mkdir(parents=True)
    Image.new("RGB", (128, 128)).save(leaf / "0001.PNG")
    assert find_div2k_split(tmp_path, "DIV2K_train_HR") == leaf


def test_find_split_requires_one_partition(tmp_path):
    with pytest.raises(FileNotFoundError, match="DIV2K_valid_HR"):
        find_div2k_split(tmp_path, "DIV2K_valid_HR")
    for name in ("first", "second"):
        folder = tmp_path / name / "DIV2K_valid_HR"
        folder.mkdir(parents=True)
        Image.new("RGB", (128, 128)).save(folder / "0001.png")
    with pytest.raises(ValueError, match="varias carpetas"):
        find_div2k_split(tmp_path, "DIV2K_valid_HR")

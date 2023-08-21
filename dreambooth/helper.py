import glob

from PIL import Image
from huggingface_hub import snapshot_download


def download_data():
    local_dir = "./dog"
    snapshot_download(
        "diffusers/dog-example",
        local_dir=local_dir,
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )


def image_grid(imgs, rows, cols, resize=256):
    assert len(imgs) == rows * cols

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def show_train_data():
    imgs = [Image.open(path) for path in glob.glob("./dog/*.jpeg")]
    image_grid(imgs, 1, 5)



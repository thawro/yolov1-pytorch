from tqdm import tqdm
import urllib.request
import tarfile
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .pylogger import get_pylogger

log = get_pylogger(__name__)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download_file(url, filepath):
    log.info(f"Downloading {url} to {filepath}.")
    with TqdmUpTo(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split("/")[-1]
    ) as t:  # all optional kwargs
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n
    log.info("Download finished.")


def unzip_tar_gz(file_path, dst_path, remove=False):
    log.info(f"Unzipping {file_path} to {dst_path}.")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(dst_path)
    log.info(f"Unzipping finished.")
    if remove:
        os.remove(file_path)
        log.info(f"Removed {file_path}.")


def save_txt_to_file(txt, filename):
    with open(filename, "w") as file:
        file.write(txt)


def read_text_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]  # Optional: Remove leading/trailing whitespace

    return lines


def add_prefix_to_files(directory, prefix, ext=".png"):
    log.info(f"Adding {prefix} prefix to all {ext} files in {directory} directory.")

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and filename.endswith(ext):
            new_filename = prefix + filename
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    log.info("Prefix addition finished.")


def move_directory(source_dir, destination_dir):
    shutil.move(source_dir, destination_dir)
    log.info(f"Moved {source_dir} to {destination_dir}.")


def copy_directory(source_dir, destination_dir):
    shutil.copytree(source_dir, destination_dir)
    log.info(f"Copied {source_dir} to {destination_dir}.")


def remove_directory(dir_path):
    shutil.rmtree(dir_path)
    log.info(f"Removed {dir_path} directory")


def copy_all_files(source_dir, destination_dir, ext=".png"):
    filenames = os.listdir(source_dir)
    for filename in tqdm(filenames, desc="Copying files"):
        if filename.lower().endswith(ext):
            source = source_dir / filename
            destination = destination_dir / filename
            shutil.copy2(source, destination)

    log.info(f"Copied all {ext} files ({len(filenames)}) from {source_dir} to {destination_dir}.")


def copy_files(source_filepaths, dest_filepaths):
    for source_filepath, dest_filepath in tqdm(
        zip(source_filepaths, dest_filepaths), desc="Copying files"
    ):
        shutil.copy2(source_filepath, dest_filepath)
    log.info(f"Copied files ({len(source_filepaths)}).")




def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)
    # box coords: x_c, y_c, w, h

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    plt.show()


def train_loop(model, dataloader, optimizer, loss_fn, device="cpu"):
    loop = tqdm(dataloader, leave=True, desc="Train")
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    return sum(losses) / len(losses)

import argparse
from pathlib import Path
import shutil
from time import time
import lmdb
import pickle


def write_database(d: dict, database: Path):
    # Remove any existing database.
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        shutil.rmtree(database)

    # For condor usage, we create a local database on the disk.
    tmp_dir = Path("/tmp") / f"TEMP_{time()}"
    tmp_dir.mkdir(parents=True)

    tmp_database = tmp_dir / f"{database.name}"

    # Create the database.
    with lmdb.open(path=f"{tmp_database}", map_size=2**40) as env:
        # Add the protocol to the database.
        with env.begin(write=True) as txn:
            key = "protocol".encode("ascii")
            value = pickle.dumps(pickle.DEFAULT_PROTOCOL)
            txn.put(key=key, value=value, dupdata=False)
        # Add the keys to the database.
        with env.begin(write=True) as txn:
            key = pickle.dumps("keys")
            value = pickle.dumps(sorted(d.keys()))
            txn.put(key=key, value=value, dupdata=False)
        # Add the images to the database.
        for key, value in sorted(d.items()):
            with env.begin(write=True) as txn:
                with value.open("rb") as file:
                    key = pickle.dumps(key)
                    txn.put(key=key, value=file.read(), dupdata=False)

    # Move the database to its destination.
    shutil.move(f"{tmp_database}", database)

    # Remove the temporary directories.
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_images", type=Path, required=True)
    parser.add_argument("--extension", type=str, required=True)
    parser.add_argument("--dst_database", type=Path, required=True)
    args = parser.parse_args()

    src_images = args.src_images
    extension = args.extension
    dst_database = args.dst_database

    # Customize the how images are to be found and organise them as a dictionary.
    # Here it's just a recursive glob over the whole source image directory.
    image_paths = {image_path.stem: image_path for image_path in sorted(src_images.rglob(f"*{extension}"))}
    write_database(image_paths, dst_database)

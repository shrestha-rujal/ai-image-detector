import os
import shutil
import random

SOURCE_REAL = "dataset/real"
SOURCE_FAKE = "dataset/fake"

DEST = "images"

TRAIN_RATIO = 0.7
TEST_RATIO = 0.15
VAL_RATIO = 0.15


def create_folders():
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            folder = os.path.join(DEST, split, label)
            os.makedirs(folder, exist_ok=True)


def split_images(source_folder, label):
    images = [f for f in os.listdir(source_folder)
              if f.endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split, split_images in [('train', train_images),
                                ('val', val_images),
                                ('test', test_images)]:
        for img in split_images:
            src = os.path.join(source_folder, img)
            dst = os.path.join(DEST, split, label, img)
            shutil.copy(src, dst)

    print(f"{label}: {len(train_images)} train,{len(val_images)} val,{len(test_images)} test")


if __name__ == "__main__":
    create_folders()
    split_images(SOURCE_REAL, 'real')
    split_images(SOURCE_FAKE, 'fake')
    print('Dataset ready')

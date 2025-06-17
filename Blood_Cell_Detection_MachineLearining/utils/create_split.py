import os
import random

def create_train_test_split(data_dir, train_ratio=0.8, seed=42):
    img_dir = os.path.join(data_dir, 'JPEGImages')
    images = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.seed(seed)
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    train_imgs = images[:train_size]
    test_imgs = images[train_size:]

    splits_dir = os.path.join(data_dir, 'ImageSets')
    os.makedirs(splits_dir, exist_ok=True)

    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for img in train_imgs:
            f.write(img + '\n')

    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for img in test_imgs:
            f.write(img + '\n')

    print(f"[INFO] Created train ({len(train_imgs)}) and test ({len(test_imgs)}) splits.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data', help='BCCD dataset root folder')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    create_train_test_split(args.data_dir, args.train_ratio)

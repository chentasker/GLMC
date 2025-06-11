import os
import tarfile
from collections import defaultdict

def parse_split_file(split_file):
    # Maps wnid -> list of (filename, label)
    image_map = defaultdict(tuple)
    with open(split_file, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            wnid = path.split('/')[-2]
            filename = os.path.basename(path)
            imid = filename.split('_')[1].split('.')[0]
            image_map[filename] = (wnid, imid, int(label))
    return image_map


def extract_images_from_nested_tar(train_tar_path, split_file, dst_root):
    image_map = parse_split_file(split_file)
    image_map_classes = [image_map[k][0] for k in image_map.keys()]
    with tarfile.open(train_tar_path, 'r') as outer_tar:
        for member in outer_tar.getmembers():
            if not member.name.endswith('.tar'):
                continue
            wnid = os.path.splitext(os.path.basename(member.name))[0]
            if wnid not in image_map_classes:
                continue  # Skip classes not in LT split
                

            # Extract the class tar file into memory
            class_tar_bytes = outer_tar.extractfile(member).read()
            class_tar = tarfile.open(fileobj=open_tar_stream(class_tar_bytes), mode='r')

            for img_member in class_tar.getmembers():
                if os.path.basename(img_member.name) in image_map.keys():
                    wnid, imid, label = image_map[os.path.basename(img_member.name)]
                    dst_dir = os.path.join(dst_root,
                                            wnid)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, os.path.basename(img_member.name))
                    with class_tar.extractfile(img_member) as src, open(dst_path, 'wb') as out:
                        out.write(src.read())

            class_tar.close()
            
def extract_test_images(tar_path, split_file, dst_root):
    image_map = defaultdict(tuple)
    with open(split_file, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            wnid = path.split('/')[-2]
            filename = os.path.basename(path)
            imid = filename.split('_')[1].split('.')[0]
            image_map[filename] = (wnid, imid, int(label))
    
    with tarfile.open(tar_path, 'r') as class_tar:
            for img_member in class_tar.getmembers():
                if os.path.basename(img_member.name) in image_map.keys():
                    wnid, imid, label = image_map[os.path.basename(img_member.name)]
                    dst_dir = os.path.join(dst_root,
                                            wnid)
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, os.path.basename(img_member.name))
                    
                    with class_tar.extractfile(img_member) as src, open(dst_path, 'wb') as out:
                        out.write(src.read())

            class_tar.close()
                    

# Helper to read tar from bytes
from io import BytesIO
def open_tar_stream(byte_data):
    return BytesIO(byte_data)

# === Config ===
train_tar = "data/ILSVRC2012_img_val.tar"
split_file = "data/data_txt/ImageNet_LT_test.txt"
output_root = "data/imagenet-lt/val"

extract_test_images(train_tar, split_file, output_root)

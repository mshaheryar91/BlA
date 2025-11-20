from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            # attribute=,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

        attr_file_path = './datasets/identity_CelebA.txt'
        self.labels = file_to_list(attr_file_path)
        
        # Create a dictionary mapping filenames to identity IDs
        self.label_dict = {}
        for line in self.labels:
            parts = line.split(" ", 1)  # Split on the first space
            filename = parts[0].strip()  # Extract the filename
            identity_id = int(parts[1].strip())  # Extract the identity ID
            self.label_dict[filename] = identity_id  # Store in dictionary


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            key_label = f"{str(index).zfill(5)}".encode("utf-8")
            print("check key:", key, key_label)
            img_bytes = txn.get(key)
            
            # Generate the filename for the current index (e.g., "000001.jpg")
            filename = f"{str(index).zfill(6)}.jpg"
            label = self.label_dict.get(filename, -1)  # Get identity label, default to -1 if not found

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        print("check filename and label:", filename, label)

        return img, label


################################################################################

def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    train_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_train'),
                                           train_transform, config.data.image_size)
    test_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_test'),
                                          test_transform, config.data.image_size)


    return train_dataset, test_dataset



def file_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]
    return files


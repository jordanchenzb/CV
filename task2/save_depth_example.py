import numpy as np
import os
import argparse

def save_np(data, save_path):
    '''
    Input:
    ----------
    data: numpy.array
    save_path: str
        example: path/to/depth/image_name.npy
    '''
    ext = save_path.split('.')[-1].lower()
    def save_func(data, path):
        if ext == 'npy':
            save = np.save
        elif ext == 'npz':
            save = np.savez
        else:
            raise NotImplementedError(ext)
        with open(path, 'wb') as h:
            save(h, data)
    save_func(data, save_path)

def load_np(np_f):
    assert os.path.exists(np_f), f"{np_f} not exists!"
    if np_f.endswith('.npy'):
        with open(np_f, 'rb') as h:
            data = np.load(h)
        return data

    # .npz
    with open(np_f, 'rb') as h:
        data = np.load(h, allow_pickle=True)
        data = dict(data)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--depth_map",
                        help="path to depth map", type=str, required=True)
    
    args = parser.parse_args()
    return args


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


if __name__ == "__main__":
    args = parse_args()
    depth = read_array(args.depth_map) # Substitute to your depth map! save as np.float32
    depth_save_path = "./"+args.depth_map+".npz"

    # Save depth example
    save_np(depth, depth_save_path)

    # Load depth example
    depth_loaded = load_np(depth_save_path)
import os
import numpy as np
from tqdm import tqdm

def update_npz_keys(root_path):
    for dirpath, _, filenames in tqdm(os.walk(root_path)):
        for filename in filenames:
            if filename == 'density_map.npz':
                file_path = os.path.join(dirpath, filename)
                # print(f"Processing: {file_path}")
                
                # Load the data
                data = np.load(file_path)
                if 'arr_0' not in data:
                    print(f"Warning: 'arr_0' not found in {file_path}")
                    continue

                # Extract the array
                arr = data['arr_0']
                data.close()

                # Save with new key
                np.savez_compressed(file_path, density_map=arr)

# Example usage
update_npz_keys('/mnt/Text2Video/fanweichen/tx/dataset/mflow/4DGen-Dataset-tx/pexelx_density_maps_v2')

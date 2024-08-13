import h5py

def inspect_hdf5(file_path):
    with h5py.File(file_path, 'r') as hf:
        print("Keys in the file:")
        for key in hf.keys():
            print(key)
            # Print the dataset names in each key (if any)
            if isinstance(hf[key], h5py.Group):
                print(f"  Datasets in group '{key}':")
                for sub_key in hf[key].keys():
                    print(f"    {sub_key}")

inspect_hdf5('ecmschatbot_model.h5')

import h5py
import pandas as pd

def hdf5_to_pandas(file_path) -> pd.DataFrame:
    """
    Read datasets from an HDF5 file into a Pandas DataFrame.

    :param file_path: Path to the HDF5 file.
    :param keys: List of keys to read from the HDF5 file.
    :return: Pandas DataFrame.
    """
    with h5py.File(file_path, "r") as hdf_file:
        keys = list(hdf_file.keys())

    print(keys)
    keys = ['time_stamp', 'set_temperature', 'measured_temperature', 'heat_flux', 'voltage', 'current',
            'calculated_voltage']
    # Read datasets into a Pandas DataFrame
    with h5py.File(file_path, "r") as hdf_file:
        data = {key: hdf_file[key][:] for key in keys}

    return pd.DataFrame(data)



if __name__ == "__main__":
    # Path to the HDF5 file
    file_path = "../data/82_TDS_test/data.h5"


    # Read datasets from the HDF5 file into a Pandas DataFrame
    df = hdf5_to_pandas(file_path)

    print(df[:5].to_string())
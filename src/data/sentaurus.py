from __future__ import annotations
from pathlib import Path
import numpy as np

def load_folder(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, metadata) from a directory of Sentaurus device folders.

    Expected layout:
        root/
            device_0001/
                IdVg_Vds=0.1.plt
                IdVg_Vds=1.0.plt
                IdVd_Vgs=0.5.plt
                IdVd_Vgs=1.0.plt
                variables.csv
            device_0002/
                ...
    Returns
    """
    raise NotImplementedError(
        "Wire this up to the data path before reporting any real numbers. "
    )

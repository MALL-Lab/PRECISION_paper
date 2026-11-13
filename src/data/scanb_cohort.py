"""SCAN-B survival cohort with the patient rule of scripts/21_multicohort_survival.py."""

from pathlib import Path

import pandas as pd

from config import DATA_DIR


def scanb_patient_cohort(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """SCAN-B clinical table indexed by sample id, one sample per patient.

    The first sample of each patient in sample order is kept, then overall
    survival must be present and positive. Joined with the per-drug
    predictions this gives the survival cohort of Methods (7,397 patients),
    before the PAM50 filter of the Cox models (7,215).
    """
    clin = pd.read_csv(Path(data_dir) / "precision_processed" / "scanb_clinical.csv")
    clin = clin.set_index("sample_id").sort_index()
    clin = clin[~clin["patient_id"].duplicated(keep="first")]
    clin["os_time"] = pd.to_numeric(clin["os_time"], errors="coerce")
    clin["os_event"] = pd.to_numeric(clin["os_event"], errors="coerce")
    clin = clin.dropna(subset=["os_time", "os_event"])
    return clin[clin["os_time"] > 0]

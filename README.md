# fNIRS Spatial Integration & Global Signal Removal Pipeline

An end-to-end **functional near-infrared spectroscopy (fNIRS)** processing pipeline implementing:

- Subject → **fsaverage** landmark alignment
- Spherical projection of channel locations
- **Geodesic Gaussian spatial filtering**
- **Zhang / Klein spatial principal component (PC) global signal removal**
- MNE-compatible `Raw` and `Epochs` construction
- Optional wavelet detrending and band-pass filtering
- Robust task → events / annotations handling

This pipeline is designed for **research-grade fNIRS preprocessing** with a strong emphasis on **spatially informed global physiology removal**.

---

## 📦 Features

### Geometry & Alignment
- Umeyama least-squares similarity transform (rotation, translation, optional scaling)
- Alignment from subject fiducials (NAS/LPA/RPA ± Inion/Cz) to **fsaverage MRI space**
- Conversion between **MRI**, **HEAD**, and **spherical** coordinate systems
- Robust unit handling (mm ↔ m)

### Spatial Filtering
- Projection of channels to a unit sphere
- Pairwise **great-circle (geodesic) distances**
- Gaussian spatial kernel on the head surface
- Spatial PC filtering following:
  - Zhang et al.
  - Klein et al.
- Separation of:
  - `H_global` (systemic / global component)
  - `H_neur` (neural component)

### MNE Integration
- Native `mne.io.RawArray` creation
- HbO / HbR channel typing
- Optode-based montages (`S#`, `D#`)
- Automatic validation of montage vs channel definitions
- Seamless epoching via MNE

### Task & Events
- Convert behavioral/task DataFrame → MNE events
- Optional rest events
- Continuous block annotations
- Direct `Epochs` creation helper

---
Subject Optodes & Channels (mm)
↓
Landmark Alignment (Umeyama)
↓
fsaverage MRI Coordinates
↓
Spherical Projection
↓
Geodesic Gaussian Kernel
↓
Spatial PC Filtering
↓
HbO / HbR Cleaned Signals
↓
MNE Raw → Epochs


---

## 📁 File Structure


.
├── spatial_Integration.py # Full pipeline implementation
├── README.md # This file


---

## 🔧 Requirements

### Python
- Python ≥ 3.9

### Core Dependencies
```bash
pip install numpy scipy pandas matplotlib mne
pip install pywavelets
```
🚀 Quick Start
1. Import the Pipeline
```bash
from spatial_Integration import run_pipeline
```
2. Required Inputs
Fiducials (mm)
```bash
fid_subj_mm = {
    "N": np.array([x, y, z]),
    "L": np.array([x, y, z]),
    "R": np.array([x, y, z]),
    # Optional:
    "I": np.array([x, y, z]),
    "CZ": np.array([x, y, z])
}
```
Channel Coordinates (mm)
```bash
channel_coords = np.ndarray((n_channels, 3))
```
Optode Table
optode_df columns:
["Name", "X", "Y", "Z"]   # Name = S# or D#

fNIRS Data

```bash
H_hbo = np.ndarray((n_times, n_channels))
H_hbr = np.ndarray((n_times, n_channels))
sfreq = 10.0  # Hz
```

3. Run the Pipeline

```bash
result = run_pipeline(
    channel_coords=channel_coords,
    fid_subj_mm=fid_subj_mm,
    ch_xyz_mm=channel_coords,
    ch_names_hbo=hbo_names,
    ch_names_hbr=hbr_names,
    optode_df=optode_df,
    H_hbo=H_hbo,
    H_hbr=H_hbr,
    sfreq=sfreq,
    sigma_deg=48.0,
    do_wavelet_detrend=False
)

```

📤 Outputs

The pipeline returns a dictionary:

{
  "raw"           : mne.io.Raw,
  "xf"            : alignment diagnostics,
  "H_hbo_neur"    : neural HbO component,
  "H_hbo_global"  : global HbO component,
  "H_hbr_neur"    : neural HbR component,
  "H_hbr_global"  : global HbR component,
  "K"             : spatial kernel,
  "epochs"        : mne.Epochs (if events provided)
}


📊 Visualization Helpers

Included plotting utilities:

plot_sample_comparison

plot_correlation_matrices_and_kernel

Example:
```bash
plot_sample_comparison(
    raw=H_hbo,
    H_clean=result["H_hbo_neur"],
    H_global=result["H_hbo_global"],
    channel_index=1
)
```


🧪 Event & Epoch Utilities
Create Events from Task Table

events, event_id = events_from_task_df(df, raw)

Annotate Continuous Blocks
annotate_task_blocks(df, raw)

Direct Epoch Creation
epochs, events, event_id = epochs_from_task_df(df, raw)



⚠️ Notes & Best Practices

Units matter: Inputs are assumed in millimeters, MNE uses meters

Ensure all S#/D# optodes referenced in channel names exist in the montage

Bad channels marked in MNE are automatically excluded from spatial filtering

Spatial PC filtering is applied separately to HbO and HbR



📚 References

Zhang et al. (Hirsch Lab, Yale), NeuroImage — spatial global physiology removal

Klein et al., Human Brain Mapping

MNE-Python fNIRS module documentation



🧑‍🔬 Intended Use

This codebase is designed for:

Cognitive & systems neuroscience

High-density fNIRS studies

Research pipelines requiring spatially principled denoising

It is not a black-box clinical preprocessing tool.


✨ Acknowledgments

Built on top of:

NumPy / SciPy

MNE-Python



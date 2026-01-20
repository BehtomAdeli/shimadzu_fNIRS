# -*- coding: utf-8 -*-
"""
End-to-end fNIRS pipeline:
INLR(+Cz) landmark alignment -> fsaverage MRI coords -> spherical projection ->
Zhang/Klein spatial PC filter -> MNE Raw with montage -> filtering/epoching
"""

import math
import matplotlib
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
import pandas as pd
import mne
from numpy.linalg import svd
from scipy.signal import butter, filtfilt
from typing import Dict, Tuple, Optional, List
from mne.io.constants import FIFF
from matplotlib import pyplot as plt

# =========================
# 0) UTILS: units & checks
# =========================

def ensure_meters(X: np.ndarray) -> np.ndarray:
    """
    Convert coordinates to meters if they look like millimeters (> 10 cm magnitudes).
    MNE expects meters.
    """
    X = np.asarray(X, float)
    scale = 0.001 if np.nanmedian(np.linalg.norm(X, axis=1)) > 0.5 else 1.0
    return X * scale

def ensure_millimeters(X: np.ndarray) -> np.ndarray:
    """
    Convert coordinates to millimeters if they look like meters.
    """
    X = np.asarray(X, float)
    scale = 1000.0 if np.nanmedian(np.linalg.norm(X, axis=1)) < 0.5 else 1.0
    return X * scale

# ===================================
# 1) Landmark alignment (Umeyama LSQ)
# ===================================

def umeyama(src: np.ndarray, dst: np.ndarray, with_scaling: bool = True):
    """
    Closed-form similarity transform y = s R x + t
    src, dst: (N,3) with N>=3
    Returns (s, R, t) with R ∈ SO(3) (reflection handled).
    """
    assert src.shape == dst.shape and src.shape[1] == 3 and src.shape[0] >= 3
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    # Cross-covariance
    Sigma = (Y.T @ X) / n

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)
    # Handle reflection
    S_sign = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_sign[-1, -1] = -1.0

    R = U @ S_sign @ Vt

    if with_scaling:
        var_src = (X**2).sum() / n  # scalar
        # scale = trace(diag(D) @ S_sign) / var_src  == sum(D * diag(S_sign)) / var_src
        scale_num = float((D * np.diag(S_sign)).sum())
        s = scale_num / max(var_src, 1e-15)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return float(s), R, t

def apply_similarity(X: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return (s * (X @ R.T)) + t

def align_to_fsaverage(
    fid_subj_mm: dict,
    ch_xyz_mm: np.ndarray,
    use_extra_landmarks: bool = False,          # set True only if you provide template coords for them
    template_extra_mri: dict | None = None,     # e.g., {"CZ": np.array([...])} in meters
    allow_scaling: bool = True,
    subjects_dir: str | None = None
):
    subjects_dir = mne.datasets.fetch_fsaverage(verbose=False) if subjects_dir is None else subjects_dir
    fids = mne.coreg.get_mni_fiducials('fsaverage', subjects_dir=subjects_dir)

    from mne.io.constants import FIFF
    def _pick_fid_by_ident(fids, ident):
        for f in fids:
            if f.get('ident', None) == ident:
                return np.array(f['r'], float)  # meters
        raise RuntimeError(f"Fiducial ident={ident} not found; got idents: {[f.get('ident') for f in fids]}")

    fids_fsavg_m = {
        "N": _pick_fid_by_ident(fids, FIFF.FIFFV_POINT_NASION),
        "L": _pick_fid_by_ident(fids, FIFF.FIFFV_POINT_LPA),
        "R": _pick_fid_by_ident(fids, FIFF.FIFFV_POINT_RPA),
    }

    keys = ["N", "L", "R"]
    if use_extra_landmarks and template_extra_mri:
        for k in ("I", "CZ"):
            if (k in fid_subj_mm) and (k in template_extra_mri):
                fids_fsavg_m[k] = np.asarray(template_extra_mri[k], float)  # meters
                keys.append(k)

    # Stack matched landmarks (convert mm->m)
    src_m = (np.stack([fid_subj_mm[k] for k in keys], axis=0) / 1000.0).astype(float)
    dst_m =  np.stack([fids_fsavg_m[k]  for k in keys], axis=0).astype(float)

    # Similarity transform
    s, R, t = umeyama(src_m, dst_m, with_scaling=allow_scaling)

    # Apply to channels
    ch_m = (np.asarray(ch_xyz_mm, float) / 1000.0)
    ch_fsavg_mri_m = (s * (ch_m @ R.T)) + t

    # RMS landmark fit (meters)
    fit_src = (s * (src_m @ R.T)) + t
    rms = float(np.sqrt(np.mean(np.sum((fit_src - dst_m)**2, axis=1))))

    xf = {"scale": s, "R": R, "t": t, "fid_keys_used": keys, "rms_m": rms,
          "subjects_dir": subjects_dir, "subject": "fsaverage"}
    return ch_fsavg_mri_m, xf
# ======================================
# 2) Sphere projection & geodesic kernel
# ======================================
def cartesian_to_spherical(arr: np.ndarray) -> np.ndarray:
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        z (float): The z-coordinate.

    Returns:
        tuple: A tuple containing (r, theta, phi) in radians.
    """
    X = np.zeros_like(arr, float)
    for i, (x, y, z) in enumerate(arr):
        r = math.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            theta = 0.0  # Undefined, but often set to 0 at the origin
        else:
            theta = math.asin(z / r)
        phi = math.atan2(y, x)
        X[i,:] = np.array([r, theta, phi])
    return X

def project_to_unit_sphere(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def great_circle_deg(X: np.ndarray) -> np.ndarray:
    """
    Pairwise great-circle angles (degrees) between 3D points after unit-sphere projection.
    """
    # U = project_to_unit_sphere(X)
    # r = np.mean(X[:,0])  # mean radius in meters
    # cosang = np.clip(U @ U.T, -1.0, 1.0)
    # delta_c = np.arccos(cosang)  # radians
    # d = delta_c*r
    # Extract spherical coordinates
    # r = X[:, 0]         # radius (not needed for central angle)
    theta = X[:, 1]     # longitude (azimuthal angle)
    phi = X[:, 2]       # colatitude or latitude (depends on convention)
 
    # # Broadcast to get pairwise differences
    theta1 = theta[:, None]   # shape (N,1)
    theta2 = theta[None, :]   # shape (1,N)
    phi1   = phi[:, None]
    phi2   = phi[None, :]

    delta_c = np.arccos(
                        np.sin(phi1) * np.sin(phi2) +
                        np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2)
                    )
    d= X[:,0]*delta_c #* (180.0 / np.pi)  # convert to degrees
    d = np.nan_to_num(d, nan=0.0)
    tolerance = 1e-3
    d[d < tolerance] = 0.0
    return d

def gaussian_kernel_geodesic(X: np.ndarray, sigma_deg: float = 48.0, row_normalize: bool = True) -> np.ndarray:
    D = great_circle_deg(X)
    #sigma_rad = sigma_deg * (np.pi / 180.0)
    K = np.exp(-(D**2) / (2.0 * (sigma_deg**2)))
    if row_normalize:
        s = K.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        K = K / s
    return K

# ======================================
# 3) Zhang spatial PC filter (SVD)
# ======================================

def spatial_pc_filter(H_raw: np.ndarray, K: np.ndarray, row_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    H_raw: (n_times, n_channels) for one chromophore (HbO or HbR)
    K    : (n_channels, n_channels) geodesic Gaussian on the head sphere
    Returns: H_neur, H_global
    """
    U, S, Vt = svd(H_raw, full_matrices=False)
    V = Vt.T                 # (n_ch, n_pc)

    V_s = np.zeros_like(V)
    V_s = K @ V
    
    if row_normalize:
        s = K.sum(axis=1, keepdims=True)
        s[s <= 0.1] = 1.0
        V_s = V_s / s

    # smooth ALL PCs' spatial patterns
    H_global = (U * S) @ (V_s.T)

    H_neur   = H_raw - H_global

    return H_neur, H_global

def spatial_direct_filter(H_raw: np.ndarray, K: np.ndarray, row_normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    H_raw: (n_times, n_channels) for one chromophore (HbO or HbR)
    K     : (n_channels, n_channels) geodesic Gaussian on the head sphere
    Returns: H_neur, H_global
    """

    V_s = H_raw @ K

    H_global = V_s
    H_neur   = H_raw - H_global
    return H_neur, H_global

# ======================================
# 4) MNE: montage & Raw construction
# ======================================

# ---- helpers to make fid picking robust across MNE versions ----
def _normalize_fids(fids):
    """Return a list of dicts with keys 'r', 'ident', 'kind' from MNE fiducials."""
    out = []
    for f in fids:
        if hasattr(f, 'r'):  # object-like
            out.append({
                'r':    np.array(getattr(f, 'r'), dtype=float),
                'ident':getattr(f, 'ident', None),
                'kind': getattr(f, 'kind', None),
            })
        else:  # dict-like
            out.append({
                'r':    np.array(f.get('r'), dtype=float),
                'ident':f.get('ident', None),
                'kind': f.get('kind', None),
            })
    return out

def _pick_fid_by_ident(fids_norm, ident):
    for f in fids_norm:
        if f['ident'] == ident:
            return f['r']  # meters
    raise RuntimeError(f"Fiducial ident={ident} not found. Idents present: {[f['ident'] for f in fids_norm]}")

# ---- montage builder (fixed) ----
def make_mne_montage_fsaverage(ch_names, ch_fsavg_mri_m, subjects_dir, subject='fsaverage'):
    """
    Build a DigMontage in MRI space using fsaverage fiducials (meters).
    Falls back to building a montage without NAS/LPA/RPA if unavailable.
    """
    ch_pos = {name: ch_fsavg_mri_m[i] for i, name in enumerate(ch_names)}

    # Try to fetch fsaverage fiducials and pick by 'ident'
    nasion = lpa = rpa = None
    try:
        fids = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
        fids_norm = _normalize_fids(fids)
        nasion = _pick_fid_by_ident(fids_norm, FIFF.FIFFV_POINT_NASION)
        lpa    = _pick_fid_by_ident(fids_norm, FIFF.FIFFV_POINT_LPA)
        rpa    = _pick_fid_by_ident(fids_norm, FIFF.FIFFV_POINT_RPA)
    except Exception as e:
        # Safe fallback: montage without cardinal fiducials
        print(f"[WARN] Could not pick fsaverage fiducials by ident ({e}). "
              f"Proceeding without NAS/LPA/RPA in montage.")

    mont = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=nasion, lpa=lpa, rpa=rpa,  # can be None; montage will still be created
        coord_frame='mri'
    )
    return mont

def make_raw_from_arrays(hbo: np.ndarray, hbr: np.ndarray,
                         sfreq: float, ch_names_hbo: List[str], ch_names_hbr: List[str]) -> mne.io.RawArray:
    """
    Build an MNE RawArray with channel types 'hbo' and 'hbr'. Data arrays are shape (n_times, n_channels).
    """
    assert hbo.shape[0] == hbr.shape[0]
    n_times = hbo.shape[0]
    data = np.concatenate([hbo, hbr], axis=1).T   # (n_channels_total, n_times)
    ch_names = ch_names_hbo + ch_names_hbr
    ch_types = ['hbo'] * len(ch_names_hbo) + ['hbr'] * len(ch_names_hbr)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def build_optode_dict_from_df(opt_df: pd.DataFrame,
                              name_col="Name", xyz_cols=("X","Y","Z"),
                              convert_TR_to_SD=True,
                              units="auto"):
    """
    Return { 'S1': [x,y,z], 'D1': [x,y,z], ... } in METERS for MNE.
    opt_df should list OPTODES (not channels), one row per T#/R# (or S#/D#).

    units: 'auto' -> treat coords as meters if median norm < 1, else mm.
           'm'/'mm' to force.
    """
    df = opt_df.copy()
    df[name_col] = df[name_col].astype(str).str.strip()

    if convert_TR_to_SD:
        # T# -> S#, R# -> D#
        df[name_col] = (df[name_col]
                        .str.replace(r'^T(?=\d+)', 'S', regex=True)
                        .str.replace(r'^R(?=\d+)', 'D', regex=True))

    coords = df.loc[:, xyz_cols].to_numpy(float)
    # units → meters for MNE
    if units == "auto":
        med = float(np.median(np.linalg.norm(coords, axis=1)))
        scale = 1.0 if med < 1.0 else 1/1000.0   # if big → mm → convert to m
    elif units == "m":
        scale = 1.0
    elif units == "mm":
        scale = 1/1000.0
    else:
        raise ValueError("units must be 'auto','m','mm'")

    coords_m = coords * scale
    names = df[name_col].tolist()
    # ensure only S#/D# names exist
    bad = [n for n in names if not (n.startswith('S') or n.startswith('D'))]
    if bad:
        raise ValueError(f"Optode names must be S#/D#. Found: {bad[:5]} ...")

    return {n: coords_m[i] for i, n in enumerate(names)}


def _normalize_fids(fids):
    out = []
    for f in fids:
        if hasattr(f, 'r'):
            out.append({'r': np.array(getattr(f,'r'), float),
                        'ident': getattr(f,'ident', None)})
        else:
            out.append({'r': np.array(f.get('r'), float),
                        'ident': f.get('ident', None)})
    return out

def _pick_fid_by_ident(fids_norm, ident):
    for f in fids_norm:
        if f['ident'] == ident:
            return f['r']
    raise RuntimeError(f"Fid ident={ident} not found. Have: {[f['ident'] for f in fids_norm]}")

def make_fnirs_optode_montage(optode_pos_m: dict,
                              subjects_dir: str,
                              subject: str = 'fsaverage',
                              attach_cardinals=True) -> mne.channels.DigMontage:
    """
    Build a montage where ch_pos contains ONLY optodes ('S#','D#'), not channels.
    """
    nasion = lpa = rpa = None
    if attach_cardinals:
        try:
            fids = mne.coreg.get_mni_fiducials(subject, subjects_dir=subjects_dir)
            nf = _normalize_fids(fids)
            nasion = _pick_fid_by_ident(nf, FIFF.FIFFV_POINT_NASION)
            lpa    = _pick_fid_by_ident(nf, FIFF.FIFFV_POINT_LPA)
            rpa    = _pick_fid_by_ident(nf, FIFF.FIFFV_POINT_RPA)
        except Exception as e:
            print("[WARN] Could not attach fsaverage fiducials:", e)

    mont = mne.channels.make_dig_montage(
        ch_pos=optode_pos_m, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame='mri'
    )
    return mont

def validate_montage_vs_raw(raw: mne.io.BaseRaw, montage: mne.channels.DigMontage):
    """
    Ensure all sources/detectors referenced by raw channel names 'S#_D# <chrom>'
    exist in the montage ch_names.
    """
    m_names = set(montage.ch_names)  # should contain S#/D# only
    missing = set()
    for ch in raw.info['ch_names']:
        if ' ' not in ch:
            continue
        base = ch.split(' ')[0]
        try:
            s, d = base.split('_')
        except ValueError:
            continue
        if s not in m_names:
            missing.add(s)
        if d not in m_names:
            missing.add(d)
    return sorted(missing)


# ---------- HEAD frame from INLR ----------
def make_head_frame_from_fids(fid_sub_mm: dict):
    """
    Neuromag HEAD frame from N/L/R (mm).
    Origin: midpoint(L,R)
    +X: LPA -> RPA
    +Y: origin -> Nasion  (anterior)
    +Z: up (right-handed)
    Returns dict with rotation (3x3), origin (3,), and convenience lambdas.
    """
    N = np.asarray(fid_sub_mm["N"], float) / 1000.0  # meters
    L = np.asarray(fid_sub_mm["L"], float) / 1000.0
    R = np.asarray(fid_sub_mm["R"], float) / 1000.0

    O  = 0.5 * (L + R)                        # origin
    ex = (R - L); ex /= np.linalg.norm(ex)    # +X left->right
    ey = (N - O); ey /= np.linalg.norm(ey)    # +Y posterior->anterior
    ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)   # +Z up
    # re-orthogonalize Y to guarantee orthonormal basis
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)

    # rows of R map world->HEAD via dot([ex,ey,ez], X-O)
    Rw2h = np.vstack([ex, ey, ez])            # shape (3,3)

    def to_head(X_m):
        X = np.asarray(X_m, float)
        return (Rw2h @ (X - O).T).T           # (N,3) in HEAD (m)

    # also return NAS/LPA/RPA in HEAD
    nas_h, lpa_h, rpa_h = to_head(N), to_head(L), to_head(R)

    # L/R sanity (X increases from L to R)
    if (rpa_h[0] - lpa_h[0]) < 0:
        # extremely unlikely with formula above, but keep guard
        Rw2h[0, :] *= -1
        ez = np.cross(Rw2h[0, :], Rw2h[1, :]); ez /= np.linalg.norm(ez)
        Rw2h[2, :] = ez

    return {
        "R_w2h": Rw2h, "O_w": O,
        "to_head": to_head,
        "NAS_head": nas_h.ravel(), "LPA_head": lpa_h.ravel(), "RPA_head": rpa_h.ravel()
    }

def _optode_mask_SD(df, name_col="Name"):
    """Return a boolean mask for rows that are S#/D# only."""
    names = df[name_col].astype(str).str.strip()
    return names.str.match(r'^[SD]\d+$')

def optodes_to_head(optode_df, Rw2h, O, name="Name", already_in_head=False):
    """
    Convert optode df (S#/D#) to HEAD coords (meters).
    If already_in_head=True, skip the transform (avoid double-rotating).
    """
    df = optode_df.copy()
    # keep only S#/D#
    mask = _optode_mask_SD(df, name_col=name)
    df = df.loc[mask].reset_index(drop=True)

    P = df[["X","Y","Z"]].to_numpy(float)

    Pm = P
    Ph = Pm if already_in_head else _to_head(Pm, Rw2h, O)

    names = df[name].astype(str).tolist()
    return {names[i]: Ph[i] for i in range(len(names))}

def make_head_montage_from_optodes(optode_pos_head_m: dict, head_xf: dict):
    """
    Build DigMontage with optodes in HEAD (meters) + NAS/LPA/RPA in HEAD.
    """
    mont = mne.channels.make_dig_montage(
        ch_pos=optode_pos_head_m,
        nasion=head_xf["NAS_head"],
        lpa=head_xf["LPA_head"],
        rpa=head_xf["RPA_head"],
        coord_frame="head"
    )
    return mont

def midpoints_from_channels(raw: mne.io.BaseRaw):
    """
    Return (n_ch, 3) channel midpoints in HEAD, by parsing 'S#_D#' from ch names.
    Requires an optode-based montage already set on raw.
    """
    picks = mne.pick_types(raw.info, fnirs=True)
    pos = np.zeros((len(picks), 3))
    name_to_idx = {d["ch_name"]: i for i, d in enumerate(raw.get_montage().dig_ch_names)}
    # Dig montage stores optodes first; use a safer mapping:
    optode_pos = {d["ch_name"]: d["r"] for d in raw.get_montage()._get_dig() if d["kind"] == FIFF.FIFFV_POINT_EEG}
    # fallback if kind not EEG-tagged: include all entries with names starting S/D
    if not optode_pos:
        optode_pos = {d.get("ch_name", f"S{i}"): d["r"] for d in raw.get_montage()._get_dig()
                      if isinstance(d, dict) and "r" in d and d.get("ch_name","").startswith(("S","D"))}
    for k, ch_idx in enumerate(picks):
        base = raw.info["ch_names"][ch_idx].split(" ")[0]  # 'S#_D#'
        s, d = base.split("_")
        pos[k] = 0.5 * (optode_pos[s] + optode_pos[d])
    return pos


def _head_axes(fid_sub_mm):
    N = np.asarray(fid_sub_mm["N"], float)/1000.0
    L = np.asarray(fid_sub_mm["L"], float)/1000.0
    R = np.asarray(fid_sub_mm["R"], float)/1000.0
    I = np.asarray(fid_sub_mm.get("I", [0,0,0]), float)/1000.0  
    O  = 0.5*(L+R)
    ex = (R-L); ex /= np.linalg.norm(ex)  # L→R
    ey_1 = (N-O); ey_1 /= np.linalg.norm(ey_1)  # back→front (toward nasion)
    ey_2 = (I-O); ey_2 /= np.linalg.norm(ey_2)  # down→up (toward inion)
    ey = 0.5*(ey_1 + ey_2); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    Rw2h = np.vstack([ex, ey, ez])        # world→HEAD rotation
    return Rw2h, O, N, L, R, I

def _to_head(X_m, Rw2h, O):
    X = np.asarray(X_m, float)
    return (Rw2h @ (X - O).T).T

# 2) Optodes df (already S#/D#) → HEAD (meters)
def optodes_to_head(optode_df, Rw2h, O, name="Name"):
    P = optode_df[["X","Y","Z"]].to_numpy(float)
    Pm = P.copy()                           # meters
    Ph = _to_head(Pm, Rw2h, O)              # → HEAD (m)
    names = optode_df[name].astype(str).tolist()
    return {names[i]: Ph[i] for i in range(len(names))}

# 3) Build DigMontage in HEAD with **cardinals** also in HEAD
def make_head_montage(fid_sub_mm, ch_positions):
    Rw2h, O, N, L, R, I = _head_axes(fid_sub_mm)
    nas_h = _to_head(N, Rw2h, O).ravel()
    lpa_h = _to_head(L, Rw2h, O).ravel()
    rpa_h = _to_head(R, Rw2h, O).ravel()
    inion_h = _to_head(I, Rw2h, O).ravel()
    mont = mne.channels.make_dig_montage(
        ch_pos=ch_positions,
        nasion=nas_h, lpa=lpa_h, rpa=rpa_h,
        coord_frame="mri"
    )
    return mont
# ======================================
# 5) Filtering, optional wavelet detrend
# ======================================

def butter_bandpass_apply(X: np.ndarray, sfreq: float, l_hz=0.01, h_hz=0.10) -> np.ndarray:
    """
    4th-order zero-phase Butterworth band-pass, axis=0 time.
    """
    nyq = 0.5 * sfreq
    b, a = butter(N=4, Wn=[l_hz/nyq, h_hz/nyq], btype='band')
    return filtfilt(b, a, X, axis=0)

def wavelet_detrend_channelwise(X: np.ndarray, wavelet='db4', level=5) -> np.ndarray:
    """
    Optional: remove very slow components via wavelet approximation removal.
    Requires pywt.
    """
    import pywt
    T, C = X.shape
    Y = np.zeros_like(X)
    for i in range(C):
        coeffs = pywt.wavedec(X[:, i], wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        rec = pywt.waverec(coeffs, wavelet)
        if len(rec) < T:
            rec = np.pad(rec, (0, T - len(rec)))
        Y[:, i] = rec[:T]
    return Y

# ======================================
# 6) Event handling

# -------- core: extract events from Task column --------
def events_from_task_df(
    df: pd.DataFrame,
    raw: mne.io.BaseRaw,
    time_col: str = "Time(sec)",
    task_col: str = "Task",
    include_rest: bool = False,
    min_event_spacing_sec: float = 0.0,
):
    """
    Convert Task column transitions into an MNE events array.
    - Detects 0->1 transitions as 'task' onsets.
    - Optionally also emits 1->0 transitions as 'rest' onsets.
    - Uses df[time_col] (seconds) and raw.info['sfreq'] to convert to samples.

    Returns
    -------
    events : (n_events, 3) int array
    event_id : dict, e.g. {'task': 1} or {'task': 1, 'rest': 2}
    """
    if time_col not in df or task_col not in df:
        raise ValueError(f"df must contain '{time_col}' and '{task_col}' columns")

    times = np.asarray(df[time_col], float)
    if not np.all(np.isfinite(times)):
        raise ValueError("Time column contains NaNs/inf")

    sfreq = float(raw.info["sfreq"])
    first_samp = int(getattr(raw, "first_samp", 0))

    # Robust binarization of Task (tolerates floats/NaNs)
    task = pd.Series(df[task_col]).astype(float).ffill().fillna(0.0).to_numpy()
    task_bin = (task >= 0.5).astype(int)

    # Find rising and falling edges
    d = np.diff(task_bin, prepend=task_bin[0])
    rise_idx = np.where(d == +1)[0]          # 0 -> 1 (task onsets)
    fall_idx = np.where(d == -1)[0]          # 1 -> 0 (rest onsets)

    # Convert to sample indices (align to raw)
    def to_samples(idxs):
        t = times[idxs]
        return np.round(t * sfreq).astype(int) + first_samp

    samp_task = to_samples(rise_idx)

    # Optional: enforce minimum spacing to avoid duplicated events
    if min_event_spacing_sec and min_event_spacing_sec > 0:
        min_gap = int(np.round(min_event_spacing_sec * sfreq))
        keep = []
        last = -np.inf
        for s in samp_task:
            if s - last >= min_gap:
                keep.append(True)
                last = s
            else:
                keep.append(False)
        samp_task = samp_task[np.array(keep, dtype=bool)]

    ev_task = np.column_stack([samp_task, np.zeros_like(samp_task), np.full_like(samp_task, 1)])

    events = ev_task
    event_id = {"task": 1}

    if include_rest:
        samp_rest = to_samples(fall_idx)
        ev_rest = np.column_stack([samp_rest, np.zeros_like(samp_rest), np.full_like(samp_rest, 2)])
        events = np.vstack([events, ev_rest])
        event_id["rest"] = 2

    # Sort by sample just in case
    order = np.argsort(events[:, 0])
    events = events[order]

    return events.astype(int), event_id


# -------- optional: add annotations for continuous task blocks --------
def annotate_task_blocks(
    df: pd.DataFrame,
    raw: mne.io.BaseRaw,
    time_col: str = "Time(sec)",
    task_col: str = "Task",
    task_desc: str = "task",
    rest_desc: str = None,  # e.g., "rest"
):
    """
    Create mne.Annotations from continuous runs of Task==1 (and optionally Task==0).
    Attaches them to raw and also returns the created Annotations.
    """
    times = np.asarray(df[time_col], float)
    sfreq = float(raw.info["sfreq"])

    task = pd.Series(df[task_col]).astype(float).ffill().fillna(0.0).to_numpy()
    task_bin = (task >= 0.5).astype(int)

    # Find run starts/ends for 1s (and optionally 0s)
    def runs_of(val):
        x = (task_bin == val).astype(int)
        dx = np.diff(x, prepend=0, append=0)
        starts = np.where(dx == +1)[0]
        ends   = np.where(dx == -1)[0]
        return starts, ends

    ann = []

    # Task blocks
    s1, e1 = runs_of(1)
    for s, e in zip(s1, e1):
        onset_sec = times[s]
        dur_sec = max(0.0, times[e - 1] - times[s])
        if dur_sec > 0 and task_desc:
            ann.append((onset_sec, dur_sec, task_desc))

    # Rest blocks (optional)
    if rest_desc is not None:
        s0, e0 = runs_of(0)
        for s, e in zip(s0, e0):
            onset_sec = times[s]
            dur_sec = max(0.0, times[e - 1] - times[s])
            if dur_sec > 0:
                ann.append((onset_sec, dur_sec, rest_desc))

    if ann:
        onsets, durations, descr = zip(*ann)
        new_annot = mne.Annotations(onset=onsets, duration=durations, description=descr, orig_time=raw.info["meas_date"])
        # Merge with any existing
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw.set_annotations(raw.annotations + new_annot)
        else:
            raw.set_annotations(new_annot)
        return new_annot
    return mne.Annotations([], [], [], orig_time=raw.info["meas_date"])


# -------- convenience: build Epochs directly from df + raw --------
def epochs_from_task_df(
    df: pd.DataFrame,
    raw: mne.io.BaseRaw,
    epoch_len_sec: float = 15.0,
    base_win_sec: float = 0.5,
    include_rest: bool = False,
    reject_by_annotation: bool = True,
):
    """
    Create mne.Epochs aligned to Task onsets (and optionally Rest onsets) by
    first extracting events from the DataFrame and then calling mne.Epochs.
    """
    events, event_id = events_from_task_df(
        df, raw, time_col="Time(sec)", task_col="Task", include_rest=include_rest
    )
    # MNE epoch window: [-baseline, +epoch_len]
    tmin = -float(base_win_sec)
    tmax = float(epoch_len_sec)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0.0),
        preload=True,
        reject_by_annotation=reject_by_annotation,
        detrend=None,
        verbose=False,
    )
    return epochs, events, event_id

# ======================================
# Plotting helpers
# ======================================

def plot_sample_comparison(raw: np.ndarray, H_clean: np.ndarray, H_global: np.ndarray, channel_index=1, time_window=(1500,1600), title_suffix="Raw vs Clean vs Global"):
    """
    Parameters:
    - raw: Original data matrix (timepoints x channels)
    - H_clean: Cleaned data matrix (timepoints x channels)
    - H_global: Global component matrix (timepoints x channels)
    - Channel_index: Channel to be plotted
    - Time_window: Time window of the plot (Tmin, Tmax)
    - Title_suffix: Something to add at the end of the title 
    """
    i = channel_index -1
    plt.figure(figsize=(16,9))
    plt.plot(raw[:,i],label='raw', color = 'black')
    plt.plot(H_global[:,i],label='global', color = 'red', linestyle='--', alpha=0.7)
    plt.plot(H_clean[:,i],label='clean', color = 'blue', linestyle='--', alpha=0.7)

    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.xlim(time_window)
    plt.title(f"Channel {channel_index} Comparison {title_suffix}")
    plt.legend()
    plt.show()

def plot_correlation_matrices_and_kernel(raw, H_clean, H_global, kernel):
    """
    Plots correlation matrices for original data, cleaned data, global component,
    and the convolution kernel.
    
    Parameters:
    - raw: Original data matrix (timepoints x channels)
    - H_clean: Cleaned data matrix (timepoints x channels)
    - H_global: Global component matrix (timepoints x channels)
    - kernel: Convolution kernel matrix (channels x channels)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Original data correlation (match MATLAB: corrcoef(data0(:,:,1)'))
    corr_orig = np.corrcoef(raw[:, :].T)
    im1 = axes[0, 0].imshow(corr_orig, vmin=-1, vmax=1, cmap='RdBu_r');
    axes[0, 0].set_title('Data, Oxy')
    plt.colorbar(im1, ax=axes[0, 0])

    # Cleaned data correlation (match MATLAB: corrcoef(cleanData(:,:,1)'))
    corr_clean = np.corrcoef(H_clean[:, :].T)
    im2 = axes[0, 1].imshow(corr_clean, vmin=-1, vmax=1, cmap='RdBu_r');
    axes[0, 1].set_title('Cleaned Data, Oxy')
    plt.colorbar(im2, ax=axes[0, 1])

    # Global component correlation (match MATLAB: corrcoef(globalC(:,:,1)'))
    corr_global = np.corrcoef(H_global[:, :].T)
    im3 = axes[1, 0].imshow(corr_global, vmin=-1, vmax=1, cmap='RdBu_r');
    axes[1, 0].set_title('Global Component, Oxy')
    plt.colorbar(im3, ax=axes[1, 0])

    # Kernel visualization
    im4 = axes[1, 1].imshow(kernel, cmap='viridis')
    axes[1, 1].set_title('Convolution Kernel')
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig('globalremovaldemo.png', dpi=150, bbox_inches='tight')
    plt.show()




# ======================================
# 7) The main pipeline
# ======================================

def run_pipeline(
    # --- geometry inputs ---
    channel_coords: np.ndarray,           # (n_ch, 3) channel positions in subject MRI space (mm)
    fid_subj_mm: Dict[str, np.ndarray],   # keys: "N","L","R" required; "I","CZ" optional; values in mm
    ch_xyz_mm: np.ndarray,                # (n_ch,3) channel positions in subject space (mm)
    ch_names_hbo: List[str],
    ch_names_hbr: List[str],
    # --- optode inputs ---
    optode_df: pd.DataFrame,              # columns: "Name", "X", "Y", "Z" (mm)
    # --- data inputs ---
    H_hbo: np.ndarray,                    # (n_times, n_ch) HbO time series
    H_hbr: np.ndarray,                    # (n_times, n_ch) HbR time series (same chans, same order)
    sfreq: float,
    # --- processing params ---
    sigma_deg: float = 48.0,
    do_wavelet_detrend: bool = False,
    bp_low: float = 0.01,
    bp_high: float = 0.10,
    tmin: float = -2.0,                   # epoching window (s), if events provided
    tmax: float = 20.0,
    baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0),
    events: Optional[np.ndarray] = None,  # shape (n_events, 3) [sample, 0, event_id]
    event_id: Optional[Dict[str, int]] = None,
    subjects_dir: Optional[str] = None
):
    """
    Returns dict with:
      - raw (MNE Raw), montage set
      - H_*_neur/global (arrays)
      - xf (transform diagnostics)
      - epochs (if events provided)
    """
    assert H_hbo.shape == H_hbr.shape, "HbO/HbR must have same shape"
    n_times, n_ch = H_hbo.shape
    assert len(ch_names_hbo) == len(ch_names_hbr) == n_ch

    if bp_low is not None or bp_high is not None:
        l = bp_low if bp_low is not None else 0.
        h = bp_high if bp_high is not None else None
        if h is None:
            raise ValueError("Provide both low and high for band-pass in this helper.")
        H_hbo_bpf = butter_bandpass_apply(H_hbo, sfreq, l_hz=l, h_hz=h)
        H_hbr_bpf = butter_bandpass_apply(H_hbr, sfreq, l_hz=l, h_hz=h)


    raw = make_raw_from_arrays(H_hbo_bpf, H_hbr_bpf, sfreq, ch_names_hbo, ch_names_hbr)
    matplotlib.use('TkAgg')
    hbo_pick = mne.pick_types(raw.info, fnirs='hbo')
    hbr_pick = mne.pick_types(raw.info, fnirs='hbr')
    raw.plot(scalings=1e-2, show=True, block=True)
    bads = raw.info["bads"]
    bad_idxs = [idx for idx, name in enumerate(raw.ch_names) if name in bads]

    # raw.drop_channels(raw.info["bads"])

    # 1) Align channels to fsaverage MRI coordinates (meters)
    ch_fsavg_mri_m, xf = align_to_fsaverage(fid_subj_mm, ch_xyz_mm,
                                            allow_scaling=True,
                                            subjects_dir=subjects_dir)
    Rw2h, O, _, _, _, _ = _head_axes(fid_subj_mm)

    # If the optode_corrs_df_converted is ALREADY in HEAD, set already_in_head=True.
    optode_pos_head_m = optodes_to_head(optode_df, Rw2h, O)

    mont = make_head_montage(fid_subj_mm, optode_pos_head_m)  # coord_frame='head'

    # 2) Build geodesic Gaussian kernel on a sphere (use meters, but angles are unitless)
    channel_coords_sph = cartesian_to_spherical(channel_coords)
    K = gaussian_kernel_geodesic(channel_coords_sph, sigma_deg=sigma_deg, row_normalize=False)
    
    if len(bad_idxs) > 0:
        bad_idxs = np.unique(bad_idxs)
        good_mask = np.ones(K.shape[0], dtype=bool)
        bad_idxs_hbo = [idx for idx in bad_idxs if idx in hbo_pick]
        good_mask[bad_idxs_hbo] = False
        K_hbo = K[np.ix_(good_mask, good_mask)]

    
        good_mask = np.ones(K.shape[0], dtype=bool)
        bad_idxs_hbr = [idx - len(hbo_pick) for idx in bad_idxs if idx in hbr_pick]

        good_mask[bad_idxs_hbr] = False
        K_hbr = K[np.ix_(good_mask, good_mask)]
    else:
        K_hbo = K
        K_hbr = K
    # 3) Optional pre-clean: wavelet detrend, then band-pass (recommended order)

        
    # Pick hbo and hbr separately for processing from raw
    ch_names = np.array(raw.ch_names)
    hbo_pick = mne.pick_types(raw.info, fnirs='hbo', exclude=raw.info['bads'])
    hbr_pick = mne.pick_types(raw.info, fnirs='hbr', exclude=raw.info['bads'])
    hbo_channels = [name for indx, name in enumerate(ch_names) if 'hbo' in name and indx in hbo_pick]
    hbr_channels = [name for indx, name in enumerate(ch_names) if 'hbr' in name and indx in hbr_pick]
    H_hbo_unclean = raw.get_data(picks=hbo_pick).T   # (n_times, n_ch)
    H_hbr_unclean = raw.get_data(picks=hbr_pick).T   # (n_times, n_ch)

    # 4) Spatial PC filter (Zhang/Klein) per chromophore
    H_hbo_neur, H_hbo_global = spatial_pc_filter(H_hbo_unclean, K_hbo, row_normalize=True)
    H_hbr_neur, H_hbr_global = spatial_pc_filter(H_hbr_unclean, K_hbr, row_normalize=True)

    if do_wavelet_detrend:
        H_hbo_detrended_neur = wavelet_detrend_channelwise(H_hbo_neur)
        H_hbr_detrended_neur = wavelet_detrend_channelwise(H_hbr_neur)
        cleaned_raw = make_raw_from_arrays(H_hbo_detrended_neur, H_hbr_detrended_neur, sfreq, hbo_channels, hbr_channels)

    else:
        cleaned_raw = make_raw_from_arrays(H_hbo_neur, H_hbr_neur, sfreq, hbo_channels, hbr_channels)

    missing = validate_montage_vs_raw(cleaned_raw, mont)

    if len(missing):
        raise RuntimeError(f"Optode names missing from montage: {missing[:10]} ... "
                        f"(total {len(missing)}). "
                        "Make sure your optode table lists positions for ALL S#/D# used by channels.")

    # (D) Attach montage
    cleaned_raw.set_montage(mont, on_missing='raise', match_case=False)
    # ch_head = midpoints_from_channels(raw)                 # (n_ch, 3) in meters
    # unit = ch_head / np.linalg.norm(ch_head, axis=1, keepdims=True)
    '''  
    #  Attach montage in MRI space (fsaverage)
    mont = make_mne_montage_fsaverage(ch_names_hbo + ch_names_hbr, 
                                      np.vstack([ch_fsavg_mri_m, ch_fsavg_mri_m]),
                                      xf["subjects_dir"])

    raw.set_montage(mont, match_case=False, on_missing='warn')
    '''

    # 7) Optional: epoching if events provided

    result = {
        "raw": cleaned_raw,
        "xf": xf,
        "H_hbo_neur": H_hbo_neur,
        "H_hbo_global": H_hbo_global,
        "H_hbr_neur": H_hbr_neur,
        "H_hbr_global": H_hbr_global,
        "K": K
    }

    if events is not None and event_id is not None:
        # MNE expects sample indices for events; ensure int dtype
        events = np.asarray(events, int)
        epochs = mne.Epochs(cleaned_raw, events=events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=baseline,
                            preload=True, detrend=None, verbose=False)
        result["epochs"] = epochs

    return result


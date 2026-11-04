# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""File export: UFF dataset 58 (FRFs, coherence) via pyuff, and TDMS time data via npTDMS."""

from __future__ import annotations

import datetime as _dt
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyuff
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

from .config import DIRECTION_CODE, AppConfig, ChannelConfig

# UFF58 "specific data type" codes
_SPEC_TYPE = {
    "Accelerometer (IEPE)": 12,   # acceleration
    "Force (IEPE)": 13,           # excitation force
    "Strain gauge (bridge)": 3,   # strain
    "Microphone (IEPE)": 15,      # pressure
    "Pressure (IEPE)": 15,        # pressure
    "Voltage": 1,                 # general
    "Custom": 1,                  # general
}
FUNC_FRF, FUNC_COH, FUNC_GENERAL = 4, 6, 0
REAL_FUNCS = ("Coherence", "FundamentalFraction")


def _uff_dataset(func_type, data, freqs, resp: ChannelConfig, ref: ChannelConfig,
                 id1: str, id2: str, id3: str, coherence: bool, rec_no: int) -> dict:
    complex_data = np.iscomplexobj(data)
    freqs = np.asarray(freqs, dtype=float)
    even = len(freqs) < 3 or np.allclose(np.diff(freqs), freqs[1] - freqs[0], rtol=1e-9, atol=1e-12)
    return pyuff.prepare_58(
        binary=0,
        id1=id1[:80], id2=id2[:80], id3=id3[:80], id4="", id5="",
        func_type=func_type, ver_num=0, load_case_id=0,
        rsp_ent_name=resp.name[:10], rsp_node=int(resp.node), rsp_dir=DIRECTION_CODE[resp.direction],
        ref_ent_name=ref.name[:10], ref_node=int(ref.node), ref_dir=DIRECTION_CODE[ref.direction],
        ord_data_type=6 if complex_data else 4,     # 6 = complex double, 4 = real double
        num_pts=len(freqs),
        abscissa_spacing=1 if even else 0, abscissa_min=float(freqs[0]),
        abscissa_inc=float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
        abscissa_spec_data_type=18, abscissa_len_unit_exp=0, abscissa_force_unit_exp=0,
        abscissa_temp_unit_exp=0, abscissa_axis_units_lab="Hz",
        ordinate_spec_data_type=0 if coherence else _SPEC_TYPE.get(resp.sensor_type, 1),
        ordinate_len_unit_exp=0, ordinate_force_unit_exp=0, ordinate_temp_unit_exp=0,
        ordinate_axis_units_lab="" if coherence else resp.units,
        orddenom_spec_data_type=0 if coherence else _SPEC_TYPE.get(ref.sensor_type, 13),
        orddenom_len_unit_exp=0, orddenom_force_unit_exp=0, orddenom_temp_unit_exp=0,
        orddenom_axis_units_lab="" if coherence else ref.units,
        z_axis_spec_data_type=0, z_axis_len_unit_exp=0, z_axis_force_unit_exp=0,
        z_axis_temp_unit_exp=0, z_axis_axis_units_lab="", z_axis_value=float(rec_no),
        data=np.asarray(data), x=np.asarray(freqs, dtype=float),
    )


def write_uff58(path: str, func: str, freqs: np.ndarray, data: np.ndarray,
                responses: Sequence[ChannelConfig], ref: ChannelConfig, info: str) -> str:
    """Write one UFF file containing one dataset 58 per response channel.
    func: 'H1' | 'H2' | 'Hv' | 'Coherence' | 'FundamentalFraction' (stepped sine, written as a
    general real function).  data: (n_resp, nf)."""
    if os.path.exists(path):
        os.remove(path)
    is_coh = func in REAL_FUNCS
    ftype = FUNC_COH if func == "Coherence" else (FUNC_GENERAL if is_coh else FUNC_FRF)
    stamp = _dt.datetime.now().strftime("%d-%b-%y %H:%M:%S")
    uff = pyuff.UFF(path)
    sets = []
    for i, resp in enumerate(responses):
        tag = "FundFrac" if func == "FundamentalFraction" else func
        id1 = f"{tag} {resp.name} {resp.node}{resp.direction} / {ref.name} {ref.node}{ref.direction}"
        sets.append(_uff_dataset(ftype,
                                 np.real(data[i]) if is_coh else data[i],
                                 freqs, resp, ref, id1, info, stamp, is_coh, i + 1))
    uff.write_sets(sets, mode="overwrite")
    return path


def band_slice(freqs: np.ndarray, fmin: float, fmax: float) -> slice:
    i0 = int(np.searchsorted(freqs, fmin - 1e-9))
    i1 = int(np.searchsorted(freqs, fmax + 1e-9))
    return slice(i0, max(i1, i0 + 1))


# ----------------------------------------------------------------------------- TDMS
def _channel_props(c: Optional[ChannelConfig], fs: float, n: int, t0) -> Dict:
    props = {"wf_increment": 1.0 / fs, "wf_start_offset": 0.0, "wf_samples": n,
             "wf_start_time": t0}
    if c is None:  # drive signal
        props.update({"unit_string": "V", "description": "Commanded AO drive signal"})
        return props
    props.update({
        "unit_string": c.units, "NI_ChannelName": c.name, "physical_channel": c.physical,
        "node": int(c.node), "direction": c.direction, "sensor_type": c.type_label,
        "sensitivity": float(c.sensitivity), "sensitivity_units": c.sens_units,
        "is_reference": bool(c.is_reference),
    })
    return props


def root_properties(cfg: AppConfig, fs_actual: float) -> Dict:
    o = cfg.output
    return {
        "name": cfg.save.basename,
        "description": "Shaker modal test",
        "excitation": o.excitation, "drive_rms_V": o.rms, "f_min_Hz": o.fmin, "f_max_Hz": o.fmax,
        "sample_rate_Hz": fs_actual, "block_size": o.block_size,
        "dwell_blocks": o.dwell_blocks, "realizations": o.realizations,
        "decay_time_s": o.decay_time, "window": o.window,
    }


class TdmsBlockWriter:
    """Appends one group per realization to a TDMS file (file stays valid if a run is aborted)."""

    def __init__(self, path: str, root_props: Dict):
        self.path = path
        self._w = TdmsWriter(path)
        self._w.__enter__()
        self._w.write_segment([RootObject(properties=root_props)])

    def write_group(self, group: str, channels: List[ChannelConfig], data: np.ndarray,
                    extras: Dict[str, np.ndarray], fs: float, group_props: Optional[Dict] = None) -> None:
        """extras: output signals to store alongside the inputs, e.g. {'Drive': …, 'Trigger': …}."""
        t0 = np.datetime64(_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None), "us")
        objs = [GroupObject(group, properties=group_props or {})]
        for c, row in zip(channels, data):
            objs.append(ChannelObject(group, c.name, np.asarray(row, dtype=np.float64),
                                      properties=_channel_props(c, fs, row.size, t0)))
        for name, sig in extras.items():
            props = _channel_props(None, fs, sig.size, t0)
            if name != "Drive":
                props.update({"unit_string": "", "description": "Camera trigger output (V for AO, 0/1 for DO)"})
            objs.append(ChannelObject(group, name, np.asarray(sig, dtype=np.float64), properties=props))
        self._w.write_segment(objs)

    def close(self) -> None:
        if self._w is not None:
            self._w.__exit__(None, None, None)
            self._w = None


def write_realization_tdms(path: str, root_props: Dict, group: str, channels: List[ChannelConfig],
                           data: np.ndarray, extras: Dict[str, np.ndarray], fs: float, group_props: Dict) -> str:
    w = TdmsBlockWriter(path, root_props)
    try:
        w.write_group(group, channels, data, extras, fs, group_props)
    finally:
        w.close()
    return path

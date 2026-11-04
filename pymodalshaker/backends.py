# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""DAQ backends with an identical streaming interface:

    fs_actual, notes = backend.configure()
    backend.write_block(x)       # queue N output samples (call `queue depth` times before start)
    backend.start()
    data = backend.read_block()  # (n_channels, N) in engineering units, blocks until available
    backend.stop(); backend.close()

NIDAQBackend  - real cDAQ hardware, continuous AI + AO, regeneration disabled.
SimBackend    - software model of shaker + 3-mode structure, paced in real time; for trying the
                GUI and testing without hardware.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from .config import AppConfig, ChannelConfig, voltage_scaled


# ----------------------------------------------------------------------------- discovery
def discover_devices() -> Dict:
    """List NI devices / physical channels. Never raises."""
    out = {"ok": False, "message": "", "devices": [], "ai": [], "ao": [], "do": [], "pfi": []}
    try:
        import nidaqmx.system
        system = nidaqmx.system.System.local()
        drv = system.driver_version
        for dev in system.devices:
            try:
                ai = list(dev.ai_physical_chans.channel_names)
                ao = list(dev.ao_physical_chans.channel_names)
            except Exception:
                ai, ao = [], []
            try:
                out["do"] += list(dev.do_lines.channel_names)
            except Exception:
                pass
            try:
                out["pfi"] += [t for t in dev.terminals if "PFI" in t.upper()]
            except Exception:
                pass
            out["devices"].append(f"{dev.name} ({dev.product_type})")
            out["ai"] += ai
            out["ao"] += ao
        out["ok"] = True
        out["message"] = (f"NI-DAQmx {drv.major_version}.{drv.minor_version}: "
                          + (", ".join(out["devices"]) if out["devices"] else "no devices found"))
    except Exception as e:  # driver not installed, etc.
        out["message"] = f"NI-DAQmx not available ({type(e).__name__}: {e})"
    return out


def chassis_of(physical_channel: str) -> str:
    """'cDAQ1Mod2/ao0' -> 'cDAQ1' (asks DAQmx first, falls back to the name)."""
    module = physical_channel.split("/")[0].strip()
    try:
        import nidaqmx.system
        return nidaqmx.system.Device(module).compact_daq_chassis_device.name
    except Exception:
        m = re.match(r"(.+?)Mod\d+$", module)
        return m.group(1) if m else module


def ao_module_info(physical_channel: str) -> Tuple[float, str]:
    """(largest |V| the module's AO ranges allow, product type) for an AO channel; (0, '') if unknown."""
    try:
        import nidaqmx.system
        dev = nidaqmx.system.Device(physical_channel.split("/")[0].strip())
        rngs = list(dev.ao_voltage_rngs)          # flat [min1, max1, min2, max2, ...]
        return (max(abs(v) for v in rngs) if rngs else 0.0), dev.product_type
    except Exception:
        return 0.0, ""


def ao_common_range(drive_phys: str, drive_limit: float, trig_phys: str = "", trig_level: float = 0.0,
                    info=ao_module_info) -> float:
    """One ±range for every channel of the AO task (some modules, e.g. the NI 9260, reject channels
    with different ranges): large enough for the drive limit and trigger level, never larger than
    what the modules can output. Raises ValueError with a clear message if a level cannot be met."""
    rng = max(drive_limit, trig_level if trig_phys else 0.0)
    for name, phys, need in (("Shaker drive limit", drive_phys, drive_limit),
                             ("Trigger level", trig_phys, trig_level)):
        if not phys:
            continue
        vmax, product = info(phys)
        if vmax and need > vmax + 1e-9:
            raise ValueError(f"{name} {need:g} V exceeds the ±{vmax:g} V output range of "
                             f"{phys.split('/')[0]} ({product}). Lower it on the Output Setup tab"
                             + (", or use a DO line for the trigger." if name.startswith("Trigger") else "."))
        if vmax:
            rng = min(rng, vmax)
    return rng


# ----------------------------------------------------------------------------- NI hardware
class NIDAQBackend:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.channels: List[ChannelConfig] = cfg.enabled_channels()
        self.n = cfg.output.block_size
        self.ai = self.ao = self.do = None
        self.trig = cfg.trigger if cfg.trigger.enabled else None
        self._scale = np.ones(len(self.channels))

    # ------------------------------------------------------------------ setup
    def _add_channel(self, c: ChannelConfig):
        import nidaqmx.constants as K
        ai = self.ai.ai_channels
        r = abs(c.max_range)
        excit_src = K.ExcitationSource.INTERNAL if c.iepe_ma > 0 else K.ExcitationSource.NONE
        excit_a = c.iepe_ma / 1000.0
        t = c.sensor_type
        if t.startswith("Accelerometer"):
            units = K.AccelUnits.G if c.units == "g" else K.AccelUnits.METERS_PER_SECOND_SQUARED
            ch = ai.add_ai_accel_chan(c.physical, c.name, min_val=-r, max_val=r, units=units,
                                      sensitivity=c.sensitivity,
                                      sensitivity_units=K.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                                      current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Force"):
            lbf = c.units == "lbf"
            ch = ai.add_ai_force_iepe_chan(
                c.physical, c.name, min_val=-r, max_val=r,
                units=K.ForceUnits.POUNDS if lbf else K.ForceUnits.NEWTONS,
                sensitivity=c.sensitivity,
                sensitivity_units=(K.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_POUND if lbf
                                   else K.ForceIEPESensorSensitivityUnits.MILLIVOLTS_PER_NEWTON),
                current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Microphone"):
            spl = 20 * np.log10(max(r, 1e-3) / 20e-6)
            ch = ai.add_ai_microphone_chan(c.physical, c.name, mic_sensitivity=c.sensitivity,
                                           max_snd_press_level=float(spl),
                                           current_excit_source=excit_src, current_excit_val=excit_a)
        elif t.startswith("Strain"):
            bridge = getattr(K.StrainGageBridgeType, c.bridge.upper().replace(" ", "_"))
            ch = ai.add_ai_strain_gage_chan(c.physical, c.name, min_val=-r, max_val=r,
                                            strain_config=bridge, voltage_excit_val=c.excit_v,
                                            gage_factor=c.gage_factor,
                                            nominal_gage_resistance=c.nominal_r,
                                            poisson_ratio=c.poisson)
        else:  # Voltage, Pressure (IEPE), Custom: a voltage channel scaled in software by mV/unit
            vmax = r * c.sensitivity / 1000.0
            ch = ai.add_ai_voltage_chan(c.physical, c.name, min_val=-vmax, max_val=vmax)
            if c.iepe_ma > 0:
                ch.ai_excit_src = K.ExcitationSource.INTERNAL
                ch.ai_excit_val = excit_a
        if c.coupling in ("AC", "DC"):
            ch.ai_coupling = K.Coupling.AC if c.coupling == "AC" else K.Coupling.DC
        return 1000.0 / c.sensitivity if voltage_scaled(t) else 1.0

    def configure(self) -> Tuple[float, List[str]]:
        import nidaqmx
        import nidaqmx.constants as K
        from nidaqmx.stream_readers import AnalogMultiChannelReader
        from nidaqmx.stream_writers import AnalogMultiChannelWriter, AnalogSingleChannelWriter

        o = self.cfg.output
        notes = []
        self._set_trigger_idle()
        self.ai = nidaqmx.Task("ModalAI")
        self.ao = nidaqmx.Task("ModalAO")
        self._scale = np.array([self._add_channel(c) for c in self.channels])

        ai_buf = int(max(10 * o.fs, 8 * self.n))
        self.ai.timing.cfg_samp_clk_timing(o.fs, sample_mode=K.AcquisitionType.CONTINUOUS,
                                           samps_per_chan=ai_buf)
        fs = float(self.ai.timing.samp_clk_rate)          # delta-sigma modules coerce the rate
        if abs(fs - o.fs) > 1e-6 * o.fs:
            notes.append(f"AI sample rate coerced by hardware: {o.fs:g} → {fs:.6g} Hz")
        if self.trig is not None and self.trig.frame_sync:
            # route the AI sample clock itself to a PFI pin: one pulse per acquired sample, starting
            # with sample 0, for as long as the task runs -> camera frames cannot drift from samples
            ex = self.ai.export_signals
            ex.export_signal(K.Signal.SAMPLE_CLOCK, self.trig.physical)
            try:
                ex.samp_clk_output_behavior = K.ExportAction.PULSE
                ex.samp_clk_pulse_polarity = (K.Polarity.ACTIVE_LOW if self.trig.polarity.startswith("Active low")
                                              else K.Polarity.ACTIVE_HIGH)
            except Exception as e:
                notes.append(f"Note: could not set exported clock polarity ({e}); using the default.")
            notes.append(f"Frame sync: AI sample clock exported to {self.trig.physical} "
                         f"({fs:.6g} pulses/s, first pulse = sample 0). Set the camera to external sync.")
            self.trig = None   # nothing else to generate

        chassis = chassis_of(o.physical)
        trig_ao = self.trig is not None and self.trig.kind == "AO"
        trig_do = self.trig is not None and self.trig.kind == "DO"
        # One common output range for every channel in the AO task (some modules, e.g. the NI 9260,
        # reject channels with different ranges), large enough for drive limit and trigger level,
        # and checked against what each module can actually output.
        rng = ao_common_range(o.physical, o.max_voltage,
                              self.trig.physical if trig_ao else "", self.trig.level_v if trig_ao else 0.0)
        self.ao.ao_channels.add_ao_voltage_chan(o.physical, "Drive", min_val=-rng, max_val=rng)
        if trig_ao:   # trigger = 2nd channel of the same AO task -> sample-exact with the drive
            self.ao.ao_channels.add_ao_voltage_chan(self.trig.physical, "Trigger", min_val=-rng, max_val=rng)
            _, product = ao_module_info(self.trig.physical)
            if "926" in product:   # NI 9260 / 9262 / 9269 family: delta-sigma output
                notes.append(f"Note: {product} is a delta-sigma output module. Its reconstruction filter "
                             "delays the trigger pulse and adds some ringing to its edges. Use a pulse of "
                             "≥ 10 samples, or a DO line (NI 9401) for clean, undelayed edges.")
        use_ai_clk = o.clock_mode.startswith("AO clocked")
        clk_src = f"/{chassis}/ai/SampleClock" if use_ai_clk else ""
        q = max(2, o.ao_queue_blocks)
        self.ao.timing.cfg_samp_clk_timing(fs, source=clk_src, sample_mode=K.AcquisitionType.CONTINUOUS,
                                           samps_per_chan=self.n * 2 * q)
        self.ao.out_stream.regen_mode = K.RegenerationMode.DONT_ALLOW_REGENERATION
        self.ao.out_stream.output_buf_size = self.n * 2 * q
        if not use_ai_clk:
            self.ao.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{chassis}/ai/StartTrigger")
            fs_ao = float(self.ao.timing.samp_clk_rate)
            if abs(fs_ao - fs) > 1e-9 * fs:
                notes.append(f"WARNING: AO rate {fs_ao:.9g} Hz ≠ AI rate {fs:.9g} Hz. The excitation "
                             "will not be exactly periodic in the AI block (leakage). Choose a rate both "
                             "modules support exactly, or try 'AO clocked by AI sample clock'.")

        if trig_do:   # hardware-timed digital line (needs a correlated DIO module, e.g. NI 9401)
            self.do = nidaqmx.Task("ModalTrigDO")
            self.do.do_channels.add_do_chan(self.trig.physical, "Trigger",
                                            line_grouping=K.LineGrouping.CHAN_PER_LINE)
            self.do.timing.cfg_samp_clk_timing(fs, source=clk_src, sample_mode=K.AcquisitionType.CONTINUOUS,
                                               samps_per_chan=self.n * 2 * q)
            self.do.out_stream.regen_mode = K.RegenerationMode.DONT_ALLOW_REGENERATION
            self.do.out_stream.output_buf_size = self.n * 2 * q
            if not use_ai_clk:
                self.do.triggers.start_trigger.cfg_dig_edge_start_trig(f"/{chassis}/ai/StartTrigger")
                fs_do = float(self.do.timing.samp_clk_rate)
                if abs(fs_do - fs) > 1e-9 * fs:
                    notes.append(f"WARNING: DO trigger rate {fs_do:.9g} Hz ≠ AI rate {fs:.9g} Hz: pulse "
                                 "positions will drift from the AI sample index. Try 'AO clocked by AI "
                                 "sample clock'.")
        if self.trig is not None:
            notes.append(f"Camera trigger on {self.trig.physical} ({self.trig.kind}), "
                         f"{self.trig.pulse_samples}-sample pulse.")

        if self.cfg.null_bridges and any(c.sensor_type.startswith("Strain") for c in self.channels):
            self.ai.perform_bridge_offset_nulling_cal(skip_unsupported_channels=True)
            notes.append("Strain-bridge offsets nulled.")

        self.ai.control(K.TaskMode.TASK_COMMIT)
        self.ao.control(K.TaskMode.TASK_COMMIT)
        if self.do is not None:
            self.do.control(K.TaskMode.TASK_COMMIT)
        self.reader = AnalogMultiChannelReader(self.ai.in_stream)
        self.writer = (AnalogMultiChannelWriter(self.ao.out_stream, auto_start=False) if trig_ao
                       else AnalogSingleChannelWriter(self.ao.out_stream, auto_start=False))
        self._trig_ao = trig_ao
        self._buf = np.zeros((len(self.channels), self.n))
        self.fs = fs
        return fs, notes

    # ------------------------------------------------------------------ streaming
    def write_block(self, x: np.ndarray, trig: np.ndarray = None) -> None:
        if self._trig_ao:
            self.writer.write_many_sample(np.ascontiguousarray(np.vstack([x, trig]), dtype=np.float64),
                                          timeout=10.0)
        else:
            self.writer.write_many_sample(np.ascontiguousarray(x, dtype=np.float64), timeout=10.0)
        if self.do is not None:
            self.do.write((np.asarray(trig) > 0.5).tolist(), auto_start=False, timeout=10.0)

    def start(self) -> None:
        self.ao.start()   # armed: waits for AI start trigger / AI sample clock
        if self.do is not None:
            self.do.start()
        self.ai.start()

    def read_block(self) -> np.ndarray:
        self.reader.read_many_sample(self._buf, number_of_samples_per_channel=self.n, timeout=10.0)
        return self._buf * self._scale[:, None]

    def backlog(self) -> int:
        try:
            return int(self.ai.in_stream.avail_samp_per_chan)
        except Exception:
            return 0

    def stop(self) -> None:
        for t in (self.ai, self.ao, self.do):
            try:
                if t is not None:
                    t.stop()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        for t in (self.ai, self.ao, self.do):
            try:
                if t is not None:
                    t.close()
            except Exception:
                pass
        self.ai = self.ao = self.do = None
        self._set_trigger_idle()   # leave the trigger line in its idle state

    def _set_trigger_idle(self) -> None:
        """Static (on-demand) write of the trigger line's idle level. Done before the run too, so
        that an active-low pulse at sample 0 still produces a falling edge."""
        if self.trig is None or self.trig.frame_sync:
            return
        idle_high = self.trig.polarity.startswith("Active low")
        try:
            import nidaqmx
            with nidaqmx.Task() as z:
                if self.trig.kind == "DO":
                    z.do_channels.add_do_chan(self.trig.physical)
                    z.write(idle_high, auto_start=True)
                else:
                    z.ao_channels.add_ao_voltage_chan(self.trig.physical)
                    z.write(self.trig.level_v if idle_high else 0.0, auto_start=True)
        except Exception:
            pass
        # leave the shaker drive at 0 V
        try:
            import nidaqmx
            with nidaqmx.Task() as z:
                z.ao_channels.add_ao_voltage_chan(self.cfg.output.physical)
                z.write(0.0, auto_start=True)
        except Exception:
            pass


# ----------------------------------------------------------------------------- simulator
class SimBackend:
    """Shaker (V → N, with a high-pass) driving a 3-mode structure; per-channel sensor noise.
    The model is discrete-time and stateful across blocks, so transients, dwell convergence and
    H1/H2/Hv differences all behave like a real test. `true_frf()` gives the exact answer."""

    def __init__(self, cfg: AppConfig, realtime: bool = True, noise: float = 0.01, seed: int = 1):
        from scipy import signal
        self._signal = signal
        self.cfg = cfg
        self.channels = cfg.enabled_channels()
        self.n = cfg.output.block_size
        self.realtime = realtime
        self.noise = noise
        self.noise_floor = 1e-5          # absolute sensor noise floor (EU)
        self.distortion = 0.01           # shaker nonlinearity (harmonics grow with drive level)
        self.rng = np.random.default_rng(seed)
        self.queue: deque = deque()
        self.trig_out: deque = deque()          # simulated trigger line, consumed with the drive
        self.trigger_edges: List[int] = []      # absolute sample index of every active edge output
        self._samples_out = 0
        self._trig_prev = None
        self.max_queue = 2 * max(2, cfg.output.ao_queue_blocks)

    def configure(self) -> Tuple[float, List[str]]:
        sig = self._signal
        o = self.cfg.output
        fs = o.fs
        self.fs = fs
        lo, hi = max(o.fmin, fs / 200), min(o.fmax, 0.4 * fs)
        self.fn = lo + (hi - lo) * np.array([0.18, 0.45, 0.74])   # natural freqs inside band
        zeta = np.array([0.02, 0.015, 0.01])
        self.ref_i = self.cfg.reference_index()

        # shaker: force [N] = 20 N/V * 2nd-order high-pass (fc = 0.5 * f1)
        self.sh_b, self.sh_a = sig.butter(2, 0.5 * self.fn[0] / (fs / 2), "high")
        self.sh_b = self.sh_b * 20.0
        self.sh_zi = np.zeros(len(self.sh_a) - 1)

        # structure: acceleration per unit force, modal superposition, bilinear-discretised
        ref_node = self.channels[self.ref_i].node
        self.modes = []   # list of (b, a) per mode
        for fn, z in zip(self.fn, zeta):
            w = 2 * np.pi * fn
            b, a = sig.bilinear([1.0, 0.0, 0.0], [1.0, 2 * z * w, w * w], fs)
            self.modes.append((b, a))
        self.gain = np.zeros((len(self.channels), len(self.modes)))
        for i, c in enumerate(self.channels):
            if i == self.ref_i or c.sensor_type == "Voltage":
                continue
            for r in range(len(self.modes)):
                shape = lambda node: np.sin((r + 1) * np.pi * ((node % 10) + 0.5) / 10.5)
                g = shape(c.node) * shape(ref_node) / 5.0                    # modal mass 5 kg
                if c.units == "g":
                    g /= 9.80665
                elif c.sensor_type.startswith("Strain"):
                    g *= 2e-6
                elif c.sensor_type.startswith("Microphone"):
                    g *= 0.5
                self.gain[i, r] = g
        self.zi = [[np.zeros(2) for _ in self.modes] for _ in self.channels]
        notes = [f"SIMULATION: modes at {', '.join(f'{f:.1f}' for f in self.fn)} Hz"]
        # Phase-quadrature suite: give mode 1 a softening (amplitude-dependent) stiffness, so the
        # simulated resonance moves with the drive level like a nonlinear structure
        self.nl = self.cfg.suite == "Phase Quadrature"
        self.nl_kappa, self.nl_amp, self.zeta = 0.06, 0.0, zeta
        if self.nl:
            notes.append(f"SIMULATION: mode 1 softens with amplitude (≈ −{100 * self.nl_kappa:.0f} % frequency per "
                         "volt of resonant drive): the resonance moves with the drive level")
        t = self.cfg.trigger
        if t.enabled and t.frame_sync:
            notes.append(f"Frame sync (simulated): AI sample clock exported to {t.physical}, "
                         f"{fs:g} pulses/s from sample 0.")
        elif t.enabled:
            notes.append(f"Camera trigger on {t.physical} ({t.kind}), {t.pulse_samples}-sample pulse (simulated).")
        return fs, notes

    def true_frf(self, freqs: np.ndarray) -> np.ndarray:
        """Exact FRF response/force of the simulated structure, (n_channels, nf)."""
        w = 2 * np.pi * freqs / self.fs
        H = np.zeros((len(self.channels), freqs.size), dtype=complex)
        for r, (b, a) in enumerate(self.modes):
            _, h = self._signal.freqz(b, a, worN=w)
            H += self.gain[:, r:r + 1] * h[None, :]
        return H

    def write_block(self, x: np.ndarray, trig: np.ndarray = None) -> None:
        if len(self.queue) >= self.max_queue:
            raise RuntimeError("Simulated AO buffer full")
        self.queue.append(np.array(x, dtype=float))
        self.trig_out.append(None if trig is None else np.array(trig, dtype=float))

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.k = 0

    def read_block(self) -> np.ndarray:
        if not self.queue:
            raise RuntimeError("Simulated AO underflow: output buffer ran dry")
        v = self.queue.popleft()
        tr = self.trig_out.popleft()
        if tr is not None:   # record pulse edges, as a camera would see them
            t = self.cfg.trigger
            active = tr < 0.5 * (t.level_v if t.kind == "AO" else 1.0)
            if not t.polarity.startswith("Active low"):
                active = ~active
            prev = self._trig_prev if self._trig_prev is not None else False
            starts = np.flatnonzero(active & ~np.concatenate([[prev], active[:-1]]))
            self.trigger_edges += [int(self._samples_out + i) for i in starts]
            self._trig_prev = bool(active[-1])
        self._samples_out += v.size
        if self.realtime:   # pace like hardware
            due = self.t0 + (self.k + 1) * self.n / self.fs
            time.sleep(max(0.0, due - time.perf_counter()))
        self.k += 1
        force, self.sh_zi = self._signal.lfilter(self.sh_b, self.sh_a, v, zi=self.sh_zi)
        # mild level-dependent shaker/amplifier distortion (≈1 % 2nd harmonic at 20 N, i.e. 1 V)
        force = force + self.distortion * force * np.abs(force) / 20.0 + 0.3 * self.distortion * force ** 3 / 400.0
        out = np.zeros((len(self.channels), self.n))
        if self.nl:                                  # re-tune mode 1 to the last block's modal amplitude
            fn = self.fn[0] * (1 - self.nl_kappa * min(self.nl_amp, 3.0))
            w = 2 * np.pi * fn
            self.modes[0] = self._signal.bilinear([1.0, 0.0, 0.0], [1.0, 2 * self.zeta[0] * w, w * w], self.fs)
        m1_peak = 0.0
        for i, c in enumerate(self.channels):
            if i == self.ref_i:
                out[i] = force
                continue
            for r, (b, a) in enumerate(self.modes):
                if self.gain[i, r] != 0:
                    y, self.zi[i][r] = self._signal.lfilter(b, a, force, zi=self.zi[i][r])
                    out[i] += self.gain[i, r] * y
                    if r == 0:
                        m1_peak = max(m1_peak, float(np.max(np.abs(y))))
        if self.nl:
            self.nl_amp = m1_peak / 500.0              # ≈ volts of drive at resonance (20 N/V, Q = 25)
        # sensor noise relative to each channel's typical level (noise on input AND output)
        for i, c in enumerate(self.channels):
            level = np.sqrt(np.mean(out[i] ** 2)) + 1e-12
            out[i] += self.noise * level * self.rng.standard_normal(self.n) + self.noise_floor * self.rng.standard_normal(self.n)
            np.clip(out[i], -c.max_range, c.max_range, out=out[i])      # ADC saturation
        return out

    def backlog(self) -> int:
        return 0

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_backend(cfg: AppConfig, simulate: bool, realtime: bool = True):
    return SimBackend(cfg, realtime=realtime) if simulate else NIDAQBackend(cfg)


# ----------------------------------------------------------------------------- rate check
def candidate_rates(fs: float) -> List[float]:
    """Rates delta-sigma C Series modules typically support (12.8 / 13.1072 MHz timebases ÷ 256 ÷ n),
    near the requested one."""
    base = {51200.0 / n for n in range(1, 64)} | {50000.0 / n for n in range(1, 64)}
    lo, hi = fs / 16.0, fs * 4.0
    return sorted(r for r in base if lo <= r <= hi)


def check_rates(cfg: AppConfig, scan: bool = True) -> Dict:
    """Ask DAQmx which sample rate it would actually use for this setup, without starting anything.

    Builds the input task with every enabled channel and the output task with the drive channel,
    verifies them at the requested rate and reads back the coerced rates. With scan=True it also
    tries rates the modules typically support and lists those that both tasks accept *exactly*.
    Returns {'requested', 'ai', 'ao', 'exact': [...], 'notes': [...], 'error': ''}."""
    import nidaqmx
    import nidaqmx.constants as K
    o = cfg.output
    out = {"requested": float(o.fs), "ai": None, "ao": None, "exact": [], "notes": [], "error": ""}
    be = NIDAQBackend(cfg)
    ai = ao = None
    try:
        ai = be.ai = nidaqmx.Task("ModalRateCheckAI")
        for c in be.channels:
            be._add_channel(c)
        ao = nidaqmx.Task("ModalRateCheckAO")
        rng = ao_common_range(o.physical, o.max_voltage)
        ao.ao_channels.add_ao_voltage_chan(o.physical, "Drive", min_val=-rng, max_val=rng)
        use_ai_clk = o.clock_mode.startswith("AO clocked")
        chassis = chassis_of(o.physical)

        def rates_at(fs):
            ai.timing.cfg_samp_clk_timing(fs, sample_mode=K.AcquisitionType.CONTINUOUS, samps_per_chan=max(1000, o.block_size * 4))
            ai.control(K.TaskMode.TASK_VERIFY)
            f_ai = float(ai.timing.samp_clk_rate)
            src = f"/{chassis}/ai/SampleClock" if use_ai_clk else ""
            ao.timing.cfg_samp_clk_timing(f_ai, source=src, sample_mode=K.AcquisitionType.CONTINUOUS,
                                          samps_per_chan=max(1000, o.block_size * 4))
            ao.control(K.TaskMode.TASK_VERIFY)
            f_ao = f_ai if use_ai_clk else float(ao.timing.samp_clk_rate)
            return f_ai, f_ao

        try:
            out["ai"], out["ao"] = rates_at(o.fs)
        except Exception as e:                       # the requested rate is refused: still list what works
            out["error"] = f"{type(e).__name__}: {e}"
        if scan:
            exact = []
            for r in candidate_rates(o.fs):
                try:
                    a, b = rates_at(r)
                except Exception:
                    continue
                if abs(a - r) <= 1e-6 * r and abs(b - r) <= 1e-6 * r:
                    exact.append(r)
            # the dozen exact rates closest to the request (in log terms), in ascending order
            exact = sorted(exact, key=lambda r: abs(np.log(r / o.fs)))[:12]
            out["exact"] = sorted(exact)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        for t in (ai, ao):
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
    return out

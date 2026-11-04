# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Configuration data classes (channel / output / save) with JSON load & save."""

from __future__ import annotations

import json

import numpy as np
from dataclasses import asdict, dataclass, field, fields
from typing import List

# ----------------------------------------------------------------------------- constants
SENSOR_TYPES = [
    "Accelerometer (IEPE)",
    "Force (IEPE)",
    "Strain gauge (bridge)",
    "Microphone (IEPE)",
    "Pressure (IEPE)",
    "Voltage",
    "Custom",
]
CUSTOM_TYPE = "Custom"         # any other sensor: name, units and sensitivity (mV/unit) typed by the user
CUSTOM_UNITS = ["EU", "mm", "µm", "m", "mm/s", "m/s", "m/s²", "kPa", "N·m", "°C", "V"]

# engineering-unit choices per sensor type (first = default)
UNITS_BY_TYPE = {
    "Accelerometer (IEPE)": ["g", "m/s²"],
    "Force (IEPE)": ["N", "lbf"],
    "Strain gauge (bridge)": ["strain"],
    "Microphone (IEPE)": ["Pa"],
    "Pressure (IEPE)": ["kPa", "Pa", "bar", "psi", "MPa"],
    "Voltage": ["V"],          # editable: any EU label, scaled by sensitivity mV/EU
    "Custom": CUSTOM_UNITS,    # editable: any unit label, scaled by sensitivity mV/unit
}


def voltage_scaled(sensor_type: str) -> bool:
    """Types read as a (optionally IEPE-powered) voltage and scaled in software by mV/unit."""
    return sensor_type in ("Voltage", CUSTOM_TYPE) or sensor_type.startswith("Pressure")

DIRECTIONS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
DIRECTION_CODE = {"+X": 1, "-X": -1, "+Y": 2, "-Y": -2, "+Z": 3, "-Z": -3}

BRIDGE_TYPES = [
    "Full bridge I", "Full bridge II", "Full bridge III",
    "Half bridge I", "Half bridge II", "Quarter bridge I", "Quarter bridge II",
]

EXCITATION_TYPES = [
    "Uniform random",                    # white, uniformly distributed samples, 0 … fs/2
    "Multisine random",                  # flat amplitude in band, random phases
    "Gaussian random (band-limited)",    # Rayleigh amplitudes + random phases in band (true random)
    "Periodic chirp",                    # flat amplitude in band, Schroeder phases: lowest crest factor
    "Burst random",                      # band-limited random for part of the block, then silence
    "Custom spectrum",                   # user table of frequency / amplitude (or PSD), random phases
]
BAND_TYPES = ("Multisine random", "Gaussian random (band-limited)", "Periodic chirp", "Burst random")
CUSTOM_KINDS = ["Amplitude, linear interpolation", "PSD, log-log interpolation"]
WINDOWS = ["Rectangular", "Hann", "Hamming", "Blackman", "Blackman-Harris", "Flat top",
           "Tukey (10 %)", "Kaiser-Bessel (β = 6)"]
CLOCK_MODES = ["Shared start trigger", "AO clocked by AI sample clock"]

TRIGGER_MODES = [
    "Once, when the input task starts",
    "Start of the converged (last) block of each realization",
    "Start of the dwell blocks of each realization",
    "Frame sync: one pulse per sample (AI sample clock → PFI)",
]
FRAME_SYNC = TRIGGER_MODES[3]
TRIGGER_POLARITY = ["Active high (rising edge)", "Active low (falling edge)"]


def output_kind(physical: str) -> str:
    """'AO' for analog-output channels, 'DO' for digital lines, 'PFI' for terminals, '' if unknown."""
    p = physical.lower()
    if "pfi" in p:
        return "PFI"
    if "/ao" in p:
        return "AO"
    if "/port" in p or "/line" in p:
        return "DO"
    return ""


def sensitivity_units(sensor_type: str, eu: str) -> str:
    if sensor_type.startswith("Accelerometer"):
        return "mV/g"
    if sensor_type.startswith("Force"):
        return "mV/lbf" if eu == "lbf" else "mV/N"
    if sensor_type.startswith("Microphone"):
        return "mV/Pa"
    if sensor_type.startswith("Strain"):
        return "—"
    return f"mV/{eu or 'EU'}"


TYPE_DEFAULTS = {
    #                        sens   range  iepe  coupling
    "Accelerometer (IEPE)": (100.0, 50.0, 2.0, "AC"),
    "Force (IEPE)": (10.0, 200.0, 2.0, "AC"),
    "Strain gauge (bridge)": (0.0, 0.001, 0.0, "DC"),
    "Microphone (IEPE)": (50.0, 100.0, 2.0, "AC"),
    "Pressure (IEPE)": (10.0, 100.0, 2.0, "AC"),
    "Voltage": (1000.0, 5.0, 0.0, "Default"),
    "Custom": (100.0, 10.0, 0.0, "Default"),
}


# ----------------------------------------------------------------------------- dataclasses
@dataclass
class ChannelConfig:
    enabled: bool = True
    physical: str = "cDAQ1Mod1/ai0"
    name: str = "Ch0"
    node: int = 1
    direction: str = "+Z"
    sensor_type: str = "Accelerometer (IEPE)"
    sensitivity: float = 100.0         # mV per (g | N | lbf | Pa | kPa | EU); unused for strain
    units: str = "g"
    max_range: float = 50.0            # ± input range in engineering units
    iepe_ma: float = 2.0               # IEPE current in mA, 0 = off
    coupling: str = "AC"               # Default | AC | DC
    is_reference: bool = False
    # strain-gauge only
    bridge: str = "Quarter bridge I"
    gage_factor: float = 2.0
    excit_v: float = 2.5
    nominal_r: float = 350.0
    poisson: float = 0.3
    custom_type: str = ""              # sensor_type "Custom": the user's name for the sensor type

    @property
    def type_label(self) -> str:
        """Sensor type as shown to people: the custom name for custom sensors."""
        return (self.custom_type or "Custom") if self.sensor_type == CUSTOM_TYPE else self.sensor_type

    @property
    def sens_units(self) -> str:
        return sensitivity_units(self.sensor_type, self.units)


@dataclass
class OutputConfig:
    physical: str = "cDAQ1Mod2/ao0"
    max_voltage: float = 2.0           # safety limit on |drive| [V]
    clock_mode: str = CLOCK_MODES[0]
    excitation: str = EXCITATION_TYPES[1]
    rms: float = 0.2                   # V RMS (burst random: RMS while the burst is on)
    fmin: float = 5.0                  # Hz, band of interest (excited band for multisine)
    fmax: float = 500.0
    fs: float = 2048.0                 # Hz
    block_size: int = 2048             # samples per block
    dwell_blocks: int = 4              # blocks per realization (same random block repeated)
    realizations: int = 20             # number of distinct random blocks (= averages)
    decay_time: float = 5.0            # s of zero output before excitation (IEPE settling)
    fade: bool = True                  # one fade-in block before / fade-out block after
    window: str = "Rectangular"
    ao_queue_blocks: int = 3           # AO prefill depth (latency vs robustness)
    seed: int = -1                     # -1 = new random seed each run
    burst_fraction: float = 0.5        # burst random: fraction of each block that is driven
    custom_kind: str = CUSTOM_KINDS[0]
    custom_points: List[List[float]] = field(default_factory=lambda: [[5.0, 1.0], [100.0, 1.0],
                                                                       [300.0, 0.5], [500.0, 0.5]])

    @property
    def df(self) -> float:
        return self.fs / self.block_size


@dataclass
class TriggerConfig:
    enabled: bool = False
    physical: str = "cDAQ1Mod4/port0/line0"   # AO channel or DO line; a PFI terminal for frame sync
    mode: str = TRIGGER_MODES[0]
    pulse_samples: int = 10                   # pulse width in samples
    level_v: float = 3.3                      # active level for an AO trigger (≥2 V reads as TTL high)
    polarity: str = TRIGGER_POLARITY[0]

    @property
    def kind(self) -> str:
        return output_kind(self.physical)

    @property
    def frame_sync(self) -> bool:
        return self.mode == FRAME_SYNC


SUITES = ["Random excitation", "Stepped sine", "Harmonic", "Phase Quadrature"]
PQ_CONTROL_MODES = ["Phase quadrature (secant on frequency)",
                    "Reference-free: maximise |H1| (secant / parabolic search)"]
PQ_LEVEL_MODES = ["Number of levels", "Step (V)"]
PQ_ORDERS = ["Ascending", "Descending", "Up, then down"]
PQ_PHASE_PRESETS = [("Acceleration / force: +90°", 90.0), ("Velocity / force: 0°", 0.0),
                    ("Displacement / force: −90°", -90.0), ("Custom", None)]
PQ_FAIL_ACTIONS = [
    "Ask me (hold drive at a safe level)",
    "Accept best and continue (flagged)",
    "Skip this level",
    "Stop the test",
]
PQ_TRIGGER_LABELS = [
    "Once, when the input task starts",
    "Start of the converged block of every level",
    "Start of every drive level",
]
HARM_CONTROL_MODES = ["Quasi-Newton control (Broyden, closed loop)", "No control (fixed drive harmonics)"]
HARM_FAIL_ACTIONS = [
    "Ask me (hold drive at a safe level)",
    "Accept the best iteration and run the test (flagged)",
    "Stop the test",
]
HARM_SAVE_MODES = ["All blocks", "Converged (test) blocks only", "One converged block every S seconds"]
HARMONIC_TRIGGER_LABELS = [
    "Once, when the input task starts",
    "Start of the converged test blocks",
    "Start of every control iteration",
]
SPACINGS = ["Linear (step in Hz)", "Logarithmic (points per decade)"]
CONTROL_MODES = ["Secant control (closed loop)", "No control (constant drive)"]
TARGET_KINDS = ["RMS", "Amplitude (RMS·√2)"]
FAIL_ACTIONS = [
    "Ask me (hold drive at a safe level)",
    "Accept best and continue (flagged)",
    "Skip this frequency",
    "Stop the test",
]


@dataclass
class SteppedConfig:
    f_start: float = 20.0
    f_stop: float = 400.0
    f_step: float = 10.0               # Hz (linear) or points per decade (log)
    spacing: str = SPACINGS[0]
    settle_blocks: int = 1             # blocks discarded after every drive/frequency change
    dwell_time: float = 1.0            # s of steady state analysed at each frequency
    control: str = CONTROL_MODES[0]
    drive_amplitude: float = 0.2       # V peak, no-control mode
    target_channel: str = "Acc_1"
    target_value: float = 0.5          # in the target channel's units, as RMS or as amplitude
    target_kind: str = "RMS"           # "RMS" or "Amplitude (RMS·√2)": equivalent single-harmonic amplitude
    initial_drive: float = 0.05        # V peak for the first frequency
    tolerance_pct: float = 3.0         # converged when |error| <= tolerance
    max_iterations: int = 8
    on_failure: str = FAIL_ACTIONS[0]
    overload_pct: float = 95.0         # input overload when |peak| >= this % of the channel range
    n_harmonics: int = 5               # harmonic orders 1…n analysed, displayed and saved

    @property
    def is_amplitude(self) -> bool:
        return self.target_kind.startswith("Amplitude")

    @property
    def target_scale(self) -> float:
        """Controlled quantity = block RMS × this (1 for RMS, √2 for the equivalent amplitude)."""
        return float(np.sqrt(2.0)) if self.is_amplitude else 1.0

    @property
    def target_label(self) -> str:
        return "amplitude" if self.is_amplitude else "RMS"

    @property
    def target_peak(self) -> float:
        """Peak of a pure sine meeting the target."""
        return self.target_value if self.is_amplitude else self.target_value * float(np.sqrt(2.0))


@dataclass
class HarmonicConfig:
    """Harmonic suite: v(t) = Σ a_i·cos(n_i·Ω·t + φ_i), Ω = 2π·f0.
    No control: (n_i, a_i [V], φ_i [°]) is the drive. Control: (n_i, a_i, φ_i) is the target for
    harmonic n_i of the control channel (a_i in its units, peak); the drive harmonics are found by
    Broyden's quasi-Newton method."""
    f0: float = 20.0                   # base frequency (snapped to an FFT line k·fs/N)
    harmonics: List[List[float]] = field(default_factory=lambda: [[1, 0.5, 0.0], [2, 0.0, 0.0], [3, 0.0, 0.0]])
    control: str = HARM_CONTROL_MODES[0]
    target_channel: str = "Acc_1"
    initial_drive: float = 0.05        # V peak of the largest drive harmonic at the first iteration
    tolerance_pct: float = 3.0         # complex error |Y_i − T_i| relative to |T_i| (or to max|T| for small targets)
    max_iterations: int = 15
    on_failure: str = HARM_FAIL_ACTIONS[0]
    overload_pct: float = 95.0
    settle_blocks: int = 1             # blocks discarded after every drive change
    test_blocks: int = 10              # converged blocks run before the excitation is switched off
    n_harmonics: int = 5               # orders 1…n analysed and fitted (at least the highest listed order)

    @property
    def closed_loop(self) -> bool:
        return self.control == HARM_CONTROL_MODES[0]

    def rows(self):
        """[(order:int, amplitude:float, phase_deg:float)] with valid orders only."""
        out = []
        for r in self.harmonics:
            try:
                n, a, p = int(round(float(r[0]))), float(r[1]), float(r[2])
            except (TypeError, ValueError, IndexError):
                continue
            out.append((n, a, p))
        return out

    def complex_values(self):
        """(orders, complex a·e^{jφ})."""
        rows = self.rows()
        orders = np.array([r[0] for r in rows], dtype=int)
        vals = np.array([r[1] * np.exp(1j * np.radians(r[2])) for r in rows], dtype=complex)
        return orders, vals

    def analysis_orders(self) -> int:
        rows = self.rows()
        return max([self.n_harmonics] + [r[0] for r in rows])


@dataclass
class PQConfig:
    """Phase-quadrature / resonance-tracking suite: a sine of fixed drive amplitude per level whose
    frequency is iterated until the response is in quadrature with the reference (or until |H1|
    is maximal, reference-free), level after level."""
    v_min: float = 0.05                # V peak
    v_max: float = 0.5
    level_mode: str = PQ_LEVEL_MODES[0]
    n_levels: int = 5
    v_step: float = 0.1
    order: str = PQ_ORDERS[0]
    f_init: float = 100.0              # initial guess of the resonance frequency (Hz)
    f_min: float = 10.0                # search range (control fails outside it)
    f_max: float = 1000.0
    control: str = PQ_CONTROL_MODES[0]
    response_channel: str = "Acc_1"
    target_phase: float = 90.0         # phase of response re reference at quadrature (°)
    phase_preset: str = PQ_PHASE_PRESETS[0][0]
    phase_tol: float = 1.0             # ± degrees
    freq_tol: float = 0.05             # Hz (reference-free)
    step0: float = 1.0                 # first frequency step (Hz)
    max_step: float = 10.0             # largest frequency change per iteration (Hz)
    max_iterations: int = 20
    on_failure: str = PQ_FAIL_ACTIONS[0]
    overload_pct: float = 95.0
    settle_blocks: int = 2
    dwell_blocks: int = 1              # blocks recorded (and saved) at the converged frequency of every level
    n_harmonics: int = 5

    @property
    def quadrature(self) -> bool:
        return self.control == PQ_CONTROL_MODES[0]

    def levels(self) -> np.ndarray:
        """Drive amplitudes [V peak] in run order."""
        if self.v_max < self.v_min or self.v_min <= 0:
            return np.array([])
        if self.level_mode == PQ_LEVEL_MODES[0]:
            v = np.linspace(self.v_min, self.v_max, max(1, self.n_levels)) if self.n_levels > 1 else np.array([self.v_min])
        else:
            if self.v_step <= 0:
                return np.array([])
            v = np.arange(self.v_min, self.v_max + 1e-9 * max(1.0, self.v_max), self.v_step)
        if self.order == PQ_ORDERS[1]:
            v = v[::-1]
        elif self.order == PQ_ORDERS[2]:
            v = np.concatenate([v, v[-2::-1]])
        return v


def harmonic_line(h: "HarmonicConfig", o: "OutputConfig") -> int:
    """FFT line of the base frequency: every block then holds whole periods of every harmonic."""
    if o.fs <= 0 or o.block_size < 16 or h.f0 <= 0:
        return 0
    return int(max(1, round(h.f0 * o.block_size / o.fs)))


def harmonic_waveform(orders, vals, k: int, n: int) -> np.ndarray:
    """One block of Σ |V_i|·cos(2π·n_i·k·m/N + ∠V_i), m = 0…N−1."""
    m = np.arange(n)
    x = np.zeros(n)
    for o_, v in zip(orders, vals):
        x += np.real(v * np.exp(2j * np.pi * o_ * k * m / n))
    return x


@dataclass
class SaveConfig:
    folder: str = ""
    basename: str = "modal_test"
    save_uff: bool = True
    uff_h1: bool = True
    uff_h2: bool = True
    uff_hv: bool = True
    uff_coherence: bool = True
    save_converged_tdms: bool = True
    save_each_realization: bool = False
    save_dwell_tdms: bool = True       # stepped sine: all dwell blocks of every frequency → TDMS
    save_harmonics: bool = True        # stepped sine: harmonics CSV + harmonic-FRF UFFs (orders: stepped.n_harmonics)
    harm_save_mode: str = HARM_SAVE_MODES[1]   # harmonic suite: which blocks go to TDMS
    harm_save_every_s: float = 1.0             # harmonic suite: spacing for "one converged block every S s"
    harm_save_tdms: bool = True                # harmonic suite: save time data at all
    pq_save_blocks: bool = True                # phase quadrature: converged block of every level → TDMS
    pq_save_uff: bool = True                   # phase quadrature: H1 at resonance vs frequency → UFF 58


@dataclass
class AppConfig:
    channels: List[ChannelConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)
    save: SaveConfig = field(default_factory=SaveConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    null_bridges: bool = True
    suite: str = SUITES[0]
    stepped: SteppedConfig = field(default_factory=SteppedConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    pq: PQConfig = field(default_factory=PQConfig)

    # ---------------------------------------------------------------- helpers
    def enabled_channels(self) -> List[ChannelConfig]:
        return [c for c in self.channels if c.enabled]

    def reference_index(self) -> int:
        """Index (within enabled channels) of the reference channel, or -1."""
        for i, c in enumerate(self.enabled_channels()):
            if c.is_reference:
                return i
        return -1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AppConfig":
        d = json.loads(text)

        def build(dc, data):
            names = {f.name for f in fields(dc)}
            return dc(**{k: v for k, v in (data or {}).items() if k in names})

        return cls(
            channels=[build(ChannelConfig, c) for c in d.get("channels", [])],
            output=build(OutputConfig, d.get("output")),
            save=build(SaveConfig, d.get("save")),
            trigger=build(TriggerConfig, d.get("trigger")),
            suite=d.get("suite", SUITES[0]),
            stepped=build(SteppedConfig, _migrate_stepped(d.get("stepped"), d.get("save"))),
            harmonic=build(HarmonicConfig, d.get("harmonic")),
            pq=build(PQConfig, d.get("pq")),
            null_bridges=d.get("null_bridges", True),
        )

    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty = OK)."""
        errs = []
        ch = self.enabled_channels()
        o = self.output
        if not ch:
            errs.append("No input channels are enabled.")
        refs = [c for c in ch if c.is_reference]
        if self.suite == SUITES[3]:
            if len(refs) > 1:
                errs.append("At most one channel can be marked as the Reference.")
        elif self.suite == SUITES[2]:
            if len(refs) > 1:
                errs.append("At most one channel can be marked as the Reference.")
        else:
            if len(refs) != 1:
                errs.append("Exactly one enabled channel must be marked as the Reference (the drive-point force).")
            if len(ch) - len(refs) < 1:
                errs.append("At least one response channel (non-reference) is needed.")
        phys = [c.physical.strip() for c in ch]
        dup = {p for p in phys if phys.count(p) > 1}
        if dup:
            errs.append(f"Physical channel used more than once: {', '.join(sorted(dup))}")
        if any(not p for p in phys):
            errs.append("A channel has no physical channel name.")
        for c in ch:
            if c.sensor_type != "Strain gauge (bridge)" and c.sensitivity <= 0:
                errs.append(f"{c.name}: sensitivity must be > 0.")
            if c.sensor_type == CUSTOM_TYPE and not c.custom_type.strip():
                errs.append(f"{c.name}: enter a name for the custom sensor type.")
            if c.max_range <= 0:
                errs.append(f"{c.name}: input range must be > 0.")
        if o.block_size < 16 or o.block_size % 2:
            errs.append("Block size must be an even number ≥ 16.")
        if o.fs <= 0:
            errs.append("Sampling rate must be > 0.")
        if self.suite == SUITES[2]:
            return errs + self.harmonic_problems() + self._common_problems()
        if self.suite == SUITES[3]:
            return errs + self.pq_problems() + self._common_problems()
        if self.suite == SUITES[0] and (o.dwell_blocks < 1 or o.realizations < 1):
            errs.append("Dwell blocks and realizations must both be ≥ 1.")
        if not (0 <= o.fmin < o.fmax):
            errs.append("Frequency band: need 0 ≤ f_min < f_max.")
        if o.fmax > o.fs / 2:
            errs.append(f"f_max ({o.fmax:g} Hz) is above Nyquist (fs/2 = {o.fs / 2:g} Hz).")
        if o.excitation in BAND_TYPES and int(o.fmax // o.df) < max(1, int(-(-o.fmin // o.df))):
            errs.append("The excitation band contains no frequency lines: widen the band or increase block size.")
        if o.excitation == "Burst random" and not 0.05 <= o.burst_fraction <= 1.0:
            errs.append("Burst length must be between 5 % and 100 % of the block.")
        if o.excitation == "Custom spectrum":
            errs += custom_points_problems(o.custom_points, o.fs)
        if self.suite == SUITES[1]:
            errs += self.stepped_problems()
            return errs + self._common_problems()
        if o.rms <= 0:
            errs.append("Excitation RMS must be > 0.")
        return errs + self._common_problems()

    def _common_problems(self) -> List[str]:
        errs: List[str] = []
        o = self.output
        if o.max_voltage <= 0:
            errs.append("Output voltage limit must be > 0.")
        if not o.physical.strip():
            errs.append("No output (AO) channel set.")
        t = self.trigger
        if t.enabled and t.frame_sync:
            if t.kind != "PFI":
                errs.append(f"Frame sync needs a PFI terminal (e.g. /cDAQ1Mod4/PFI0), not '{t.physical}'.")
        elif t.enabled:
            if t.kind == "PFI":
                errs.append("A PFI terminal can only be used in frame-sync mode. For single pulses choose an "
                            "AO channel or a DO line (…/port0/lineN).")
            elif not t.kind:
                errs.append(f"Trigger channel '{t.physical}' is not recognised as an analog output "
                            "(…/aoN) or a digital line (…/portN/lineM).")
            if t.physical.strip() == o.physical.strip():
                errs.append("The trigger channel must be different from the shaker drive channel.")
            if not 1 <= t.pulse_samples < o.block_size:
                errs.append("Trigger pulse width must be at least 1 sample and shorter than one block.")
            if t.kind == "AO" and not 0 < t.level_v <= 10:
                errs.append("Analog trigger level must be between 0 and 10 V.")
        s = self.save
        if wants_folder(self) and not s.folder.strip():
            errs.append("Choose an output folder on the Measurement tab (or untick all save options).")
        return errs

    # ---------------------------------------------------------------- phase quadrature
    def pq_problems(self) -> List[str]:
        p, o = self.pq, self.output
        errs = []
        if o.fs <= 0 or o.block_size < 16:
            return errs
        lv = p.levels()
        if lv.size == 0:
            errs.append("Phase quadrature: need 0 < min ≤ max drive level and a valid number of levels / step.")
        elif lv.max() > o.max_voltage:
            errs.append(f"Phase quadrature: the highest drive level {lv.max():g} V is above the {o.max_voltage:g} V limit.")
        if not 0 < p.f_min < p.f_max:
            errs.append("Phase quadrature: need 0 < search-range minimum < maximum.")
        if p.f_max >= o.fs / 2:
            errs.append(f"Phase quadrature: the search range must stay below Nyquist ({o.fs / 2:g} Hz).")
        if not p.f_min <= p.f_init <= p.f_max:
            errs.append("Phase quadrature: the initial frequency must lie inside the search range.")
        names = {c.name: c for c in self.enabled_channels()}
        if p.response_channel not in names:
            errs.append(f"Phase quadrature: response channel '{p.response_channel}' is not an enabled channel.")
        ref = self.reference_index()
        if p.quadrature and ref < 0:
            errs.append("Phase quadrature needs a reference channel (tick Reference on the Channel Setup tab), "
                        "or use the reference-free mode.")
        if ref >= 0 and self.enabled_channels()[ref].name == p.response_channel:
            errs.append("Phase quadrature: the response channel must differ from the reference channel.")
        if p.step0 <= 0 or p.max_step <= 0:
            errs.append("Phase quadrature: frequency steps must be > 0.")
        if p.quadrature and not 0.01 <= p.phase_tol <= 45:
            errs.append("Phase quadrature: phase tolerance must be between 0.01 and 45°.")
        if not p.quadrature and p.freq_tol <= 0:
            errs.append("Phase quadrature: frequency tolerance must be > 0.")
        if p.max_iterations < 2:
            errs.append("Phase quadrature: at least 2 iterations are needed.")
        if p.dwell_blocks < 1:
            errs.append("Phase quadrature: record at least one converged block per level.")
        return errs

    # ---------------------------------------------------------------- harmonic
    def harmonic_problems(self) -> List[str]:
        h, o = self.harmonic, self.output
        errs = []
        if o.fs <= 0 or o.block_size < 16:
            return errs
        rows = h.rows()
        if len(rows) != len(h.harmonics):
            errs.append("Harmonic: every row needs a numeric order, amplitude and phase.")
        if not rows:
            errs.append("Harmonic: enter at least one harmonic (order, amplitude, phase).")
            return errs
        orders = [r[0] for r in rows]
        if any(n < 1 for n in orders):
            errs.append("Harmonic: orders must be whole numbers ≥ 1.")
        if len(set(orders)) != len(orders):
            errs.append("Harmonic: each order may appear only once.")
        if any(r[1] < 0 for r in rows):
            errs.append("Harmonic: amplitudes must be ≥ 0 (use the phase for the sign).")
        if not any(r[1] > 0 for r in rows):
            errs.append("Harmonic: at least one amplitude must be > 0.")
        if h.f0 <= 0:
            errs.append("Harmonic: the base frequency must be > 0.")
            return errs
        k = harmonic_line(h, o)
        if max(orders) * k >= o.block_size // 2:
            errs.append(f"Harmonic: the highest order ({max(orders)}·f0 = {max(orders) * k * o.fs / o.block_size:g} Hz) "
                        f"must be below Nyquist ({o.fs / 2:g} Hz).")
        if h.test_blocks < 1:
            errs.append("Harmonic: run at least one converged test block.")
        if self.save.harm_save_mode == HARM_SAVE_MODES[2] and self.save.harm_save_every_s <= 0:
            errs.append("Harmonic: the save interval S must be > 0 s.")
        if not h.closed_loop:
            o_, v = h.complex_values()
            pk = np.max(np.abs(harmonic_waveform(o_, v, k, o.block_size))) if k else 0
            if pk > o.max_voltage:
                errs.append(f"Harmonic: the drive peaks at {pk:.3g} V, above the {o.max_voltage:g} V limit.")
        else:
            names = {c.name: c for c in self.enabled_channels()}
            c = names.get(h.target_channel)
            if c is None:
                errs.append(f"Harmonic: control channel '{h.target_channel}' is not an enabled channel.")
            else:
                o_, v = h.complex_values()
                pk = np.max(np.abs(harmonic_waveform(o_, v, k, o.block_size))) if k else 0
                if pk >= c.max_range * h.overload_pct / 100:
                    errs.append(f"Harmonic: the target waveform peaks at {pk:.3g} {c.units}, above the overload "
                                f"threshold of {c.name} (range ±{c.max_range:g}).")
            if not 0 < h.initial_drive <= o.max_voltage:
                errs.append(f"Harmonic: initial drive must be > 0 and ≤ the {o.max_voltage:g} V limit.")
            if not 0.1 <= h.tolerance_pct <= 50:
                errs.append("Harmonic: tolerance must be between 0.1 and 50 %.")
        return errs

    # ---------------------------------------------------------------- stepped sine
    def stepped_problems(self) -> List[str]:
        st, o = self.stepped, self.output
        errs = []
        if o.fs <= 0 or o.block_size < 16:
            return errs
        if not (0 < st.f_start < st.f_stop):
            errs.append("Stepped sine: need 0 < start frequency < stop frequency.")
        if st.f_stop >= o.fs / 2:
            errs.append(f"Stepped sine: stop frequency must be below Nyquist ({o.fs / 2:g} Hz).")
        if st.f_step <= 0:
            errs.append("Stepped sine: the step must be > 0.")
        if not errs and len(stepped_frequencies(st, o)[0]) < 1:
            errs.append("Stepped sine: no frequencies in the range (check block size / range).")
        if st.dwell_time <= 0:
            errs.append("Stepped sine: dwell time must be > 0.")
        if st.control == CONTROL_MODES[1]:
            if not 0 < st.drive_amplitude <= o.max_voltage:
                errs.append(f"Stepped sine: drive amplitude must be > 0 and ≤ the {o.max_voltage:g} V limit.")
        else:
            names = {c.name: c for c in self.enabled_channels()}
            c = names.get(st.target_channel)
            if c is None:
                errs.append(f"Stepped sine: control channel '{st.target_channel}' is not an enabled channel.")
            elif st.target_peak >= c.max_range * st.overload_pct / 100:
                errs.append(f"Stepped sine: target {st.target_value:g} {c.units} {st.target_label} ({st.target_peak:.3g} peak) "
                            f"is above the overload threshold of {c.name} (range ±{c.max_range:g}).")
            if st.target_value <= 0:
                errs.append("Stepped sine: the target must be > 0.")
            if not 0 < st.initial_drive <= o.max_voltage:
                errs.append(f"Stepped sine: initial drive must be > 0 and ≤ the {o.max_voltage:g} V limit.")
            if not 0.1 <= st.tolerance_pct <= 50:
                errs.append("Stepped sine: tolerance must be between 0.1 and 50 %.")
            if st.max_iterations < 1:
                errs.append("Stepped sine: at least 1 iteration is needed.")
        return errs



def wants_folder(cfg: "AppConfig") -> bool:
    """True if the run will write any file (and so needs a result folder)."""
    s = cfg.save
    if cfg.suite == SUITES[2]:
        return s.harm_save_tdms
    if cfg.suite == SUITES[3]:
        return True                      # the resonance table (CSV) is always written
    return (s.save_uff or s.save_converged_tdms or s.save_each_realization or
            (cfg.suite == SUITES[1] and (s.save_dwell_tdms or s.save_harmonics)))


def _migrate_stepped(d, save=None):
    """Older setups: the target was stored as 'target_rms'; the harmonic order was a save option."""
    d = dict(d or {})
    if "target_rms" in d and "target_value" not in d:
        d["target_value"] = d.pop("target_rms")
        d.setdefault("target_kind", "RMS")
    if "n_harmonics" not in d and save and "harmonics_order" in save:
        d["n_harmonics"] = save["harmonics_order"]
    return d


def stepped_frequencies(st: "SteppedConfig", o: "OutputConfig"):
    """Planned frequencies snapped to FFT lines k·fs/N (whole cycles per block, leakage-free).
    Returns (frequencies [Hz], line indices k, number of requested points merged by snapping)."""
    df = o.fs / o.block_size if o.block_size else 0
    if st.f_start <= 0 or st.f_stop <= st.f_start or st.f_step <= 0 or df <= 0:
        return np.array([]), np.array([], dtype=int), 0
    if st.spacing.startswith("Log"):
        n = max(2, int(np.floor(np.log10(st.f_stop / st.f_start) * st.f_step + 1e-9)) + 1)
        req = np.geomspace(st.f_start, st.f_stop, n)
    else:
        req = np.arange(st.f_start, st.f_stop + 1e-9 * st.f_stop, st.f_step)
    k = np.clip(np.round(req / df).astype(int), 1, o.block_size // 2 - 1)
    uk = np.unique(k)
    return uk * df, uk, int(req.size - uk.size)


def harmonic_nyquist_note(n_h: int, f_lo: float, f_hi: float, fs: float, rates=None) -> str:
    """'' if orders 1…n_h are all below Nyquist over f_lo…f_hi; otherwise a plain-language warning with
    the highest measurable order at f_hi / f_lo and the sampling rate that would allow all n_h."""
    if fs <= 0 or f_hi <= 0 or n_h * f_hi < fs / 2:
        return ""
    m_hi = max(1, int(np.ceil(fs / 2 / f_hi)) - 1)       # largest h with h·f < fs/2
    m_lo = max(1, int(np.ceil(fs / 2 / f_lo)) - 1) if f_lo > 0 else m_hi
    need = 2 * n_h * f_hi
    pick = next((r for r in sorted(rates or []) if r > need), None)
    span = (f"orders 1…{min(m_lo, n_h)} at {f_lo:g} Hz down to 1…{m_hi} at {f_hi:g} Hz" if m_lo != m_hi
            else f"orders 1…{m_hi} at {f_hi:g} Hz")
    return (f"Only {span} are below Nyquist (fs/2 = {fs / 2:g} Hz); higher orders cannot be measured. "
            f"All {n_h} orders at {f_hi:g} Hz need fs > {need:g} Hz"
            + (f" (e.g. {pick:g} Hz)." if pick else "."))


def custom_points_problems(points, fs: float) -> List[str]:
    pts = [(float(f), float(a)) for f, a in points]
    if len(pts) < 2:
        return ["Custom spectrum: enter at least two (frequency, amplitude) points."]
    errs = []
    fr = [f for f, _ in pts]
    if any(b <= a for a, b in zip(fr, fr[1:])):
        errs.append("Custom spectrum: frequencies must be strictly increasing.")
    if any(f < 0 for f in fr) or any(a < 0 for _, a in pts):
        errs.append("Custom spectrum: frequencies and amplitudes must be ≥ 0.")
    if not any(a > 0 and f < fs / 2 for f, a in pts):
        errs.append("Custom spectrum: no non-zero amplitude below Nyquist (fs/2).")
    return errs


def default_config() -> AppConfig:
    cfg = AppConfig()
    cfg.channels = [
        ChannelConfig(physical="cDAQ1Mod1/ai0", name="Force_drive", node=1, direction="+Z",
                      sensor_type="Force (IEPE)", sensitivity=10.0, units="N", max_range=200.0,
                      is_reference=True),
        ChannelConfig(physical="cDAQ1Mod1/ai1", name="Acc_1", node=1, direction="+Z"),
        ChannelConfig(physical="cDAQ1Mod1/ai2", name="Acc_2", node=2, direction="+Z"),
        ChannelConfig(physical="cDAQ1Mod1/ai3", name="Acc_3", node=3, direction="+Z"),
    ]
    return cfg

#!/usr/bin/env python3
"""
nvgpu-exporter — comprehensive NVIDIA GPU stats for Linux, including the
GDDR6/GDDR6X VRAM-junction + hotspot temps that nvidia-smi hides on consumer
cards (read straight off PCIe BAR0 via the bundled `gputemps` reader).

Modes (env MODE):
  json        (default) — one JSON object per line to stdout every INTERVAL s
  prometheus            — HTTP server exposing /metrics (and /power_limit if
                          control is enabled)

Optional GATED control (env ENABLE_POWER_LIMIT=1, prometheus mode only):
  POST /power_limit?watts=NNN   -> nvidia-smi -pl NNN, clamped to [PL_MIN,PL_MAX]
                                   (defaults to the card's own min/max limits)
Other NVML control knobs EXIST but are intentionally NOT implemented here
(keep the shareable image safe):
  - persistence mode        nvidia-smi -pm 1
  - lock GPU clocks         nvidia-smi -lgc min,max   (crude undervolt substitute)
  - lock memory clocks      nvidia-smi -lmc min,max
  - application clocks      nvidia-smi -ac mem,gfx
  - compute mode / reset    nvidia-smi -c / -r
  NOTE: true V/F undervolt (Afterburner) has NO Linux/NVML API; fan control
  needs X+coolbits and is not available headless.
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

GPUTEMPS = "/usr/local/bin/gputemps"

# ---------------- config layer (env defaults, /config/settings.json overrides) ----------------
STATE_DIR = os.environ.get("NVGPU_STATE_DIR", "/config")
CONFIG_FILE = os.path.join(STATE_DIR, "settings.json")

# keys editable via the settings page / docker vars. (bool) => on/off toggle.
SETTING_KEYS = [
    "NODE_NAME", "INTERVAL",
    "ENABLE_DASHBOARD", "ENABLE_METRICS", "ENABLE_JSON",     # web routes (bool)
    "ENABLE_MQTT", "ENABLE_INFLUX",                          # publishers (bool)
    "ENABLE_POWER_LIMIT", "PL_MIN", "PL_MAX",                # control
    "MQTT_HOST", "MQTT_PORT", "MQTT_USER", "MQTT_PASS", "MQTT_DISCOVERY_PREFIX",
    "INFLUX_URL", "INFLUX_DB", "INFLUX_USER", "INFLUX_PASS", "INFLUX_INTERVAL",
    "INFLUX_ORG", "INFLUX_BUCKET", "INFLUX_TOKEN",
]
SECRET_KEYS = {"MQTT_PASS", "INFLUX_PASS", "INFLUX_TOKEN"}
BOOL_KEYS = {"ENABLE_DASHBOARD", "ENABLE_METRICS", "ENABLE_JSON",
             "ENABLE_MQTT", "ENABLE_INFLUX", "ENABLE_POWER_LIMIT"}


def _load_saved():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_SAVED = _load_saved()


def cfg(key, default=None):
    """Saved settings.json wins over env var wins over default."""
    v = _SAVED.get(key)
    if v not in (None, ""):
        return v
    return os.environ.get(key, default)


def cfg_bool(key, default=True):
    v = cfg(key, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def save_settings(new):
    merged = dict(_SAVED)
    for k in SETTING_KEYS:
        if k in new:
            val = new[k]
            # empty string clears an override (falls back to env/default)
            if val in (None, ""):
                merged.pop(k, None)
            else:
                merged[k] = val
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=1)
    os.replace(tmp, CONFIG_FILE)
    _SAVED.clear()
    _SAVED.update(merged)


def settings_state():
    """Current effective settings for the settings page. Secrets are never
    returned — only whether one is set."""
    eff, is_saved, is_set = {}, {}, {}
    for k in SETTING_KEYS:
        v = cfg(k)
        is_saved[k] = k in _SAVED
        if k in SECRET_KEYS:
            eff[k] = ""
            is_set[k] = bool(v)
        elif k in BOOL_KEYS:
            eff[k] = cfg_bool(k)
        else:
            eff[k] = v if v is not None else ""
    return {"effective": eff, "saved": is_saved, "secret_set": is_set,
            "bool_keys": sorted(BOOL_KEYS), "secret_keys": sorted(SECRET_KEYS)}

# --- nvidia-smi comprehensive field list (all valid on Ampere/595) ---
FIELDS = [
    "index", "uuid", "name", "pstate",
    "temperature.gpu",
    "utilization.gpu", "utilization.memory",
    "utilization.encoder", "utilization.decoder",
    "utilization.jpeg", "utilization.ofa",
    "encoder.stats.sessionCount", "encoder.stats.averageFps",
    "encoder.stats.averageLatency",
    "clocks.gr", "clocks.sm", "clocks.mem", "clocks.video",
    "clocks.max.gr", "clocks.max.mem",
    "power.draw", "power.limit", "enforced.power.limit",
    "power.default_limit", "power.min_limit", "power.max_limit",
    "memory.total", "memory.used", "memory.free",
    "fan.speed",
    "pcie.link.gen.current", "pcie.link.gen.max",
    "pcie.link.width.current", "pcie.link.width.max",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sync_boost",
    "clocks_event_reasons.gpu_idle",
]

# static per-card thresholds (queried once from -q)
_THRESHOLDS = None


class _EmptyResult:
    stdout = ""
    returncode = 127

    def __init__(self, err=""):
        self.stderr = err


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as e:
        # nvidia-smi missing/hung -> degrade gracefully (no GPU) instead of 500
        return _EmptyResult(str(e))


def _num(v):
    """Parse an nvidia-smi cell to float, or None for N/A/[Not Supported]."""
    if v is None:
        return None
    v = v.strip()
    if not v or v.startswith("[") or v in ("N/A", "Enabled", "Disabled"):
        return None
    try:
        return float(v.split()[0])
    except (ValueError, IndexError):
        return None


def _active(v):
    return 1 if v and v.strip() == "Active" else 0


def _mb(v):
    """nvidia-smi reports VRAM in MiB; convert to MB (10^6 bytes) for a unit
    consistent with PCIe bandwidth (MB/s)."""
    n = _num(v)
    return round(n * 1.048576) if n is not None else None


def bdf_list():
    r = _run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"])
    out = []
    for line in r.stdout.strip().splitlines():
        s = line.strip().lower()
        if s.startswith("0000") and len(s) > 12:  # 00000000:d8:00.0 -> 0000:d8:00.0
            s = s[4:]
        if s:
            out.append(s)
    return out


def thresholds():
    global _THRESHOLDS
    if _THRESHOLDS is not None:
        return _THRESHOLDS
    _THRESHOLDS = {}
    r = _run(["nvidia-smi", "-q", "-d", "TEMPERATURE"])
    keymap = {
        "GPU Target Temperature": "target",
        "GPU Slowdown Temp": "slowdown",
        "GPU Shutdown Temp": "shutdown",
        "GPU Max Operating Temp": "max_op",
    }
    for line in r.stdout.splitlines():
        if ":" in line:
            k, _, val = line.partition(":")
            k = k.strip()
            if k in keymap:
                _THRESHOLDS[keymap[k]] = _num(val)
    return _THRESHOLDS


# PCI subsystem (add-in board) vendor IDs -> friendly AIB name. The low 16 bits
# of a GPU's Sub System Id identify who built the physical card. Pre-populated
# with the common ones; if a card shows a raw 0x**** code here instead of a
# name, please report it at https://github.com/edddeduck/nvidia-status/issues
# so we can add it.
PCI_SUBVENDORS = {
    0x10DE: "NVIDIA (Founders/reference)",
    0x1043: "ASUS",
    0x1458: "Gigabyte",
    0x1462: "MSI",
    0x3842: "EVGA",
    0x19DA: "Zotac",
    0x1569: "Palit",
    0x10B0: "Gainward",
    0x1DA2: "Sapphire",
    0x196E: "PNY",
    0x1682: "XFX",
    0x148C: "PowerColor",
    0x1B4C: "KFA2/Galax",
    0x7377: "Colorful",
    0x1ACC: "Point of View",
    0x174B: "PC Partner",
    0x2646: "Kingston",
    0x1E83: "Yeston",
    0x1DEE: "Biostar",
    0x1EAE: "Emtek/Leadtek",
    0x1462: "MSI",
    0x0000: "Unknown/OEM",
}


def vendor_name(subsystem_id):
    """Decode a Sub System Id (e.g. '0x87AF1043') to the add-in board vendor.
    Returns the vendor name, or the raw 0x**** sub-vendor code if unknown."""
    if not subsystem_id:
        return None
    try:
        sub = int(str(subsystem_id).strip(), 16) & 0xFFFF
    except (ValueError, TypeError):
        return None
    return PCI_SUBVENDORS.get(sub, f"0x{sub:04X}")


def _smi_q_blocks():
    """Parse `nvidia-smi -q` into one flat dict per GPU, plus the cumulative
    'Clocks Event Reasons Counters' (µs) which share key names with the live
    reasons section, so we track that sub-section by indent to disambiguate.
    Returns {uuid: {"info": {...}, "counters": {reason: microseconds}}}."""
    r = _run(["nvidia-smi", "-q"])
    text = getattr(r, "stdout", "") or ""
    # header-level Driver/CUDA versions apply to every GPU on the host
    drv = cuda = None
    for line in text.splitlines()[:8]:
        if line.startswith("Driver Version"):
            drv = line.partition(":")[2].strip() or None
        elif line.startswith("CUDA Version"):
            cuda = line.partition(":")[2].strip() or None

    want = {
        "Product Name": "name", "Product Brand": "brand",
        "Product Architecture": "architecture", "Serial Number": "serial",
        "GPU PDI": "pdi", "GPU Part Number": "part_number",
        "VBIOS Version": "vbios", "Board ID": "board_id",
        "GSP Firmware Version": "gsp_firmware", "Sub System Id": "subsystem_id",
        "Device Id": "device_id", "Image Version": "inforom_img",
        "OEM Object": "inforom_oem",
    }
    counter_keys = {
        "SW Power Capping": "sw_power_cap", "SW Thermal Slowdown": "sw_thermal",
        "HW Thermal Slowdown": "hw_thermal", "HW Power Braking": "hw_power_brake",
        "Sync Boost": "sync_boost",
    }
    out, cur, uuid = {}, None, None
    in_counters = False
    for line in text.splitlines():
        stripped = line.strip()
        # a new GPU block starts at "GPU 00000000:D8:00.0"
        if line.startswith("GPU ") and stripped.startswith("GPU "):
            cur, uuid, in_counters = {}, None, False
            continue
        if cur is None or ":" not in line:
            # entering/leaving the counters sub-section (4-space section header)
            if stripped == "Clocks Event Reasons Counters":
                in_counters = True
            continue
        indent = len(line) - len(line.lstrip(" "))
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        # counter lines sit at indent 8 under the section; a shallower line ends it
        if in_counters and indent <= 4:
            in_counters = False
        if in_counters and k in counter_keys:
            try:
                cur.setdefault("_counters", {})[counter_keys[k]] = int(v.split()[0])
            except (ValueError, IndexError):
                pass
            continue
        if k == "GPU UUID":
            uuid = v
            cur["uuid"] = v
            cur["driver_version"] = drv
            cur["cuda_version"] = cuda
            out[v] = {"info": cur, "counters": cur.get("_counters", {})}
        elif k in want and want[k] not in cur:
            cur[want[k]] = None if v in ("N/A", "") else v
    # decode vendor + counter linkage after each block is complete
    for u, blk in out.items():
        blk["counters"] = blk["info"].pop("_counters", {})
        blk["info"]["vendor"] = vendor_name(blk["info"].get("subsystem_id"))
    return out


_DEVICE_INFO = None


def device_info(blocks=None):
    """Static identity + firmware per GPU UUID. Cached — never changes at runtime."""
    global _DEVICE_INFO
    if _DEVICE_INFO is not None:
        return _DEVICE_INFO
    b = blocks if blocks is not None else _smi_q_blocks()
    _DEVICE_INFO = {u: blk["info"] for u, blk in b.items()}
    return _DEVICE_INFO


def bar0_temps(bdfs):
    """core/junction/vram per GPU index from the BAR0 reader."""
    out = {}
    if not bdfs:
        return out
    r = _run([GPUTEMPS, "--device", ",".join(bdfs), "--once", "--json"])
    for line in r.stdout.strip().splitlines():
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        for g in doc.get("gpus", []):
            out[g["index"]] = {
                "core": g.get("core"),
                "junction": g.get("junction"),
                "vram": g.get("vram"),
            }
    return out


def pcie_throughput():
    """rx/tx MB/s per GPU via `nvidia-smi dmon -s t`.
    rx = host->device (uploads: model weights, input frames, textures).
    tx = device->host (downloads: generated frames, tokens, decoded video)."""
    out = {}
    r = _run(["nvidia-smi", "dmon", "-s", "t", "-c", "1"])
    for line in r.stdout.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) >= 3:
            try:
                out[int(p[0])] = {"rx": float(p[1]), "tx": float(p[2])}
            except ValueError:
                pass
    return out


# optional ro mount of /var/lib/docker/containers -> friendly container names
CONTAINERS_DIR = "/hostcontainers"


def _pid_container_id(pid):
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            m = re.search(r"docker[-/]([0-9a-f]{12,64})", f.read())
            return m.group(1) if m else None
    except OSError:
        return None


def _container_name(cid):
    if not cid:
        return None
    try:
        for d in os.listdir(CONTAINERS_DIR):
            if d.startswith(cid) or cid.startswith(d):
                with open(os.path.join(CONTAINERS_DIR, d, "config.v2.json")) as f:
                    return (json.load(f).get("Name") or "").lstrip("/") or cid[:12]
    except (OSError, ValueError):
        pass
    return cid[:12]


def processes(indices=None):
    """Per-process GPU users + owning container, attributed to the right GPU.
    Queries compute-apps per-GPU (so each process is tied to its card).
    Needs --pid=host to see host PIDs; container names need the optional
    /var/lib/docker/containers ro mount (else falls back to short id)."""
    if not indices:
        indices = [0]
    procs = {}  # keyed (gpu, pid)
    for idx in indices:
        r = _run(["nvidia-smi", "-i", str(idx),
                  "--query-compute-apps=pid,process_name,used_gpu_memory",
                  "--format=csv,noheader,nounits"])
        for row in r.stdout.strip().splitlines():
            c = [x.strip() for x in row.split(",")]
            if len(c) >= 3:
                try:
                    pid = int(c[0])
                except ValueError:
                    continue
                procs[(idx, pid)] = {"gpu": idx, "pid": pid, "name": c[1],
                                     "used_mb": _mb(c[2]),
                                     "sm": None, "enc": None, "dec": None}
    # pmon gives per (gpu,pid) sm/enc/dec — first column is the GPU index
    rp = _run(["nvidia-smi", "pmon", "-c", "1"])
    for line in rp.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 7:
            try:
                g, pid = int(p[0]), int(p[1])
            except ValueError:
                continue
            if (g, pid) in procs:
                procs[(g, pid)]["sm"] = _num(p[3])
                procs[(g, pid)]["enc"] = _num(p[5])
                procs[(g, pid)]["dec"] = _num(p[6])
    for pr in procs.values():
        cid = _pid_container_id(pr["pid"])
        pr["container_id"] = cid[:12] if cid else None
        pr["container"] = _container_name(cid) if cid else None
    return list(procs.values())


def fbc_stats():
    """Frame-buffer capture (NVFBC) stats per GPU — used by streaming capture
    (Sunshine/Steam-headless). Parsed from `nvidia-smi -q` (no query field)."""
    out = {}
    r = _run(["nvidia-smi", "-q"])
    idx = -1
    section = None
    for line in r.stdout.splitlines():
        s = line.strip()
        m = re.match(r"GPU (\d+):", s) or re.match(r"Minor Number\s*:\s*(\d+)", s)
        if s.startswith("FBC Stats"):
            section = "fbc"
            continue
        if s and not s.startswith(" ") and ":" in s and section == "fbc" \
                and not s.startswith(("Active", "Average")):
            section = None
        if s.startswith("Minor Number"):
            try:
                idx = int(s.split(":")[1])
            except (ValueError, IndexError):
                pass
        if section == "fbc":
            d = out.setdefault(max(idx, 0), {})
            if s.startswith("Active Sessions"):
                d["sessions"] = _num(s.split(":")[1])
            elif s.startswith("Average FPS"):
                d["fps"] = _num(s.split(":")[1])
            elif s.startswith("Average Latency"):
                d["latency_us"] = _num(s.split(":")[1])
    return out


# --- NVML per-session encode detail (codec/resolution) via ctypes ---
_NVML = None
_NVML_HANDLES = {}
_CODECS = {0: "H264", 1: "HEVC", 2: "AV1"}


class _EncSession(ctypes.Structure):
    _fields_ = [("sessionId", ctypes.c_uint), ("pid", ctypes.c_uint),
                ("vgpuInstance", ctypes.c_uint), ("codecType", ctypes.c_uint),
                ("hRes", ctypes.c_uint), ("vRes", ctypes.c_uint),
                ("fps", ctypes.c_uint), ("lat", ctypes.c_uint)]


def _nvml():
    global _NVML
    if _NVML is None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit_v2 failed")
            _NVML = lib
        except Exception as e:
            sys.stderr.write(f"nvml: encode-session detail unavailable ({e})\n")
            _NVML = False
    return _NVML or None


def encoder_sessions(idx):
    """Per-session encode detail (codec/res/fps/latency + container) for GPU idx."""
    lib = _nvml()
    if not lib:
        return []
    try:
        h = _NVML_HANDLES.get(idx)
        if h is None:
            h = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h)) != 0:
                return []
            _NVML_HANDLES[idx] = h
        cnt = ctypes.c_uint(0)
        lib.nvmlDeviceGetEncoderSessions(h, ctypes.byref(cnt), None)  # sets cnt
        if not cnt.value:
            return []
        arr = (_EncSession * cnt.value)()
        if lib.nvmlDeviceGetEncoderSessions(h, ctypes.byref(cnt), arr) != 0:
            return []
        out = []
        for s in arr[:cnt.value]:
            cid = _pid_container_id(s.pid)
            out.append({"pid": s.pid, "codec": _CODECS.get(s.codecType, str(s.codecType)),
                        "width": s.hRes, "height": s.vRes, "fps": s.fps,
                        "latency_us": s.lat,
                        "container": _container_name(cid) if cid else None})
        return out
    except Exception as e:
        sys.stderr.write(f"nvml: encoder_sessions error {e}\n")
        return []


def collect():
    bdfs = bdf_list()
    bar0 = bar0_temps(bdfs)
    thr = thresholds()
    pcie_bw = pcie_throughput()
    fbc = fbc_stats()
    blocks = _smi_q_blocks()          # static identity + cumulative throttle counters
    dinfo = device_info(blocks)       # cached after first call
    q = "--query-gpu=" + ",".join(FIELDS)
    r = _run(["nvidia-smi", q, "--format=csv,noheader,nounits"])
    gpus = []
    for row in r.stdout.strip().splitlines():
        cells = [c.strip() for c in row.split(",")]
        if len(cells) != len(FIELDS):
            continue
        d = dict(zip(FIELDS, cells))
        idx = int(_num(d["index"]) or 0)
        b = bar0.get(idx, {})
        gpu = {
            "index": idx,
            "uuid": d["uuid"],
            "name": d["name"],
            "pstate": d["pstate"],
            "temp": {
                "core": _num(d["temperature.gpu"]),
                "junction": b.get("junction"),   # BAR0
                "vram": b.get("vram"),            # BAR0
                "target": thr.get("target"),
                "slowdown": thr.get("slowdown"),
                "shutdown": thr.get("shutdown"),
                "max_op": thr.get("max_op"),
            },
            "util": {
                "gpu": _num(d["utilization.gpu"]),
                "memory": _num(d["utilization.memory"]),
                "encoder": _num(d["utilization.encoder"]),
                "decoder": _num(d["utilization.decoder"]),
            },
            "clocks_mhz": {
                "graphics": _num(d["clocks.gr"]),
                "sm": _num(d["clocks.sm"]),
                "memory": _num(d["clocks.mem"]),
                "video": _num(d["clocks.video"]),
                "graphics_max": _num(d["clocks.max.gr"]),
                "memory_max": _num(d["clocks.max.mem"]),
            },
            "power_w": {
                "draw": _num(d["power.draw"]),
                "limit": _num(d["power.limit"]),
                "enforced_limit": _num(d["enforced.power.limit"]),
                "default_limit": _num(d["power.default_limit"]),
                "min_limit": _num(d["power.min_limit"]),
                "max_limit": _num(d["power.max_limit"]),
            },
            "memory_mb": {
                "total": _mb(d["memory.total"]),
                "used": _mb(d["memory.used"]),
                "free": _mb(d["memory.free"]),
            },
            "fan_pct": _num(d["fan.speed"]),
            "pcie": {
                "gen_current": _num(d["pcie.link.gen.current"]),
                "gen_max": _num(d["pcie.link.gen.max"]),
                "width_current": _num(d["pcie.link.width.current"]),
                "width_max": _num(d["pcie.link.width.max"]),
                "rx_mbps": pcie_bw.get(idx, {}).get("rx"),  # host->device (upload)
                "tx_mbps": pcie_bw.get(idx, {}).get("tx"),  # device->host (download)
            },
            "throttle": {
                "sw_power_cap": _active(d["clocks_event_reasons.sw_power_cap"]),
                "hw_thermal": _active(d["clocks_event_reasons.hw_thermal_slowdown"]),
                "sw_thermal": _active(d["clocks_event_reasons.sw_thermal_slowdown"]),
                "hw_power_brake": _active(d["clocks_event_reasons.hw_power_brake_slowdown"]),
                "sync_boost": _active(d["clocks_event_reasons.sync_boost"]),
                "idle": _active(d["clocks_event_reasons.gpu_idle"]),
                # cumulative throttle time in µs since driver load (not lifetime-
                # persistent; resets on reboot/driver reload). {reason: microseconds}
                "counters_us": blocks.get(d["uuid"], {}).get("counters", {}),
            },
            # static identity + firmware (survives power cycles). Never changes.
            "info": dinfo.get(d["uuid"], {}),
            "encode": {  # NVENC
                "sessions": _num(d["encoder.stats.sessionCount"]),
                "avg_fps": _num(d["encoder.stats.averageFps"]),
                "avg_latency_us": _num(d["encoder.stats.averageLatency"]),
                "util": _num(d["utilization.encoder"]),
                "jpeg_util": _num(d["utilization.jpeg"]),
                "ofa_util": _num(d["utilization.ofa"]),
                "fbc_sessions": fbc.get(idx, {}).get("sessions"),
                "fbc_fps": fbc.get(idx, {}).get("fps"),
                "fbc_latency_us": fbc.get(idx, {}).get("latency_us"),
                "sessions_detail": encoder_sessions(idx),  # codec/res per session
            },
            "decode": {  # NVDEC — NVIDIA exposes utilization only (no per-session/codec)
                "util": _num(d["utilization.decoder"]),
            },
        }
        gpus.append(gpu)
    return {"timestamp": int(time.time() * 1000), "gpus": gpus,
            "processes": processes([g["index"] for g in gpus])}


# ---------------- Prometheus ----------------
def prom_metrics():
    snap = collect()
    lines = []

    def m(name, help_, typ, samples):
        if not samples:
            return
        lines.append(f"# HELP {name} {help_}")
        lines.append(f"# TYPE {name} {typ}")
        for labels, val in samples:
            if val is None:
                continue
            lines.append(f"{name}{{{labels}}} {val}")

    def lbl(g):
        return f'gpu="{g["index"]}",uuid="{g["uuid"]}",name="{g["name"]}"'

    gs = snap["gpus"]
    for key, sub in (("core", "core"), ("junction", "junction"), ("vram", "vram"),
                     ("target", "target"), ("slowdown", "slowdown"),
                     ("shutdown", "shutdown"), ("max_op", "max_op")):
        m(f"nvgpu_temp_{sub}_celsius", f"{sub} temperature", "gauge",
          [(lbl(g), g["temp"][key]) for g in gs])
    for key in ("gpu", "memory", "encoder", "decoder"):
        m(f"nvgpu_util_{key}_ratio", f"{key} utilization %", "gauge",
          [(lbl(g), g["util"][key]) for g in gs])
    for key in ("graphics", "sm", "memory", "video", "graphics_max", "memory_max"):
        m(f"nvgpu_clock_{key}_mhz", f"{key} clock", "gauge",
          [(lbl(g), g["clocks_mhz"][key]) for g in gs])
    for key in ("draw", "limit", "enforced_limit", "default_limit", "min_limit", "max_limit"):
        m(f"nvgpu_power_{key}_watts", f"power {key}", "gauge",
          [(lbl(g), g["power_w"][key]) for g in gs])
    for key in ("total", "used", "free"):
        m(f"nvgpu_memory_{key}_megabytes", f"framebuffer {key} (MB)", "gauge",
          [(lbl(g), g["memory_mb"][key]) for g in gs])
    m("nvgpu_fan_ratio", "fan speed %", "gauge", [(lbl(g), g["fan_pct"]) for g in gs])
    for key in ("gen_current", "gen_max", "width_current", "width_max"):
        m(f"nvgpu_pcie_{key}", f"pcie {key}", "gauge",
          [(lbl(g), g["pcie"][key]) for g in gs])
    m("nvgpu_pcie_rx_megabytes_per_second",
      "PCIe host->device MB/s (uploads: weights/frames in)", "gauge",
      [(lbl(g), g["pcie"]["rx_mbps"]) for g in gs])
    m("nvgpu_pcie_tx_megabytes_per_second",
      "PCIe device->host MB/s (downloads: generated frames/tokens out)", "gauge",
      [(lbl(g), g["pcie"]["tx_mbps"]) for g in gs])
    for key in ("sw_power_cap", "hw_thermal", "sw_thermal", "hw_power_brake", "sync_boost", "idle"):
        m(f"nvgpu_throttle_{key}", f"throttle reason {key} active (0/1)", "gauge",
          [(lbl(g), g["throttle"][key]) for g in gs])
    # cumulative throttle time per reason (µs since driver load; resets on reboot)
    for key in ("sw_power_cap", "sw_thermal", "hw_thermal", "hw_power_brake", "sync_boost"):
        m(f"nvgpu_throttle_{key}_microseconds", f"cumulative {key} throttle time (us since driver load)", "counter",
          [(lbl(g), g["throttle"].get("counters_us", {}).get(key)) for g in gs])
    # static identity + firmware as a labelled info gauge (Prometheus idiom: value 1)
    m("nvgpu_info", "static GPU identity + firmware (value always 1)", "gauge",
      [(f'{lbl(g)},'
        f'vendor="{(g["info"].get("vendor") or "-")}",'
        f'architecture="{(g["info"].get("architecture") or "-")}",'
        f'vbios="{(g["info"].get("vbios") or "-")}",'
        f'inforom="{(g["info"].get("inforom_img") or "-")}",'
        f'gsp_firmware="{(g["info"].get("gsp_firmware") or "-")}",'
        f'driver="{(g["info"].get("driver_version") or "-")}",'
        f'part_number="{(g["info"].get("part_number") or "-")}",'
        f'serial="{(g["info"].get("serial") or "-")}"', 1) for g in gs])

    # encode/decode
    m("nvgpu_encode_sessions", "active NVENC sessions", "gauge",
      [(lbl(g), g["encode"]["sessions"]) for g in gs])
    m("nvgpu_encode_avg_fps", "NVENC average FPS", "gauge",
      [(lbl(g), g["encode"]["avg_fps"]) for g in gs])
    m("nvgpu_encode_avg_latency_us", "NVENC average latency (us)", "gauge",
      [(lbl(g), g["encode"]["avg_latency_us"]) for g in gs])
    m("nvgpu_encode_util_ratio", "NVENC utilization %", "gauge",
      [(lbl(g), g["encode"]["util"]) for g in gs])
    m("nvgpu_decode_util_ratio", "NVDEC utilization %", "gauge",
      [(lbl(g), g["decode"]["util"]) for g in gs])
    m("nvgpu_fbc_sessions", "frame-buffer-capture sessions (streaming)", "gauge",
      [(lbl(g), g["encode"]["fbc_sessions"]) for g in gs])
    # per-encode-session codec detail
    esamp = []
    for g in gs:
        for s in g["encode"]["sessions_detail"]:
            el = (f'gpu="{g["index"]}",pid="{s["pid"]}",codec="{s["codec"]}",'
                  f'resolution="{s["width"]}x{s["height"]}",'
                  f'container="{s.get("container") or "-"}"')
            esamp.append((el, s["fps"]))
    m("nvgpu_encode_session_fps", "per-session encode FPS (labelled codec/res/container)",
      "gauge", esamp)

    # per-process / per-container GPU users
    def plbl(p):
        return (f'gpu="{p.get("gpu", 0)}",pid="{p["pid"]}",proc="{p["name"]}",'
                f'container="{p.get("container") or "-"}",'
                f'container_id="{p.get("container_id") or "-"}"')
    procs = snap.get("processes", [])
    m("nvgpu_process_memory_megabytes", "per-process VRAM usage (MB)", "gauge",
      [(plbl(p), p["used_mb"]) for p in procs])
    for key in ("sm", "enc", "dec"):
        m(f"nvgpu_process_{key}_ratio", f"per-process {key} utilization %", "gauge",
          [(plbl(p), p[key]) for p in procs])
    return "\n".join(lines) + "\n"


ENABLE_PL = cfg_bool("ENABLE_POWER_LIMIT", False)
LATCH_FILE = os.path.join(STATE_DIR, "power_limit.latch")


def get_latch():
    """Return {gpu_index: watts}. Migrates the old single-int format to GPU 0."""
    try:
        with open(LATCH_FILE) as f:
            d = json.load(f)
        if isinstance(d, int):
            return {0: d}
        return {int(k): int(v) for k, v in d.items()}
    except (OSError, ValueError):
        return {}


def _write_latch(d):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = LATCH_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({str(k): v for k, v in d.items()}, f)
    os.replace(tmp, LATCH_FILE)


def _set_latch(idx, watts):
    try:
        d = get_latch()
        d[int(idx)] = int(watts)
        _write_latch(d)
    except OSError as e:
        sys.stderr.write(f"latch: could not persist ({e}); mount a volume at {STATE_DIR}\n")


def _clear_latch(idx):
    d = get_latch()
    d.pop(int(idx), None)
    try:
        if d:
            _write_latch(d)
        else:
            os.remove(LATCH_FILE)
    except OSError:
        pass


def pl_bounds(g):
    """(lo, hi) from env override OR the GPU's queried limits. Returns None for
    either bound if it can't be determined — callers must NOT assume a default."""
    env_lo, env_hi = os.environ.get("PL_MIN"), os.environ.get("PL_MAX")
    q = g.get("power_w", {}) if g else {}
    lo = int(env_lo) if env_lo else (int(q["min_limit"]) if q.get("min_limit") else None)
    hi = int(env_hi) if env_hi else (int(q["max_limit"]) if q.get("max_limit") else None)
    return lo, hi


def set_power_limit(watts, latch=None, gpu_idx=0):
    snap = collect()
    g = next((x for x in snap.get("gpus", []) if x["index"] == int(gpu_idx)), None)
    if g is None:
        return 500, f"GPU {gpu_idx} not found"
    lo, hi = pl_bounds(g)
    if lo is None or hi is None:
        return 503, (f"power-limit range could not be read from GPU {gpu_idx} — "
                     "refusing to set for safety (set PL_MIN/PL_MAX to override)")
    try:
        w = int(watts)
    except (TypeError, ValueError):
        return 400, "watts must be an integer"
    if w < lo or w > hi:
        return 400, f"watts {w} out of range [{lo},{hi}] for GPU {gpu_idx}"
    r = _run(["nvidia-smi", "-i", str(gpu_idx), "-pl", str(w)])
    if r.returncode != 0:
        return 500, (r.stdout + r.stderr).strip()
    if latch is True:
        _set_latch(gpu_idx, w)
    elif latch is False:
        _clear_latch(gpu_idx)
    msg = (r.stdout + r.stderr).strip()
    if latch is True:
        msg += f"  [GPU {gpu_idx} latched: {w}W re-applies on restart]"
    elif latch is False:
        msg += f"  [GPU {gpu_idx} latch cleared]"
    return 200, msg


def power_state():
    snap = collect()
    latched = get_latch()
    out = {"enabled": ENABLE_PL, "gpus": []}
    for g in snap.get("gpus", []):
        idx = g["index"]
        pw = g["power_w"]
        lo, hi = pl_bounds(g)
        cur = pw.get("limit")
        avail, reason = ENABLE_PL, None
        if ENABLE_PL and (lo is None or hi is None or cur is None):
            avail, reason = False, (
                "could not read the power-limit range from nvidia-smi — control "
                "disabled for safety. Set PL_MIN/PL_MAX to override.")
        out["gpus"].append({"index": idx, "name": g.get("name"), "available": avail,
                            "reason": reason, "current": cur,
                            "default": pw.get("default_limit"),
                            "min": lo, "max": hi, "latched": latched.get(idx)})
    if ENABLE_PL and not out["gpus"]:
        out["reason"] = "no GPU detected"
    return out


def apply_latch_on_startup():
    if ENABLE_PL:
        for idx, w in get_latch().items():
            code, msg = set_power_limit(w, None, idx)
            sys.stderr.write(f"power-limit: startup re-apply GPU {idx} {w}W -> {msg}\n")


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nvidia Status</title>
<style>
:root{
  --bg:#0d1117;--card:#161b22;--edge:#30363d;--fg:#e6edf3;--muted:#8b949e;
  --track:#21262d;--green:#3fb950;--amber:#d29922;--orange:#db6d28;--red:#f85149;
  --accent:#58a6ff;
}
@media (prefers-color-scheme:light){:root{
  --bg:#f6f8fa;--card:#fff;--edge:#d0d7de;--fg:#1f2328;--muted:#636c76;--track:#eaeef2;--accent:#0969da;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--edge)}
header h1{font-size:16px;margin:0;font-weight:600}
header .dot{width:9px;height:9px;border-radius:50%;background:var(--green);box-shadow:0 0 6px var(--green)}
header .meta{margin-left:auto;color:var(--muted);font-size:12px}
.wrap{padding:16px 20px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:16px}
.card h2{font-size:15px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12px;margin-bottom:12px}
.temps{display:flex;gap:10px;margin-bottom:14px}
.temp{flex:1;text-align:center;background:var(--track);border-radius:8px;padding:8px 4px}
.temp .v{font-size:22px;font-weight:700;line-height:1}
.temp .l{font-size:11px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.03em}
.row{display:flex;justify-content:space-between;font-size:12px;margin:7px 0 3px}
.row .k{color:var(--muted)}
.bar{height:7px;background:var(--track);border-radius:4px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:4px;transition:width .4s,background .4s}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px 18px;margin-top:6px}
.pcie{display:flex;gap:14px;margin-top:12px;font-size:13px}
.pcie b{font-variant-numeric:tabular-nums}
.thr{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.chip{font-size:11px;padding:2px 8px;border-radius:20px;background:var(--track);color:var(--muted)}
.chip.on{background:var(--red);color:#fff}
table{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}
th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--edge)}
th{color:var(--muted);font-weight:500}
th.n,td.n{text-align:right;font-variant-numeric:tabular-nums}
td.mono,th.mono{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted)}
.tag{background:var(--accent);color:#fff;padding:1px 7px;border-radius:5px;font-size:11px}
.empty{color:var(--muted);padding:10px 0}
.scroll{overflow-x:auto}
</style></head>
<body>
<header><span class="dot"></span><h1>Nvidia Status</h1>
<a href="/settings" style="margin-left:16px;color:var(--accent);text-decoration:none;font-size:13px">Settings</a>
<a href="/help" style="margin-left:14px;color:var(--accent);text-decoration:none;font-size:13px">Help / API ↗</a>
<span class="meta" id="meta">connecting…</span></header>
<div class="wrap" id="gpus"></div>
<div class="wrap"><div class="card" style="grid-column:1/-1">
  <h2>GPU processes &amp; containers</h2>
  <div class="sub">who is using the card right now</div>
  <table id="proc"><thead><tr>
    <th>Container</th><th class="n">GPU</th><th>Process</th><th>PID</th>
    <th class="n">VRAM MB</th><th class="n">Compute %</th><th class="n">Encode %</th><th class="n">Decode %</th>
  </tr></thead><tbody></tbody></table>
</div></div>
<div class="wrap"><div class="card" style="grid-column:1/-1">
  <h2>Device info</h2>
  <div class="sub">static identity &amp; firmware — survives power cycles</div>
  <div class="scroll"><table id="info"><thead><tr>
    <th>GPU</th><th>Model</th><th>Vendor</th><th>Architecture</th><th>VBIOS</th>
    <th>InfoROM</th><th>GSP fw</th><th>Driver</th><th>Part no.</th><th>Serial</th><th class="mono">UUID</th>
  </tr></thead><tbody></tbody></table></div>
</div></div>
<div class="wrap"><div class="card" style="grid-column:1/-1">
  <h2>Throttle counters</h2>
  <div class="sub">cumulative time the card was held below clocks, per reason · since driver load (resets on reboot)</div>
  <div class="scroll"><table id="throt"><thead><tr>
    <th>GPU</th><th class="n">SW power cap</th><th class="n">SW thermal</th>
    <th class="n">HW thermal</th><th class="n">HW power brake</th><th class="n">Sync boost</th>
  </tr></thead><tbody></tbody></table></div>
</div></div>
<script>
const F=(v,d=0)=>v==null?"—":Number(v).toFixed(d);
// microseconds -> compact human duration (e.g. "9h 3m", "12s", "0")
function DUR(us){if(us==null)return"—";let s=us/1e6;if(s<1)return s<=0?"0":s.toFixed(1)+"s";
  const d=Math.floor(s/86400);s-=d*86400;const h=Math.floor(s/3600);s-=h*3600;
  const m=Math.floor(s/60);const sec=Math.floor(s-m*60);
  if(d)return d+"d "+h+"h";if(h)return h+"h "+m+"m";if(m)return m+"m "+sec+"s";return sec+"s";}
function col(v,warn,hot,crit){if(v==null)return"var(--muted)";
  if(v>=crit)return"var(--red)";if(v>=hot)return"var(--orange)";
  if(v>=warn)return"var(--amber)";return"var(--green)";}
function bar(pct,color){pct=Math.max(0,Math.min(100,pct||0));
  return '<div class="bar"><i style="width:'+pct+'%;background:'+color+'"></i></div>';}
function tempBox(label,v,warn,hot,crit){
  const c=col(v,warn,hot,crit);
  return '<div class="temp"><div class="v" style="color:'+c+'">'+F(v)+'°</div>'+
         '<div class="l">'+label+'</div></div>';}
function gpuCard(g){
  const t=g.temp,p=g.power_w,c=g.clocks_mhz,u=g.util,m=g.memory_mb,pc=g.pcie,th=g.throttle;
  const shut=t.shutdown||98, slow=t.slowdown||95, targ=t.target||83;
  let h='<div class="card"><h2>'+g.name+'</h2>'+
    '<div class="sub">GPU '+g.index+' · '+g.pstate+' · '+(g.uuid||"").slice(0,19)+'…</div>';
  h+='<div class="temps">'+
     tempBox("Core",t.core,targ,slow,shut)+
     tempBox("Junction",t.junction,85,95,100)+
     tempBox("VRAM",t.vram,85,95,100)+'</div>';
  // power
  h+='<div class="row"><span class="k">Power</span><span>'+F(p.draw)+' / '+F(p.limit)+' W (max '+F(p.max_limit)+')</span></div>'+
     bar(100*(p.draw||0)/(p.max_limit||1),col(p.draw,0.6*p.max_limit,0.85*p.max_limit,p.max_limit));
  // vram
  h+='<div class="row"><span class="k">VRAM</span><span>'+F(m.used)+' / '+F(m.total)+' MB</span></div>'+
     bar(100*(m.used||0)/(m.total||1),"var(--accent)");
  // util + clocks
  h+='<div class="grid2">'+
     '<div><div class="row"><span class="k">GPU compute</span><span>'+F(u.gpu)+'%</span></div>'+bar(u.gpu,"var(--accent)")+'</div>'+
     '<div><div class="row"><span class="k">Memory controller</span><span>'+F(u.memory)+'%</span></div>'+bar(u.memory,"var(--accent)")+'</div>'+
     '<div><div class="row"><span class="k">Graphics clock</span><span>'+F(c.graphics)+' / '+F(c.graphics_max)+' MHz</span></div>'+bar(100*(c.graphics||0)/(c.graphics_max||1),"var(--muted)")+'</div>'+
     '<div><div class="row"><span class="k">Memory clock</span><span>'+F(c.memory)+' MHz</span></div>'+bar(100*(c.memory||0)/(c.memory_max||1),"var(--muted)")+'</div>'+
     '</div>';
  // pcie bandwidth + link
  // PCIe up/down as bars (% of link max) with gen×width tag inline
  const en=g.encode||{}, de=g.decode||{};
  const perLane={1:250,2:500,3:985,4:1969,5:3938};   // MB/s per lane by gen
  const maxbw=(perLane[pc.gen_current]||985)*(pc.width_current||16);
  const gentag=' <span style="color:var(--muted);font-weight:400">(PCIe gen'+F(pc.gen_current)+' ×'+F(pc.width_current)+')</span>';
  const pct=(v)=>100*(v||0)/maxbw;
  h+='<div class="grid2">'+
     '<div><div class="row"><span class="k">↑ Upload'+gentag+'</span><span>'+F(pc.rx_mbps)+' MB/s</span></div>'+bar(pct(pc.rx_mbps),"var(--green)")+'</div>'+
     '<div><div class="row"><span class="k">↓ Download'+gentag+'</span><span>'+F(pc.tx_mbps)+' MB/s</span></div>'+bar(pct(pc.tx_mbps),"var(--accent)")+'</div>'+
     '<div><div class="row"><span class="k">Video encode (NVENC)</span><span>'+F(en.util)+'%</span></div>'+bar(en.util,"var(--accent)")+'</div>'+
     '<div><div class="row"><span class="k">Video decode (NVDEC)</span><span>'+F(de.util)+'%</span></div>'+bar(de.util,"var(--accent)")+'</div>'+
     '</div>';
  // active encode sessions — count in the header, details directly below
  const sd=en.sessions_detail||[];
  h+='<div class="row"><span class="k">Active encode sessions ('+F(en.sessions)+')</span>'+
     (en.fbc_sessions?'<span style="color:var(--muted)">Frame capture (FBC): '+F(en.fbc_sessions)+'</span>':'')+'</div>';
  if(sd.length){h+='<div class="thr">';
    for(const s of sd){h+='<span class="chip on" style="background:var(--accent)">'+
      s.codec+' '+s.width+'×'+s.height+' @'+F(s.fps)+'fps'+
      (s.container?' · '+s.container:'')+'</span>';}
    h+='</div>';}
  // throttle chips
  const anyThr=["sw_power_cap","hw_thermal","sw_thermal","hw_power_brake","sync_boost"].some(k=>th[k]);
  h+='<div class="row"><span class="k">Clock limiters</span>'+
     '<span style="color:'+(anyThr?"var(--orange)":"var(--muted)")+'">'+
     (anyThr?"capping clocks":"none active")+'</span></div>';
  h+='<div class="thr">';
  const names={sw_power_cap:"Power cap",hw_thermal:"Thermal (hardware)",sw_thermal:"Thermal (target)",hw_power_brake:"Power brake",sync_boost:"Sync boost",idle:"Idle (low load)"};
  for(const k in names){h+='<span class="chip'+(th[k]?" on":"")+'">'+names[k]+'</span>';}
  h+='</div></div>';
  return h;
}
async function tick(){
  try{
    const d=await (await fetch('/json',{cache:'no-store'})).json();
    document.getElementById('gpus').innerHTML=(d.gpus||[]).map(gpuCard).join('');
    const tb=document.querySelector('#proc tbody');
    const ps=d.processes||[];
    tb.innerHTML=ps.length?ps.map(p=>'<tr>'+
      '<td>'+(p.container?'<span class="tag">'+p.container+'</span>':(p.container_id||'—'))+'</td>'+
      '<td class="n">'+(p.gpu??0)+'</td>'+
      '<td>'+(p.name||'—')+'</td><td>'+p.pid+'</td>'+
      '<td class="n">'+F(p.used_mb)+'</td><td class="n">'+F(p.sm)+'</td>'+
      '<td class="n">'+F(p.enc)+'</td><td class="n">'+F(p.dec)+'</td></tr>').join('')
      :'<tr><td colspan="8" class="empty">No processes currently using the GPU</td></tr>';
    const gs=d.gpus||[];
    // throttle counters (µs -> human duration)
    const tc=document.querySelector('#throt tbody');
    tc.innerHTML=gs.length?gs.map(g=>{const c=(g.throttle||{}).counters_us||{};return '<tr>'+
      '<td class="n">'+g.index+'</td>'+
      '<td class="n">'+DUR(c.sw_power_cap)+'</td><td class="n">'+DUR(c.sw_thermal)+'</td>'+
      '<td class="n">'+DUR(c.hw_thermal)+'</td><td class="n">'+DUR(c.hw_power_brake)+'</td>'+
      '<td class="n">'+DUR(c.sync_boost)+'</td></tr>';}).join('')
      :'<tr><td colspan="6" class="empty">No GPU detected</td></tr>';
    // device info (static)
    const it=document.querySelector('#info tbody');
    it.innerHTML=gs.length?gs.map(g=>{const i=g.info||{};const D=v=>v||'—';return '<tr>'+
      '<td class="n">'+g.index+'</td>'+
      '<td>'+D(i.name)+'</td><td>'+D(i.vendor)+'</td><td>'+D(i.architecture)+'</td>'+
      '<td>'+D(i.vbios)+'</td><td>'+D(i.inforom_img)+'</td><td>'+D(i.gsp_firmware)+'</td>'+
      '<td>'+D(i.driver_version)+'</td><td>'+D(i.part_number)+'</td><td>'+D(i.serial)+'</td>'+
      '<td class="mono">'+D(i.uuid)+'</td></tr>';}).join('')
      :'<tr><td colspan="11" class="empty">No GPU detected</td></tr>';
    document.getElementById('meta').textContent='updated '+new Date(d.timestamp).toLocaleTimeString()+' · refresh 2s';
  }catch(e){document.getElementById('meta').textContent='fetch error: '+e;}
}
tick();setInterval(tick,2000);
</script>
</body></html>
"""


HELP_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nvidia Status — Help / API</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--edge:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--code:#1f2630}
@media (prefers-color-scheme:light){:root{--bg:#f6f8fa;--card:#fff;--edge:#d0d7de;--fg:#1f2328;--muted:#636c76;--accent:#0969da;--code:#f0f3f6}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--edge);display:flex;gap:14px;align-items:center}
header h1{font-size:16px;margin:0}
a{color:var(--accent)}
main{max-width:960px;margin:0 auto;padding:20px}
section{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:18px 20px;margin-bottom:16px}
h2{font-size:15px;margin:0 0 10px}
h3{font-size:13px;margin:16px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
pre{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;font-size:12px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--edge);vertical-align:top}
th{color:var(--muted);font-weight:500}
.mono{font-family:ui-monospace,monospace;font-size:12px}
</style></head>
<body>
<header><h1>Nvidia Status — Help / API</h1><a href="/">← back to dashboard</a></header>
<main>

<section>
<h2>Endpoints</h2>
<table>
<tr><th>Path</th><th>Method</th><th>Returns</th></tr>
<tr><td class="mono">/</td><td>GET</td><td>Live HTML dashboard</td></tr>
<tr><td class="mono">/metrics</td><td>GET</td><td>Prometheus exposition format</td></tr>
<tr><td class="mono">/json</td><td>GET</td><td>Full JSON snapshot (all fields)</td></tr>
<tr><td class="mono">/power_limit</td><td>GET</td><td>Power-limit state (enabled, current, min/max, latched)</td></tr>
<tr><td class="mono">/power_limit?gpu=N&amp;watts=W&amp;latch=0|1</td><td>POST</td><td>Set board power limit for GPU N (needs ENABLE_POWER_LIMIT=1 + --privileged). latch=1 persists across restarts. GET returns per-GPU state.</td></tr>
<tr><td class="mono">/help</td><td>GET</td><td>This page</td></tr>
</table>
</section>

<section>
<h2>Which service uses what</h2>
<table>
<tr><th>Service</th><th>Use</th><th>How</th></tr>
<tr><td>Prometheus</td><td class="mono">/metrics</td><td>Add a scrape target <code>host:9835</code></td></tr>
<tr><td>Grafana</td><td>via Prometheus <b>or</b> InfluxDB</td><td>Query <code>nvgpu_*</code> metrics; build panels</td></tr>
<tr><td>InfluxDB</td><td>native push (this exporter)</td><td>Set <code>INFLUX_URL</code> (+ v1 <code>INFLUX_DB</code> or v2 <code>INFLUX_TOKEN</code>) — writes line protocol directly, no Telegraf</td></tr>
<tr><td>Netdata</td><td class="mono">/metrics</td><td><code>go.d/prometheus</code> collector → the URL</td></tr>
<tr><td>Telegraf</td><td class="mono">/metrics</td><td><code>[[inputs.prometheus]]</code> urls = the endpoint</td></tr>
<tr><td>VictoriaMetrics</td><td class="mono">/metrics</td><td>vmagent scrape config</td></tr>
<tr><td>Home Assistant</td><td>MQTT (best) / <span class="mono">/metrics</span> / <span class="mono">/json</span></td><td>MQTT auto-discovery (set <code>MQTT_HOST</code>); or Prometheus integration; or a REST sensor</td></tr>
<tr><td>Homepage / Dashy</td><td class="mono">/json</td><td>Custom-API widget → pick fields with a JSONPath</td></tr>
<tr><td>Node-RED / n8n</td><td class="mono">/json</td><td>HTTP-request node, parse JSON</td></tr>
<tr><td>Glances</td><td>plugin</td><td>External <code>-P</code> plugin reads <span class="mono">/json</span> (see project docs)</td></tr>
</table>
</section>

<section>
<h2>JSON (<span class="mono">/json</span>)</h2>
<p>One object per poll: <code>{timestamp, gpus:[...], processes:[...]}</code>. Each GPU:</p>
<pre>{
 "index":0, "uuid":"GPU-…", "name":"…", "pstate":"P2",
 "temp":   {core, junction, vram, target, slowdown, shutdown, max_op},   // °C  (junction/vram = BAR0)
 "util":   {gpu, memory, encoder, decoder},                             // %
 "clocks_mhz": {graphics, sm, memory, video, graphics_max, memory_max},
 "power_w":{draw, limit, enforced_limit, default_limit, min_limit, max_limit},
 "memory_mb": {total, used, free},   // MB (converted from nvidia-smi MiB)
 "fan_pct": 0,
 "pcie":   {gen_current, gen_max, width_current, width_max, rx_mbps, tx_mbps}, // rx=upload, tx=download
 "throttle": {sw_power_cap, hw_thermal, sw_thermal, hw_power_brake, sync_boost, idle}, // 0/1
 "encode": {sessions, avg_fps, avg_latency_us, util, jpeg_util, ofa_util,
            fbc_sessions, fbc_fps, fbc_latency_us,
            sessions_detail:[{pid, codec, width, height, fps, latency_us, container}]},
 "decode": {util}   // NVIDIA exposes decode utilization only (no per-session/codec)
}
// processes:[{gpu, pid, name, used_mb, sm, enc, dec, container, container_id}]</pre>
</section>

<section>
<h2>Prometheus (<span class="mono">/metrics</span>)</h2>
<p>All gauges, labelled <code>gpu</code>,<code>uuid</code>,<code>name</code> (per-session/process add more labels):</p>
<table>
<tr><th>Metric</th><th>Meaning</th></tr>
<tr><td class="mono">nvgpu_temp_{core,junction,vram,target,slowdown,shutdown,max_op}_celsius</td><td>temps + thresholds</td></tr>
<tr><td class="mono">nvgpu_util_{gpu,memory,encoder,decoder}_ratio</td><td>engine utilisation %</td></tr>
<tr><td class="mono">nvgpu_clock_{graphics,sm,memory,video,graphics_max,memory_max}_mhz</td><td>clocks</td></tr>
<tr><td class="mono">nvgpu_power_{draw,limit,enforced_limit,default_limit,min_limit,max_limit}_watts</td><td>power</td></tr>
<tr><td class="mono">nvgpu_memory_{total,used,free}_megabytes</td><td>VRAM (MB)</td></tr>
<tr><td class="mono">nvgpu_fan_ratio</td><td>fan %</td></tr>
<tr><td class="mono">nvgpu_pcie_{gen_current,gen_max,width_current,width_max}</td><td>PCIe link</td></tr>
<tr><td class="mono">nvgpu_pcie_{rx,tx}_megabytes_per_second</td><td>upload / download bandwidth</td></tr>
<tr><td class="mono">nvgpu_throttle_{sw_power_cap,hw_thermal,sw_thermal,hw_power_brake,sync_boost,idle}</td><td>clock limiter active 0/1</td></tr>
<tr><td class="mono">nvgpu_encode_{sessions,avg_fps,avg_latency_us,util_ratio}</td><td>NVENC aggregate</td></tr>
<tr><td class="mono">nvgpu_decode_util_ratio</td><td>NVDEC %</td></tr>
<tr><td class="mono">nvgpu_fbc_sessions</td><td>frame-capture sessions</td></tr>
<tr><td class="mono">nvgpu_encode_session_fps</td><td>per session; labels <code>codec,resolution,container,pid</code></td></tr>
<tr><td class="mono">nvgpu_process_memory_megabytes / _sm_ratio / _enc_ratio / _dec_ratio</td><td>per process; labels <code>pid,proc,container,container_id</code></td></tr>
</table>
</section>

<section>
<h2>MQTT + Home Assistant</h2>
<p>Set <code>MQTT_HOST</code> (+ <code>MQTT_USER</code>/<code>MQTT_PASS</code> if needed). The exporter publishes:</p>
<table>
<tr><th>Topic</th><th>Payload</th></tr>
<tr><td class="mono">nvgpu/&lt;node&gt;/status</td><td><code>online</code> / <code>offline</code> (LWT)</td></tr>
<tr><td class="mono">nvgpu/&lt;node&gt;/gpu&lt;i&gt;/state</td><td>JSON: core, junction, vram, power_draw, power_limit, util_gpu, mem_used/total, pcie_rx/tx, fan, clock_graphics, throttle_thermal, encode_sessions, encode_util, decode_util, encode_codec</td></tr>
<tr><td class="mono">homeassistant/sensor/nvgpu_&lt;node&gt;_gpu&lt;i&gt;_&lt;key&gt;/config</td><td>HA discovery — sensors auto-appear grouped under one device</td></tr>
</table>
<p><code>&lt;node&gt;</code> = <code>NODE_NAME</code> or the container hostname. Discovery prefix = <code>MQTT_DISCOVERY_PREFIX</code> (default <code>homeassistant</code>).</p>
</section>

<section>
<h2>InfluxDB (native push → Grafana)</h2>
<p>Set <code>INFLUX_URL</code> and the exporter writes line protocol directly (no Telegraf). Measurements: <code>nvgpu</code> (per-GPU), <code>nvgpu_process</code> (per-container), <code>nvgpu_encode_session</code> (per encode session).</p>
<h3>InfluxDB 1.x</h3>
<pre>INFLUX_URL=http://influx-host:8086
INFLUX_DB=gpu
INFLUX_USER=…   INFLUX_PASS=…      # if auth is on
# first create the DB:  curl -XPOST 'http://influx-host:8086/query' \
#   --data-urlencode 'q=CREATE DATABASE gpu' -u user:pass</pre>
<h3>InfluxDB 2.x</h3>
<pre>INFLUX_URL=http://influx-host:8086
INFLUX_TOKEN=…   INFLUX_ORG=…   INFLUX_BUCKET=gpu</pre>
<p>Optional <code>INFLUX_INTERVAL</code> (seconds, default 10). Presence of <code>INFLUX_TOKEN</code> selects the v2 API; otherwise v1. Grafana then uses the InfluxDB datasource — tags <code>host, gpu, name, codec, resolution, container</code> are available for filtering.</p>
</section>

</main></body></html>
"""


SETTINGS_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nvidia Status — Settings</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--edge:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--track:#21262d;--green:#3fb950;--orange:#db6d28}
@media (prefers-color-scheme:light){:root{--bg:#f6f8fa;--card:#fff;--edge:#d0d7de;--fg:#1f2328;--muted:#636c76;--accent:#0969da;--track:#eaeef2}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--edge);display:flex;gap:14px;align-items:center}
header h1{font-size:16px;margin:0}
a{color:var(--accent)}
main{max-width:720px;margin:0 auto;padding:20px}
.sec{background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:16px 18px;margin-bottom:14px}
.sec h2{font-size:14px;margin:0 0 12px}
.f{display:flex;align-items:center;gap:12px;margin:9px 0}
.f label{flex:0 0 210px;color:var(--muted);font-size:13px}
.f input[type=text],.f input[type=number],.f input[type=password]{flex:1;padding:7px 9px;background:var(--track);border:1px solid var(--edge);color:var(--fg);border-radius:6px;font-size:13px}
.f input[type=checkbox]{width:18px;height:18px}
.bar{position:sticky;bottom:0;background:var(--card);border:1px solid var(--edge);border-radius:10px;padding:12px 16px;display:flex;gap:12px;align-items:center;margin-top:4px}
button{padding:8px 16px;border:0;border-radius:6px;cursor:pointer;font-size:13px}
.primary{background:var(--accent);color:#fff}
.ghost{background:var(--track);color:var(--fg);border:1px solid var(--edge)}
#msg{font-size:12px;color:var(--muted)}
.note{font-size:12px;color:var(--muted);margin:2px 0 10px}
code{background:var(--track);padding:1px 5px;border-radius:4px;font-size:12px}
</style></head>
<body>
<header><h1>Nvidia Status — Settings</h1><a href="/">← dashboard</a><a href="/help">help / API</a></header>
<main>
<div class="note">Settings are saved to <code>/config/settings.json</code> (mount a volume at <code>/config</code> to persist) and <b>override the docker env vars</b>. Changes apply after a container restart. Leave a password blank to keep the current one.</div>
<form id="form"></form>
<div class="sec">
  <h2>Power-limit control</h2>
  <div class="note" id="pl-sub" style="margin-top:-4px">loading…</div>
  <div id="pl-list"></div>
</div>
<div class="bar">
  <button class="primary" id="save">Save</button>
  <button class="ghost" id="saver">Save &amp; restart now</button>
  <span id="msg"></span>
</div>
</main>
<script>
const FIELDS=[
 ['General','NODE_NAME','Node name (tag/topic id)','text','YourServerHere'],
 ['General','INTERVAL','Poll interval (seconds)','number','5'],
 ['Servers (HTTP endpoints)','ENABLE_DASHBOARD','Dashboard  /','bool'],
 ['Servers (HTTP endpoints)','ENABLE_METRICS','Prometheus  /metrics','bool'],
 ['Servers (HTTP endpoints)','ENABLE_JSON','JSON API  /json','bool'],
 ['MQTT / Home Assistant','ENABLE_MQTT','Enable MQTT publisher','bool'],
 ['MQTT / Home Assistant','MQTT_HOST','Broker host','text','e.g. 192.168.1.x'],
 ['MQTT / Home Assistant','MQTT_PORT','Broker port','number','1883'],
 ['MQTT / Home Assistant','MQTT_USER','Username','text'],
 ['MQTT / Home Assistant','MQTT_PASS','Password','secret'],
 ['MQTT / Home Assistant','MQTT_DISCOVERY_PREFIX','HA discovery prefix','text','homeassistant'],
 ['InfluxDB','ENABLE_INFLUX','Enable InfluxDB writer','bool'],
 ['InfluxDB','INFLUX_URL','URL','text','http://host:8086'],
 ['InfluxDB','INFLUX_INTERVAL','Write interval (seconds)','number','10'],
 ['InfluxDB','INFLUX_DB','Database (v1)','text','gpu'],
 ['InfluxDB','INFLUX_USER','User (v1)','text'],
 ['InfluxDB','INFLUX_PASS','Password (v1)','secret'],
 ['InfluxDB','INFLUX_TOKEN','Token (v2)','secret'],
 ['InfluxDB','INFLUX_ORG','Org (v2)','text'],
 ['InfluxDB','INFLUX_BUCKET','Bucket (v2)','text'],
];
const NOTES={
 'General':'Identity and timing. <b>Node name</b> tags this host in MQTT topics, InfluxDB rows and the Home Assistant device name (default = container hostname). <b>Poll interval</b> is how often stats are refreshed.',
 'Servers (HTTP endpoints)':'Turn each HTTP endpoint on/off — disable what you don\'t use. See <a href="/help">Help / API</a> for what each one serves and which apps consume it.',
 'MQTT / Home Assistant':'Publishes metrics to an MQTT broker with <b>Home Assistant auto-discovery</b> (sensors appear automatically). Leave the host blank or untick to disable. Full topic/field list in <a href="/help">Help / API</a>.',
 'InfluxDB':'Writes line protocol directly to InfluxDB for Grafana — v1 (URL + DB + user/pass) or v2 (URL + token + org + bucket). Setup details and the measurement/tag list are in <a href="/help">Help / API</a>.',
};
let STATE={};
function render(){
 const form=document.getElementById('form'); form.innerHTML='';
 let cur=null,box=null;
 for(const [sec,k,label,type,ph] of FIELDS){
  if(sec!==cur){cur=sec;box=document.createElement('div');box.className='sec';box.innerHTML='<h2>'+sec+'</h2>'+(NOTES[sec]?'<div class="note" style="margin-top:-4px">'+NOTES[sec]+'</div>':'');form.appendChild(box);}
  const row=document.createElement('div');row.className='f';
  const eff=STATE.effective||{};
  if(type==='bool'){
   row.innerHTML='<label for="'+k+'">'+label+'</label><input type="checkbox" id="'+k+'" '+(eff[k]?'checked':'')+'>';
  }else if(type==='secret'){
   const set=(STATE.secret_set||{})[k];
   row.innerHTML='<label for="'+k+'">'+label+'</label><input type="password" id="'+k+'" placeholder="'+(set?'(set — blank keeps it)':'(not set)')+'">';
  }else{
   row.innerHTML='<label for="'+k+'">'+label+'</label><input type="'+(type||'text')+'" id="'+k+'" value="'+(eff[k]??'')+'" placeholder="'+(ph||'')+'">';
  }
  box.appendChild(row);
 }
}
async function load(){STATE=await (await fetch('/api/settings',{cache:'no-store'})).json();render();}
async function save(restart){
 const body={_restart:restart};
 for(const [sec,k,label,type] of FIELDS){
  const el=document.getElementById(k); if(!el)continue;
  if(type==='bool'){body[k]=el.checked?'1':'0';}
  else if(type==='secret'){ if(el.value!=='')body[k]=el.value; }   // blank = keep
  else{ body[k]=el.value; }                                        // blank = clear override
 }
 const msg=document.getElementById('msg'); msg.textContent='saving…';
 try{
  const r=await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();
  msg.textContent=r.ok?(restart?'saved — restarting…':j.message):('error: '+JSON.stringify(j));
  if(!restart) load();
 }catch(e){msg.textContent='error: '+e;}
}
document.getElementById('save').addEventListener('click',e=>{e.preventDefault();save(false);});
document.getElementById('saver').addEventListener('click',e=>{e.preventDefault();if(confirm('Save and restart the container now?'))save(true);});
load();

// --- power-limit control (per-GPU; moved here from the dashboard) ---
const Fw=v=>v==null?'—':Math.round(v);
async function loadPL(){
 try{
  const s=await (await fetch('/power_limit',{cache:'no-store'})).json();
  const sub=document.getElementById('pl-sub'), list=document.getElementById('pl-list');
  if(!s.enabled){sub.innerHTML='Read-only. Enable by running the container with <code>ENABLE_POWER_LIMIT=1</code> and <code>--privileged</code> (mount <code>/config</code> so latches persist).';list.innerHTML='';return;}
  const gpus=s.gpus||[];
  if(!gpus.length){sub.textContent=s.reason||'No GPU detected.';list.innerHTML='';return;}
  sub.innerHTML='Set the board power limit'+(gpus.length>1?' per GPU':'')+'. Lower = cooler/quieter, less peak performance. <b>Latch</b> re-applies on restart.';
  list.innerHTML='';
  for(const g of gpus){
   const row=document.createElement('div'); row.style.cssText='padding:6px 0;border-top:1px solid var(--edge)';
   if(!g.available){
    row.innerHTML='<div class="note" style="margin:6px 0">⚠️ <b style="color:var(--orange)">GPU '+g.index+'</b>: '+(g.reason||'control unavailable')+'</div>';
    list.appendChild(row); continue;
   }
   row.innerHTML=
    '<div class="f"><label>GPU '+g.index+(gpus.length>1?' · '+(g.name||''):'')+'</label>'+
    '<span>current <b>'+Fw(g.current)+'</b> W · default '+Fw(g.default)+' W · range '+g.min+'–'+g.max+' W'+(g.latched!=null?' · <b>latched '+g.latched+' W</b>':'')+'</span></div>'+
    '<div class="f"><label>Set watts</label>'+
    '<input type="number" class="pl-w" data-gpu="'+g.index+'" value="'+Fw(g.current)+'" min="'+g.min+'" max="'+g.max+'" style="flex:0 0 120px">'+
    '<label style="flex:0 0 auto;color:var(--fg);cursor:pointer"><input type="checkbox" class="pl-l" data-gpu="'+g.index+'" '+(g.latched!=null?'checked':'')+'> latch</label>'+
    '<button class="primary pl-a" data-gpu="'+g.index+'">Apply</button>'+
    '<span class="pl-m" data-gpu="'+g.index+'" style="font-size:12px;color:var(--muted)"></span></div>';
   list.appendChild(row);
  }
  list.querySelectorAll('.pl-a').forEach(btn=>btn.addEventListener('click',async e=>{
   e.preventDefault();
   const gi=btn.dataset.gpu;
   const w=list.querySelector('.pl-w[data-gpu="'+gi+'"]').value;
   const latch=list.querySelector('.pl-l[data-gpu="'+gi+'"]').checked?1:0;
   const msg=list.querySelector('.pl-m[data-gpu="'+gi+'"]');
   if(!confirm('Set GPU '+gi+' power limit to '+w+' W'+(latch?' and LATCH it (re-apply on every restart)?':'?')))return;
   msg.textContent='applying…';
   try{const r=await fetch('/power_limit?gpu='+gi+'&watts='+encodeURIComponent(w)+'&latch='+latch,{method:'POST'});msg.textContent=(await r.text()).trim();loadPL();}
   catch(err){msg.textContent='error: '+err;}
  }));
 }catch(e){}
}
loadPL();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/dashboard"):
            if not cfg_bool("ENABLE_DASHBOARD"):
                self._send(404, "dashboard disabled\n")
                return
            self._send(200, DASHBOARD_HTML, "text/html; charset=utf-8")
        elif p == "/help":
            self._send(200, HELP_HTML, "text/html; charset=utf-8")
        elif p == "/settings":
            self._send(200, SETTINGS_HTML, "text/html; charset=utf-8")
        elif p == "/api/settings":
            self._send(200, json.dumps(settings_state()), "application/json")
        elif p == "/metrics":
            if not cfg_bool("ENABLE_METRICS"):
                self._send(404, "metrics endpoint disabled\n")
                return
            self._send(200, prom_metrics())
        elif p == "/json":
            if not cfg_bool("ENABLE_JSON"):
                self._send(404, "json endpoint disabled\n")
                return
            self._send(200, json.dumps(collect()), "application/json")
        elif p == "/power_limit":
            self._send(200, json.dumps(power_state()), "application/json")
        else:
            self._send(404, "not found\n")

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/power_limit":
            if not ENABLE_PL:
                self._send(403, "power-limit control disabled "
                                "(run with ENABLE_POWER_LIMIT=1 and --privileged)\n")
                return
            q = parse_qs(p.query)
            lraw = (q.get("latch") or [None])[0]
            latch = True if lraw == "1" else False if lraw == "0" else None
            try:
                gpu_idx = int((q.get("gpu") or ["0"])[0])
            except ValueError:
                gpu_idx = 0
            code, msg = set_power_limit((q.get("watts") or [None])[0], latch, gpu_idx)
            self._send(code, msg + "\n")
        elif p.path == "/api/settings":
            length = int(self.headers.get("Content-Length") or 0)
            try:
                new = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._send(400, "invalid json\n")
                return
            restart = bool(new.pop("_restart", False))
            try:
                save_settings(new)
            except OSError as e:
                self._send(500, f"could not save (mount a volume at {STATE_DIR}): {e}\n")
                return
            self._send(200, json.dumps({"ok": True,
                       "message": "Saved. Restart the container to apply."}),
                       "application/json")
            if restart:
                threading.Thread(target=lambda: (time.sleep(0.4), os._exit(0)),
                                 daemon=True).start()
        else:
            self._send(404, "not found\n")

    def log_message(self, *a):
        pass  # quiet


# ---------------- MQTT + Home Assistant auto-discovery (optional) ----------------
MQTT_HOST = cfg("MQTT_HOST")
NODE = re.sub(r"[^a-z0-9_]", "_",
              (cfg("NODE_NAME") or os.environ.get("HOSTNAME") or "nvgpu").lower())

# (key, friendly name, unit, device_class, state_class, icon)
MQTT_SENSORS = [
    ("core", "Core Temp", "°C", "temperature", "measurement", None),
    ("junction", "Junction Temp", "°C", "temperature", "measurement", None),
    ("vram", "VRAM Temp", "°C", "temperature", "measurement", None),
    ("power_draw", "Power Draw", "W", "power", "measurement", None),
    ("power_limit", "Power Limit", "W", "power", None, None),
    ("util_gpu", "GPU Utilisation", "%", None, "measurement", "mdi:expansion-card"),
    ("mem_used", "VRAM Used", "MB", None, "measurement", "mdi:memory"),
    ("mem_total", "VRAM Total", "MB", None, None, "mdi:memory"),
    ("pcie_rx", "PCIe Upload", "MB/s", "data_rate", "measurement", "mdi:upload"),
    ("pcie_tx", "PCIe Download", "MB/s", "data_rate", "measurement", "mdi:download"),
    ("fan", "Fan", "%", None, "measurement", "mdi:fan"),
    ("clock_graphics", "Graphics Clock", "MHz", "frequency", "measurement", None),
    ("throttle_thermal", "Thermal Throttle", None, None, None, "mdi:thermometer-alert"),
    ("encode_sessions", "Encode Sessions", None, None, "measurement", "mdi:movie-open"),
    ("encode_util", "Encode Utilisation", "%", None, "measurement", "mdi:movie-open-play"),
    ("decode_util", "Decode Utilisation", "%", None, "measurement", "mdi:movie-play"),
    ("encode_codec", "Encode Codec", None, None, None, "mdi:video"),
]


def _mqtt_payload(g):
    return {
        "core": g["temp"]["core"], "junction": g["temp"]["junction"],
        "vram": g["temp"]["vram"],
        "power_draw": g["power_w"]["draw"], "power_limit": g["power_w"]["limit"],
        "util_gpu": g["util"]["gpu"],
        "mem_used": g["memory_mb"]["used"], "mem_total": g["memory_mb"]["total"],
        "pcie_rx": g["pcie"]["rx_mbps"], "pcie_tx": g["pcie"]["tx_mbps"],
        "fan": g["fan_pct"], "clock_graphics": g["clocks_mhz"]["graphics"],
        "throttle_thermal": "ON" if (g["throttle"]["hw_thermal"]
                                     or g["throttle"]["sw_thermal"]) else "OFF",
        "encode_sessions": g["encode"]["sessions"],
        "encode_util": g["encode"]["util"],
        "decode_util": g["decode"]["util"],
        "encode_codec": (g["encode"]["sessions_detail"][0]["codec"]
                         if g["encode"]["sessions_detail"] else "none"),
    }


def mqtt_loop():
    import paho.mqtt.client as mqtt
    host = cfg("MQTT_HOST")
    port = int(cfg("MQTT_PORT", "1883"))
    disc = cfg("MQTT_DISCOVERY_PREFIX", "homeassistant")
    interval = float(cfg("MQTT_INTERVAL", cfg("INTERVAL", "5")))
    base = f"nvgpu/{NODE}"
    avail = f"{base}/status"
    cli = mqtt.Client(client_id=f"nvgpu-exporter-{NODE}")
    if cfg("MQTT_USER"):
        cli.username_pw_set(cfg("MQTT_USER"), cfg("MQTT_PASS"))
    cli.will_set(avail, "offline", retain=True)
    while True:
        try:
            cli.connect(host, port, 60)
            break
        except OSError as e:
            sys.stderr.write(f"mqtt: connect {host}:{port} failed ({e}); retry 10s\n")
            time.sleep(10)
    cli.loop_start()
    cli.publish(avail, "online", retain=True)
    sys.stderr.write(f"mqtt: connected {host}:{port}, discovery under {disc}/\n")
    discovered = set()
    while True:
        try:
            snap = collect()
            for g in snap["gpus"]:
                idx = g["index"]
                st = f"{base}/gpu{idx}/state"
                if idx not in discovered:
                    dev = {"identifiers": [f"nvgpu_{NODE}_gpu{idx}"],
                           "name": f"{g['name']} ({NODE} GPU{idx})",
                           "manufacturer": "NVIDIA", "model": g["name"]}
                    for key, name, unit, dclass, sclass, icon in MQTT_SENSORS:
                        uid = f"nvgpu_{NODE}_gpu{idx}_{key}"
                        dcfg = {"name": name, "unique_id": uid, "state_topic": st,
                                "value_template": "{{ value_json.%s }}" % key,
                                "availability_topic": avail, "device": dev}
                        if unit:
                            dcfg["unit_of_measurement"] = unit
                        if dclass:
                            dcfg["device_class"] = dclass
                        if sclass:
                            dcfg["state_class"] = sclass
                        if icon:
                            dcfg["icon"] = icon
                        cli.publish(f"{disc}/sensor/{uid}/config",
                                    json.dumps(dcfg), retain=True)
                    discovered.add(idx)
                cli.publish(st, json.dumps(_mqtt_payload(g)))
        except Exception as e:  # keep the thread alive
            sys.stderr.write(f"mqtt: publish error {e}\n")
        time.sleep(interval)


# ---------------- InfluxDB line-protocol writer (optional) ----------------
INFLUX_URL = cfg("INFLUX_URL")


def _esc(s):
    return (str(s).replace("\\", "\\\\").replace(" ", "\\ ")
            .replace(",", "\\,").replace("=", "\\="))


def influx_lines(snap, node):
    ts = snap.get("timestamp")   # ms
    out = []

    def line(meas, tags, fields):
        fs = ",".join(f"{k}={float(v)}" for k, v in fields.items() if v is not None)
        if not fs:
            return
        tg = "".join("," + _esc(k) + "=" + _esc(v)
                     for k, v in tags.items() if v not in (None, ""))
        out.append(f"{meas}{tg} {fs} {ts}")

    for g in snap.get("gpus", []):
        t, p, u, c, m = g["temp"], g["power_w"], g["util"], g["clocks_mhz"], g["memory_mb"]
        pc, th, en, de = g["pcie"], g["throttle"], g["encode"], g["decode"]
        line("nvgpu",
             {"host": node, "gpu": g["index"], "uuid": g.get("uuid"), "name": g.get("name")},
             {"temp_core": t["core"], "temp_junction": t["junction"], "temp_vram": t["vram"],
              "power_draw": p["draw"], "power_limit": p["limit"],
              "util_gpu": u["gpu"], "util_memory": u["memory"],
              "util_encoder": u["encoder"], "util_decoder": u["decoder"],
              "clock_graphics": c["graphics"], "clock_memory": c["memory"],
              "mem_used_mb": m["used"], "mem_total_mb": m["total"], "mem_free_mb": m["free"],
              "fan_pct": g["fan_pct"],
              "pcie_rx_mbps": pc["rx_mbps"], "pcie_tx_mbps": pc["tx_mbps"],
              "pcie_gen": pc["gen_current"], "pcie_width": pc["width_current"],
              "encode_sessions": en["sessions"], "encode_util": en["util"],
              "decode_util": de["util"], "fbc_sessions": en["fbc_sessions"],
              "throttle_sw_power_cap": th["sw_power_cap"],
              "throttle_hw_thermal": th["hw_thermal"],
              "throttle_sw_thermal": th["sw_thermal"]})
        for s in en.get("sessions_detail", []):
            line("nvgpu_encode_session",
                 {"host": node, "gpu": g["index"], "pid": s["pid"], "codec": s["codec"],
                  "resolution": f'{s["width"]}x{s["height"]}', "container": s.get("container")},
                 {"fps": s["fps"], "latency_us": s["latency_us"]})
    for pr in snap.get("processes", []):
        line("nvgpu_process",
             {"host": node, "gpu": pr.get("gpu"), "pid": pr["pid"],
              "proc": pr["name"], "container": pr.get("container")},
             {"mem_mb": pr["used_mb"], "sm": pr["sm"], "enc": pr["enc"], "dec": pr["dec"]})
    return "\n".join(out)


def influx_loop():
    import urllib.request
    import urllib.parse
    v2 = bool(cfg("INFLUX_TOKEN"))
    interval = float(cfg("INFLUX_INTERVAL", cfg("INTERVAL", "10")))
    base = cfg("INFLUX_URL").rstrip("/")
    if v2:
        q = {"org": cfg("INFLUX_ORG", ""),
             "bucket": cfg("INFLUX_BUCKET", "gpu"), "precision": "ms"}
        url = base + "/api/v2/write?" + urllib.parse.urlencode(q)
        headers = {"Authorization": "Token " + cfg("INFLUX_TOKEN"),
                   "Content-Type": "text/plain"}
    else:
        q = {"db": cfg("INFLUX_DB", "gpu"), "precision": "ms"}
        if cfg("INFLUX_USER"):
            q["u"] = cfg("INFLUX_USER")
            q["p"] = cfg("INFLUX_PASS", "")
        url = base + "/write?" + urllib.parse.urlencode(q)
        headers = {"Content-Type": "text/plain"}
    sys.stderr.write(f"influx: writing to {base} ({'v2' if v2 else 'v1'}) every {interval}s\n")
    while True:
        try:
            body = influx_lines(collect(), NODE).encode()
            if body:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:
            sys.stderr.write(f"influx: write error {e}\n")
        time.sleep(interval)


def main():
    apply_latch_on_startup()
    if cfg_bool("ENABLE_MQTT", True) and cfg("MQTT_HOST"):
        threading.Thread(target=mqtt_loop, daemon=True).start()
    if cfg_bool("ENABLE_INFLUX", True) and cfg("INFLUX_URL"):
        threading.Thread(target=influx_loop, daemon=True).start()
    mode = os.environ.get("MODE", "json")
    interval = float(cfg("INTERVAL", "5"))
    port = int(os.environ.get("PORT", "9835"))
    if mode == "prometheus":
        srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        sys.stderr.write(
            f"nvgpu-exporter: http on :{port}  "
            f"dashboard={cfg_bool('ENABLE_DASHBOARD')} metrics={cfg_bool('ENABLE_METRICS')} "
            f"json={cfg_bool('ENABLE_JSON')} power-control={'ON' if ENABLE_PL else 'off'}\n")
        srv.serve_forever()
    else:
        while True:
            sys.stdout.write(json.dumps(collect()) + "\n")
            sys.stdout.flush()
            if "--once" in sys.argv:
                break
            time.sleep(interval)


if __name__ == "__main__":
    main()

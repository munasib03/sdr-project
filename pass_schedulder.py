#!/usr/bin/env python3
"""
Satellite Pass Scheduler — SSEC ACS1 — B206mini-i (UHD 4.10, Apptainer)
Drives the validated uhd_record.py recorder at each pass time.

Per-satellite RF profiles handle the antenna-side downconverter automatically:
    IF tuned by B206 = rf_freq - downconv_lo

The X-band downconverter LO tracks the target: EVERY X-band downlink lands at
720.0 MHz IF, so downconv_lo_hz is always rf_freq_hz - 720 MHz. A per-satellite
LO that isn't 7092 is therefore expected, not a mistake -- it just means that
satellite isn't at 7812. L-band (Metop AHRPT) uses a fixed 1556.0 MHz LO.
Metop-B    : 1701.3 - 1556.0 = 145.3 MHz IF  (METOP AHRPT)
Metop-C    : 1701.3 - 1556.0 = 145.3 MHz IF  (METOP AHRPT)
METOP-SGA1 : 7825.0 - 7105.0 = 720.0 MHz IF  (EPS-SG DDB)
NOAA-20    : 7812.0 - 7092.0 = 720.0 MHz IF  (NPP HRD)
NOAA-21    : 7812.0 - 7092.0 = 720.0 MHz IF  (JPSS HRD)
SNPP	   : 7812.0 - 7092.0 = 720.0 MHz IF  (NPP HRD)
FENGYUN-3G : 7812.0 - 7092.0 = 720.0 MHz IF  (FY-3G MERSI)
"""
import subprocess, time, os, json, logging, shutil
from datetime import datetime, timezone, timedelta

# ── PATHS (verify both) ──────────────────────────────────────────────────────
SANDBOX     = "/home/ilham/uhd/uhd_sandbox/"     # CHECK
RECORDER    = "/home/ilham/uhd_record.py"        # CHECK absolute path
OUTPUT_DIR  = "/home/ilham/recordings"
PASSES_FILE = "/home/ilham/passes.json"          # CHECK — pass schedule, see load_passes()

# ── RECORDING BEHAVIOUR ──────────────────────────────────────────────────────
PRE_AOS_BUFFER  = 30       # start this many seconds before AOS
POST_LOS_BUFFER = 30       # keep recording this many seconds after LOS
NUM_RECV_FRAMES = 1000     # host ring buffer; larger absorbs shared-server stalls
DISK_MARGIN     = 1.1      # require 10% headroom over the projected file size

# ── PER-SATELLITE RF PROFILES ────────────────────────────────────────────────
# IF (Intermediate Frequency) the radio actually tunes to = rf_freq_hz - downconv_lo_hz
SAT_PROFILES = {
    "metop-b": {
	    "rf_freq_hz":	  1701.3e6,
	    "downconv_lo_hz": 1556.0e6,
	    "sample_rate":	  7e6,        # -> 145.3 MHz IF; sps = 3.0 at 7/3 Msym/s
	    "format":	      "cf32",     # metop_ahrpt pipeline
	    "gain_db":	      40,
	    "ant":		      "RX2",
    },
    "metop-c": {
        "rf_freq_hz":     1701.3e6,
        "downconv_lo_hz": 1556.0e6,   # -> 145.3 MHz IF
        "sample_rate":    7e6,        # sps = 3.0 at 7/3 Msym/s
        "format":         "cf32",     # metop_ahrpt pipeline
        "gain_db":        40,
        "ant":            "RX2",
    },
    "metop-sga1": {
        "rf_freq_hz":     7825.0e6,
        "downconv_lo_hz": 7105.0e6,   # -> 720.0 MHz IF
        "sample_rate":    56e6,
        "format":         "s8",       # 61.44 Msps only fits the bus/disk as 8-bit
        "gain_db":        30,
        "ant":            "RX2",
    },
    "noaa-20": {
        "rf_freq_hz":     7812.0e6,
        "downconv_lo_hz": 7092.0e6,   # -> 720.0 MHz IF
        "sample_rate":    24e6,
        "format":         "cs16",     # HRD ~15 Msps — verify USB can sustain, prefer cs16
        "gain_db":        30,
        "ant":            "RX2",
    },
    "noaa-21": {
        "rf_freq_hz":     7812.0e6,
        "downconv_lo_hz": 7092.0e6,   # -> 720.0 MHz IF
        "sample_rate":    40e6,
        "format":         "cs16",     
        "gain_db":        30,
        "ant":            "RX2",
    },
    "snpp": {
        "rf_freq_hz":	  7812.0e6,
        "downconv_lo_hz": 7092.0e6,
        "sample_rate":	  24e6,
        "format":	      "cs16",
        "gain_db":	      30,
        "ant":		      "RX2",
    },
    "fengyun-3g": {
        "rf_freq_hz":     7812.0e6,
        "downconv_lo_hz": 7092.0e6,   # -> 720.0 MHz IF
        "sample_rate":    16e6,
        "format":         "cs16",
        "gain_db":        30,
        "ant":            "RX2",
    },
}

# ── BASEBAND FORMATS ─────────────────────────────────────────────────────────
# bytes/sample on disk, plus the file extension. The extension doubles as the
# SatDump --baseband_format token (verified accepted by satdump v1.2.2:
# cf32, cs16, s8 -- note cf_32/cs_16 are NOT accepted).
# NOTE: a baseband .s8 here is complex int8 IQ. Not to be confused with the
# soft-symbol .s8 files gr_demod_tail.py writes into the same directory;
# those always carry a _live suffix.
FORMATS = {
    "cf32": {"bytes": 8, "ext": ".cf32"},
    "cs16": {"bytes": 4, "ext": ".cs16"},
    "s8":   {"bytes": 2, "ext": ".s8"},
}

# ── PASS SCHEDULE (UTC) ──────────────────────────────────────────────────────
# Loaded from PASSES_FILE (JSON array of {"sat","norad_id","aos","los","max_el"}).
# Entries with "enabled": false are kept in the file but skipped here.
def load_passes(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Pass schedule not found: {path}")
    with open(path) as f:
        data = json.load(f)
    required = {"sat", "norad_id", "aos", "los", "max_el"}
    passes = []
    for i, p in enumerate(data):
        if not p.get("enabled", True):
            continue
        missing = required - p.keys()
        if missing:
            raise SystemExit(f"{path}: entry {i} missing keys {missing}: {p}")
        passes.append(p)
    return passes

PASSES = load_passes(PASSES_FILE)

# ── LOGGING ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s UTC] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(OUTPUT_DIR, "scheduler.log"))],
)
log = logging.getLogger(__name__)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def parse_utc(s):      return datetime.fromisoformat(s.replace("Z", "+00:00"))
def now_utc():         return datetime.now(timezone.utc)
def seconds_until(dt): return (dt - now_utc()).total_seconds()

def record_pass(p):
    prof = SAT_PROFILES.get(p["sat"])
    if prof is None:
        log.error(f"No RF profile for '{p['sat']}' — skipping.")
        return None

    if_hz = prof["rf_freq_hz"] - prof["downconv_lo_hz"]
    rate  = prof["sample_rate"]
    fmt   = prof.get("format", "cf32")
    gain  = prof["gain_db"]
    ant   = prof.get("ant")

    finfo = FORMATS.get(fmt)
    if finfo is None:
        log.error(f"Unknown baseband format '{fmt}' for '{p['sat']}' "
                  f"(known: {', '.join(FORMATS)}) — skipping.")
        return None

    aos = parse_utc(p["aos"]); los = parse_utc(p["los"])
    duration = int((los - aos).total_seconds()) + PRE_AOS_BUFFER + POST_LOS_BUFFER
    ext   = finfo["ext"]
    fname = os.path.join(OUTPUT_DIR,
                         f"{p['sat'].replace('-','_')}_{aos.strftime('%Y%m%d_%H%M')}{ext}")

    # ── disk space pre-flight ────────────────────────────────────────────────
    need = duration * rate * finfo["bytes"]
    free = shutil.disk_usage(OUTPUT_DIR).free
    if free < need * DISK_MARGIN:
        log.error(f"Insufficient disk: need {need/1e9:.1f} GB "
                  f"(+{int((DISK_MARGIN-1)*100)}% margin), have {free/1e9:.1f} GB "
                  f"— skipping {fname}")
        return None
    log.info(f"Disk pre-flight OK: need {need/1e9:.1f} GB, have {free/1e9:.1f} GB free")

    cmd = [
        "apptainer", "exec", SANDBOX,
        "python3", RECORDER,
        "--freq",   str(if_hz),
        "--rate",   str(rate),
        "--gain",   str(gain),
        "--duration", str(duration),
        "--output", fname,
        "--format", fmt,
        "--num-recv-frames", str(NUM_RECV_FRAMES),
    ]
    if ant:
        cmd += ["--ant", ant]

    log.info("─────────────────────────────────────────")
    log.info(f"SAT {p['sat'].upper()} | AOS {aos:%H:%M:%S} LOS {los:%H:%M:%S} UTC | MaxEl {p['max_el']:.1f}°")
    log.info(f"RF {prof['rf_freq_hz']/1e6:.1f} MHz - LO {prof['downconv_lo_hz']/1e6:.1f} MHz "
             f"= IF {if_hz/1e6:.1f} MHz | {rate/1e6:.2f} Msps | gain {gain} dB | {duration}s")
    log.info(f"FORMAT {fmt} ({finfo['bytes']} B/samp) | "
             f"disk ~{rate*finfo['bytes']/1e6:.1f} MB/s | need {need/1e9:.1f} GB")
    log.info(f"OUT {fname}")
    log.info(f"CMD {' '.join(cmd)}")
    log.info("─────────────────────────────────────────")

    try:
        subprocess.run(cmd, check=True)     # recorder stdout (incl. overflow count) streams live
        log.info(f"Recording complete (clean): {fname}")
    except subprocess.CalledProcessError as e:
        # uhd_record.py exits nonzero on overflows/timeouts/abort — file may
        # still exist but contain discontinuities (expect deframer sync loss).
        log.error(f"Recording FAILED or LOSSY rc={e.returncode} for {fname}")
    except Exception as e:
        log.exception("Unexpected error")
    return fname

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    log.info("═══ Pass Scheduler — ACS1 B206mini-i ═══")
    log.info(f"Recorder {RECORDER} | {len(PASSES)} passes")

    for p in sorted(PASSES, key=lambda x: parse_utc(x["aos"])):
        aos      = parse_utc(p["aos"])
        start_at = aos - timedelta(seconds=PRE_AOS_BUFFER)

        if seconds_until(start_at) < -60:
            log.warning(f"Skipping {p['sat']} AOS {p['aos']} — already finished")
            continue

        wait = seconds_until(start_at)
        log.info(f"Next: {p['sat'].upper()} AOS {aos:%H:%M:%S}Z MaxEl {p['max_el']:.1f}° "
                 f"| sleeping {int(wait)}s ({wait/60:.1f} min)")

        pre = seconds_until(start_at) - 300     # wake 5 min out
        if pre > 0:
            time.sleep(pre)
            log.info("5 minutes to recording.")

        while True:
            r = seconds_until(start_at)
            if r <= 0:
                break
            if r > 60:
                log.info(f"  {int(r//60)}m {int(r%60)}s to start...")
                time.sleep(60)
            else:
                log.info(f"  {int(r)}s to start...")
                time.sleep(max(0.5, r - 1))

        log.info("AOS imminent — recording now.")
        record_pass(p)

    log.info("All scheduled passes complete.")

if __name__ == "__main__":
    main()

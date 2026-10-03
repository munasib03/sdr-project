#!/usr/bin/env python3
"""
LIVE Pass Scheduler — SSEC ACS1 — B206mini-i (UHD 4.10, Apptainer)

    apptainer exec <sandbox> python3 uhd_record2.py --output <pass>.cs16 ... &
    python3 gr_demod_tail.py --in <pass>.cs16 --out <pass>.s8 --duration ...

Metop-B/C + NOAA-20 + Suomi NPP. gr_demod_tail.py's RRC/Costas/M&M chain is
the verified offline v5/v6 SatDump mirror; it takes --iq-format/--costas-bw
per satellite (Metop AHRPT cf32 @ 2.333333 Msym/s, HRD cs16 @ 15 Msym/s).

SNPP AND NOAA-20 ARE THE SAME WAVEFORM -- QPSK, 15 Msym/s, 7812 MHz. SatDump
covers both with a single pipeline, `npp_hrd`, literally named "Suomi NPP /
JPSS-1 HRD", and NASA DRL's RT-STPS configs npp.xml / jpss1.xml differ only in
spacecraft ID (157 vs 159). So SNPP reuses the noaa20 demod profile verbatim.
(An earlier version of this comment claimed SNPP needed a different chain. It
did not; that was true only of NOAA-21.)

NOAA-21 / JPSS-2 IS genuinely different -- OQPSK, 25 Msym/s, 40 Msps,
1279-byte CADU, interleave 5 -- and is NOT wired up here. Passes for other
satellites in PASSES_FILE are skipped with a warning; use pass_scheduler.py
(raw baseband + offline demod) for those.

SNPP HAS A DEADLINE: NOAA disables its HRD Direct Broadcast after
2026-11-01 13:00 UTC, after which its entry below stops being useful.

Per-satellite RF profile: IF tuned by B206 = rf_freq - downconv_lo
Metop-B / Metop-C : 1701.3 - 1556.0 = 145.3 MHz IF  (METOP AHRPT)
NOAA-20           : 7812.0 - 7092.0 = 720.0 MHz IF  (NPP HRD, JPSS-1)
Suomi NPP         : 7812.0 - 7092.0 = 720.0 MHz IF  (NPP HRD)
"""
import subprocess, time, os, json, logging, shutil
from datetime import datetime, timezone, timedelta
# shm_ring is imported lazily, at its only call site in the "ring" branch below.
# At module level it made the RING path's dependency a hard startup requirement
# for the DIRECT path too, which never touches a ring -- so moving shm_ring.py
# out of this directory took the whole scheduler down with it.

# ── PATHS (verify all) ───────────────────────────────────────────────────────
SANDBOX     = "/home/ilham/uhd/uhd_sandbox/"        # CHECK
RECORDER    = "/home/ilham/uhd_record2.py"          # runs INSIDE apptainer (radio side)
TAIL_DEMOD  = "/home/ilham/gr_demod_tail.py"        # runs on HOST (GNU Radio side)
LIVE_UHD    = "/home/ilham/gr_live_uhd.py"          # single process, gr-uhd straight to the B206
# Absolute interpreter, NOT bare "python3": the default python3 on PATH is the
# miniforge base env, which has no gnuradio. Relying on PATH means this only
# works from an activated shell -- under cron/systemd the demod would die at
# import while the recorder ran on, and RAM-only mode keeps no baseband to
# reprocess, so the pass would be lost outright.
DEMOD_PYTHON = "/home/ilham/miniforge3/envs/gnuradio/bin/python3"
OUTPUT_DIR  = "/home/ilham/recordings"
PASSES_FILE = "/home/ilham/passes.json"             # shared with pass_scheduler.py

# ── RECORDING BEHAVIOUR ──────────────────────────────────────────────────────
PRE_AOS_BUFFER  = 30       # start this many seconds before AOS
POST_LOS_BUFFER = 30       # keep recording this many seconds after LOS
NUM_RECV_FRAMES = 1000     # host ring buffer; larger absorbs shared-server stalls
                           # RING PATH ONLY. The direct path's frame sizing now
                           # lives in gr_live_uhd.py's per-satellite profiles,
                           # because NOAA-20 needs 2000 frames x 16360 B and
                           # Metop does not -- one global constant cannot serve
                           # both. Passing --num-recv-frames here would override
                           # the profile, so the direct branch no longer does.
DISK_MARGIN     = 1.1      # require 10% headroom over the projected size (.s8 only, now)
RAM_MARGIN      = 1.5      # require this x the ring capacity free before starting
RING_BACKLOG_SECONDS = 30  # ring sized to absorb this many seconds of raw baseband
                           # backlog, NOT the whole pass -- there's no archival copy
                           # anymore, so it only needs to cover a transient demod stall
MAX_RING_BYTES  = 4 * 1024**3   # hard ceiling per pass regardless of rate/backlog math
                                # -- box has 8GB total; this plus RAM_MARGIN headroom
                                # plus OS/GNU Radio overhead is meant to never get close
GRACE_PERIOD    = 20.0     # tail demod: hard stop this long past nominal duration
STALL_TIMEOUT   = 5.0      # tail demod: finish this long after recorder goes quiet

# uhd_record2.py exit status: 1 = failed/aborted, 2 = full pass captured but
# lossy (ring drops). Kept separate so a lossy pass isn't logged as a failure.
RC_RECORDER_LOSSY = 2

# Write the raw baseband to disk and have the demod tail that file, instead of
# handing off through the shm ring. Costs disk (baseband + .s8 per pass) but
# every pass becomes reproducible offline: a bad decode can be re-demodulated
# with different settings rather than being gone forever. Set False to go back
# to the RAM-only ring.
RETAIN_BASEBAND = False

# Which live pipeline to run:
#   "direct" -> gr_live_uhd.py: ONE GNU Radio process talks to the B206 through
#               gr-uhd and writes the .s8 and the baseband archive off the same
#               stream. No shm ring, no Apptainer, no process hand-off -- and
#               the live .s8 is provably identical to re-demodulating its own
#               archive (verified byte-for-byte). Runs on the host; the conda
#               env has gr-uhd + UHD 4.10 and borrows the sandbox's FPGA images.
#   "ring"   -> uhd_record2.py (in Apptainer) -> shm ring -> gr_demod_tail.py.
PIPELINE = "direct"

# ── SUPPORTED SATELLITES ──────────────────────────────────────────────────────
# "demod_profile" names the DemodProfile subclass in gr_live_uhd.py that owns
# this satellite's transport tuning and flowgraph wiring (direct pipeline only;
# the ring pipeline's gr_demod_tail.py has no profiles). Satellites that share
# an engineering regime share a profile -- Metop-B and Metop-C are the same
# 7 Msps problem, so they both use "metop".
SAT_PROFILES = {
    "metop-b": {
        "rf_freq_hz":     1701.3e6,
        "downconv_lo_hz": 1556.0e6,   # -> 145.3 MHz IF
        "sample_rate":    7e6,
        "symbol_rate":    7e6 / 3,    # 2333333.333... -> sps = 3.0 exactly
        "gain_db":        40,
        "ant":            "RX2",
        "format":         "cf32",
        "demod_profile":  "metop",
        # costas_bw omitted -> the metop profile's own default (0.003)
    },
    "metop-c": {
        "rf_freq_hz":     1701.3e6,
        "downconv_lo_hz": 1556.0e6,   # -> 145.3 MHz IF
        "sample_rate":    7e6,
        "symbol_rate":    7e6 / 3,    # 2333333.333... -> sps = 3.0 exactly
        "gain_db":        40,
        "ant":            "RX2",
        "format":         "cf32",
        "demod_profile":  "metop",
    },
    "noaa-20": {
        "rf_freq_hz":     7812.0e6,
        "downconv_lo_hz": 7092.0e6,   # -> 720.0 MHz IF
        "sample_rate":    24e6,
        "symbol_rate":    15e6,
        "gain_db":        30,
        "ant":            "RX2",
        "format":         "cs16",     # halves pipe bandwidth vs cf32 at 24 Msps
        "demod_profile":  "noaa20",
        # costas_bw omitted -> the noaa20 profile already carries 0.002
        # (the npp_hrd value, verified offline). Setting it here too would
        # just be a second place to keep in sync.
    },
    # Suomi NPP (NORAD 37849). RF values copied from pass_scheduler.py's "snpp"
    # entry (the offline record-only path), plus symbol_rate, which that path
    # does not need. Identical to noaa-20 in every respect -- same IF, same
    # 24 Msps, same 15 Msym/s -- because it is the same waveform; see the
    # module docstring. demod_profile "snpp" only changes the log label.
    #
    # 24 Msps, NOT SatDump's 25e6 pipeline default: the July 2026 SNPP capture
    # decoded at 24 Msps with sps 1.600000 and "Resample : 0"
    # (recordings/past_recordings/snpp_20260709_1814.log), 24 MHz is an exact
    # master clock rate on this B206mini, and 25 would cost ~4% more CPU on a
    # chain with only ~15% margin.
    #
    # DEADLINE: SNPP HRD Direct Broadcast is disabled after 2026-11-01 13:00Z.
    "snpp": {
        "rf_freq_hz":     7812.0e6,
        "downconv_lo_hz": 7092.0e6,   # -> 720.0 MHz IF
        "sample_rate":    24e6,
        "symbol_rate":    15e6,
        "gain_db":        30,
        "ant":            "RX2",
        "format":         "cs16",
        "demod_profile":  "snpp",
    },
}

# ── LOGGING ──────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s UTC] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(OUTPUT_DIR, "live_scheduler.log"))],
)
log = logging.getLogger(__name__)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def parse_utc(s):      return datetime.fromisoformat(s.replace("Z", "+00:00"))
def now_utc():         return datetime.now(timezone.utc)
def seconds_until(dt): return (dt - now_utc()).total_seconds()

def available_ram_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024   # kB -> bytes
    raise RuntimeError("MemAvailable not found in /proc/meminfo")

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
        if p["sat"] not in SAT_PROFILES:
            log.warning(f"Skipping {p['sat']} AOS {p['aos']} — "
                        f"no live demod chain wired up for this satellite.")
            continue
        passes.append(p)
    return passes

def record_pass_live(p):
    prof      = SAT_PROFILES[p["sat"]]
    if_hz     = prof["rf_freq_hz"] - prof["downconv_lo_hz"]
    rate      = prof["sample_rate"]
    symrate   = prof["symbol_rate"]
    gain      = prof["gain_db"]
    ant       = prof.get("ant")
    fmt       = prof.get("format", "cf32")
    costas_bw = prof.get("costas_bw")   # None -> let the demod use its own default
    dprofile  = prof.get("demod_profile")  # None -> gr_live_uhd.py's --profile auto

    aos = parse_utc(p["aos"]); los = parse_utc(p["los"])
    duration = int((los - aos).total_seconds()) + PRE_AOS_BUFFER + POST_LOS_BUFFER
    tag = f"{p['sat'].replace('-','_')}_{aos.strftime('%Y%m%d_%H%M%S')}_live"
    shm_name = tag
    bb_fname = os.path.join(OUTPUT_DIR, tag + ("." + fmt))
    s8_fname = os.path.join(OUTPUT_DIR, tag + ".s8")
    recorder_log_fname = os.path.join(OUTPUT_DIR, tag + "_recorder.log")
    demod_log_fname = os.path.join(OUTPUT_DIR, tag + "_demod.log")

    bytes_per_samp = 8 if fmt == "cf32" else 4
    bytes_per_sec = rate * bytes_per_samp
    need_s8 = duration * symrate * 2
    need_bb = duration * bytes_per_sec

    # ── DIRECT pipeline: one process, gr-uhd straight to the radio ──────────
    if PIPELINE == "direct":
        need_total = need_s8 + (need_bb if RETAIN_BASEBAND else 0)
        free = shutil.disk_usage(OUTPUT_DIR).free
        if free < need_total * DISK_MARGIN:
            log.error(f"Insufficient disk: need {need_total/1e9:.2f} GB "
                      + (f"(baseband {need_bb/1e9:.2f} + .s8 {need_s8/1e9:.2f}) "
                         if RETAIN_BASEBAND else f"(.s8 only) ")
                      + f"+{int((DISK_MARGIN-1)*100)}% margin, have {free/1e9:.2f} GB "
                        f"— skipping {tag}")
            return None

        cmd = [
            DEMOD_PYTHON, LIVE_UHD,
            "--freq", str(if_hz), "--rate", str(rate), "--gain", str(gain),
            "--duration", str(duration), "--symbol-rate", str(symrate),
            "--out", s8_fname,
            "--grace-period", str(GRACE_PERIOD),
        ]
        # Transport tuning (recv frames/size, buffers, work-size caps) comes
        # from the named profile -- deliberately NOT passed here, since an
        # explicit flag overrides the profile inside gr_live_uhd.py.
        if dprofile:
            cmd += ["--profile", dprofile]
        if RETAIN_BASEBAND:
            cmd += ["--archive", bb_fname, "--archive-format", fmt]
        if ant:
            cmd += ["--ant", ant]
        if costas_bw is not None:
            cmd += ["--costas-bw", str(costas_bw)]

        log.info("─────────────────────────────────────────")
        log.info(f"SAT {p['sat'].upper()} | AOS {aos:%H:%M:%S} LOS {los:%H:%M:%S} UTC | MaxEl {p['max_el']:.1f}°")
        log.info(f"RF {prof['rf_freq_hz']/1e6:.1f} MHz - LO {prof['downconv_lo_hz']/1e6:.1f} MHz "
                 f"= IF {if_hz/1e6:.1f} MHz | {rate/1e6:.0f} Msps | symrate {symrate/1e6:.4f} Msym/s "
                 f"| sps {rate/symrate:.6f} | gain {gain} dB | {duration}s")
        log.info(f"PIPELINE direct (gr-uhd, single process — no ring, no Apptainer) "
                 f"| demod profile '{dprofile or 'auto'}'")
        if RETAIN_BASEBAND:
            log.info(f"BASEBAND {bb_fname} ({need_bb/1e9:.2f} GB archive, RETAINED)")
        else:
            log.info(f"BASEBAND not written (.s8 only) — a bad pass CANNOT be "
                     f"re-demodulated offline")
        log.info(f"S8 {s8_fname} ({need_s8/1e9:.2f} GB) | disk free {free/1e9:.2f} GB")
        log.info(f"CMD {' '.join(cmd)}")
        log.info(f"LOG {demod_log_fname}")
        log.info("─────────────────────────────────────────")

        proc = None
        try:
            with open(demod_log_fname, "w", buffering=1) as lg:
                # start_new_session so a Ctrl-C in the scheduler doesn't orphan
                # a process still holding the B206.
                proc = subprocess.Popen(cmd, start_new_session=True,
                                        stdout=lg, stderr=subprocess.STDOUT)
                rc = proc.wait()
            if rc != 0:
                log.error(f"Direct live demod FAILED rc={rc} for {tag} — see "
                          f"{demod_log_fname}. Baseband archive (if written) at "
                          f"{bb_fname} can be re-demodulated offline.")
            else:
                # gr-uhd has no overflow counter; it prints a bare 'O' to stdout
                # per overflow, with no newline. Count ONLY those: every other
                # line in this log starts with '[' (UHD's [INFO]/[WARNING] and
                # our own [*]/[!]/[telemetry]), and naively counting "O" would
                # match INFO, CODEC, Operating... which reported 16 phantom
                # overflows on a pass that had none.
                try:
                    with open(demod_log_fname) as lg:
                        stray = "".join(l for l in lg if not l.lstrip().startswith("["))
                    overflows = stray.count("O")
                except OSError:
                    overflows = -1
                log.info(f"Live demod complete: {s8_fname}"
                         + (f" | archive {bb_fname}" if RETAIN_BASEBAND else "")
                         + (f" | WARNING {overflows} overflow marker(s) in the log"
                            if overflows > 0 else " | no overflow markers"))
        except Exception as e:
            log.exception("Unexpected error")
            if proc and proc.poll() is None:
                proc.terminate()
        return s8_fname

    if RETAIN_BASEBAND:
        # ── ARCHIVE + RAM mode: the recorder writes the baseband to disk AND
        # feeds the shm ring; the demod reads the RING, so the live chain never
        # competes with the archive writer for disk reads. Disk is the
        # authoritative capture -- if the ring hiccups, the pass is still fully
        # recorded and can be re-demodulated offline. Costs both disk and RAM.
        need_total = need_bb + need_s8
        free = shutil.disk_usage(OUTPUT_DIR).free
        if free < need_total * DISK_MARGIN:
            log.error(f"Insufficient disk: need {need_total/1e9:.2f} GB "
                      f"(baseband {need_bb/1e9:.2f} + .s8 {need_s8/1e9:.2f}) "
                      f"+{int((DISK_MARGIN-1)*100)}% margin, have {free/1e9:.2f} GB "
                      f"— skipping {tag}")
            return None

        ring_capacity = min(int(RING_BACKLOG_SECONDS * bytes_per_sec), MAX_RING_BYTES)
        ram_free = available_ram_bytes()
        if ram_free < ring_capacity * RAM_MARGIN:
            log.error(f"Insufficient RAM for the live ring: need "
                      f"{ring_capacity * RAM_MARGIN / 1e9:.2f} GB, have "
                      f"{ram_free/1e9:.2f} GB — skipping {tag}")
            return None

        log.info(f"Pre-flight OK (archive+RAM): baseband {need_bb/1e9:.2f} GB + "
                 f".s8 {need_s8/1e9:.2f} GB = {need_total/1e9:.2f} GB disk "
                 f"({free/1e9:.2f} GB free) | ring {ring_capacity/1e9:.2f} GB RAM "
                 f"({ram_free/1e9:.2f} GB available)")
        sink_args   = ["--output", bb_fname,
                       "--shm-name", shm_name,
                       "--shm-capacity-bytes", str(ring_capacity)]
        source_args = ["--shm-name", shm_name]
    else:
        # ── RAM mode: ring sized to a bounded backlog window, NOT the whole
        # pass -- capacity stays reserved for the pass duration regardless of
        # how much has been consumed, so tying it to total pass bytes would
        # demand tens of GB. No baseband artifact survives.
        ring_capacity = min(int(RING_BACKLOG_SECONDS * bytes_per_sec), MAX_RING_BYTES)
        if ring_capacity < MAX_RING_BYTES:
            backlog_secs_actual = RING_BACKLOG_SECONDS
        else:
            backlog_secs_actual = ring_capacity / bytes_per_sec
            log.warning(f"{tag}: {RING_BACKLOG_SECONDS}s backlog at {bytes_per_sec/1e6:.0f} MB/s "
                         f"exceeds MAX_RING_BYTES ({MAX_RING_BYTES/1e9:.1f} GB) -- "
                         f"clipped to {backlog_secs_actual:.1f}s of backlog margin")

        ram_free = available_ram_bytes()
        if ram_free < ring_capacity * RAM_MARGIN:
            log.error(f"Insufficient RAM for baseband ring: need {ring_capacity * RAM_MARGIN / 1e9:.2f} GB "
                      f"({RAM_MARGIN}x {ring_capacity/1e9:.2f} GB ring, {backlog_secs_actual:.0f}s backlog), "
                      f"have {ram_free/1e9:.2f} GB available — skipping {tag}")
            return None

        free = shutil.disk_usage(OUTPUT_DIR).free
        if free < need_s8 * DISK_MARGIN:
            log.error(f"Insufficient disk for .s8: need {need_s8/1e9:.2f} GB "
                      f"+{int((DISK_MARGIN-1)*100)}% margin, have {free/1e9:.2f} GB "
                      f"— skipping {tag}")
            return None
        log.info(f"Pre-flight OK: ring {ring_capacity/1e9:.2f} GB RAM "
                 f"({ram_free/1e9:.2f} GB available), .s8 {need_s8/1e9:.2f} GB disk "
                 f"({free/1e9:.2f} GB free)")
        sink_args   = ["--shm-name", shm_name,
                       "--shm-capacity-bytes", str(ring_capacity)]
        source_args = ["--shm-name", shm_name]

    # ── recorder (in sandbox) + tail demod (on host), run concurrently ──────
    recorder_cmd = [
        "apptainer", "exec", SANDBOX, "python3", RECORDER,
        "--freq", str(if_hz), "--rate", str(rate), "--gain", str(gain),
        "--duration", str(duration),
        *sink_args,
        "--format", fmt, "--num-recv-frames", str(NUM_RECV_FRAMES),
    ]
    if ant:
        recorder_cmd += ["--ant", ant]

    demod_cmd = [
        DEMOD_PYTHON, TAIL_DEMOD, *source_args, "--out", s8_fname,
        "--duration", str(duration), "--rate", str(rate),
        "--symbol-rate", str(symrate), "--iq-format", fmt,
        "--grace-period", str(GRACE_PERIOD), "--stall-timeout", str(STALL_TIMEOUT),
    ]
    if costas_bw is not None:
        demod_cmd += ["--costas-bw", str(costas_bw)]

    log.info("─────────────────────────────────────────")
    log.info(f"SAT {p['sat'].upper()} | AOS {aos:%H:%M:%S} LOS {los:%H:%M:%S} UTC | MaxEl {p['max_el']:.1f}°")
    log.info(f"RF {prof['rf_freq_hz']/1e6:.1f} MHz - LO {prof['downconv_lo_hz']/1e6:.1f} MHz "
             f"= IF {if_hz/1e6:.1f} MHz | {rate/1e6:.0f} Msps | symrate {symrate/1e6:.4f} Msym/s "
             f"| gain {gain} dB | {duration}s")
    if RETAIN_BASEBAND:
        log.info(f"BASEBAND {bb_fname} ({need_bb/1e9:.2f} GB archive, RETAINED) "
                 f"+ live ring shm:{shm_name} ({ring_capacity/1e9:.2f} GB)")
    else:
        log.info(f"BASEBAND RING shm:{shm_name} ({ring_capacity/1e9:.2f} GB, no disk artifact)")
    log.info(f"S8 {s8_fname}")
    log.info(f"RECORDER CMD {' '.join(recorder_cmd)}")
    log.info(f"RECORDER LOG {recorder_log_fname}")
    log.info(f"DEMOD CMD {' '.join(demod_cmd)}")
    log.info(f"DEMOD LOG {demod_log_fname}")
    log.info("─────────────────────────────────────────")

    recorder_proc = None
    demod_proc = None
    try:
        # both children's stdout/stderr go to their own log file -- this is
        # the actual overflow/backlog telemetry (uhd_record2.py's throttled
        # progress prints, gr_demod_tail.py's eprint()s), which otherwise
        # only ever went to the scheduler's own inherited terminal and was
        # lost. line-buffered so `tail -f` works during a live pass too.
        with open(recorder_log_fname, "w", buffering=1) as rec_log, \
             open(demod_log_fname, "w", buffering=1) as dem_log:
            # start_new_session on the recorder so a Ctrl-C / stop doesn't
            # orphan it holding the B206; the tail demod has no hardware to
            # release.
            recorder_proc = subprocess.Popen(recorder_cmd, start_new_session=True,
                                              stdout=rec_log, stderr=subprocess.STDOUT)
            demod_proc = subprocess.Popen(demod_cmd,   # ring source waits for the segment
                                           stdout=dem_log, stderr=subprocess.STDOUT)

            rc_rec = recorder_proc.wait()
            rc_demod = demod_proc.wait()

        fallback = (f"baseband retained at {bb_fname} — re-demod offline with "
                    f"{TAIL_DEMOD} --in {bb_fname}") if RETAIN_BASEBAND else \
                   "no baseband retained, RAM-only pipeline — cannot re-demod offline"

        if rc_rec == RC_RECORDER_LOSSY:
            # The live ring feed had gaps. With an archive that is NOT data loss
            # -- the disk capture is complete and the .s8 can be regenerated.
            if RETAIN_BASEBAND:
                log.warning(f"Live ring feed had gaps (rc={rc_rec}) for {tag} — "
                            f"{s8_fname} has gaps, but NO DATA WAS LOST: the archive "
                            f"{bb_fname} is complete. See {recorder_log_fname}. "
                            f"Regenerate a gap-free .s8 with: {TAIL_DEMOD} --in {bb_fname}")
            else:
                log.warning(f"Recorder completed the full pass but DROPPED samples "
                            f"(rc={rc_rec}) for {tag} — reader could not keep up; see "
                            f"{recorder_log_fname} for the count and timing. "
                            f"{s8_fname} is usable but has gaps. {fallback}")
        elif rc_rec != 0:
            log.error(f"Recorder FAILED/LOSSY rc={rc_rec} for {tag} "
                      f"(demod rc={rc_demod}) — {fallback}")
        elif rc_demod != 0:
            log.error(f"Tail demod FAILED rc={rc_demod} for {s8_fname} — {fallback}")
        else:
            log.info(f"Live demod complete: {s8_fname}"
                     + (f" | baseband kept: {bb_fname}" if RETAIN_BASEBAND else ""))
    except Exception as e:
        log.exception("Unexpected error")
        for proc in (recorder_proc, demod_proc):
            if proc and proc.poll() is None:
                proc.terminate()
    finally:
        # The ring exists in BOTH modes now, so it always needs unlinking.
        # Scheduler owns it: the only party guaranteed to still be alive after
        # BOTH recorder and reader exit, so it can't unlink out from under a
        # reader still draining backlog.
        try:
            from shm_ring import ShmRing   # ring path only -- see note at the imports
            ring = ShmRing(shm_name, create=False)
            ring.close()
            ring.unlink()
        except FileNotFoundError:
            pass
        # the disk archive is deliberately left in place -- that's the whole
        # point; deleting it is a manual decision once the pass checks out
    return s8_fname

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def preflight_demod_python():
    """Fail at startup, not 30 s before AOS. A missing gnuradio only shows up
    when the demod actually imports it, by which point the recorder is already
    streaming into a ring nobody will drain."""
    if not os.path.isfile(DEMOD_PYTHON):
        raise SystemExit(f"Demod interpreter not found: {DEMOD_PYTHON}")
    r = subprocess.run([DEMOD_PYTHON, "-c", "import gnuradio, gnuradio.digital"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{DEMOD_PYTHON} cannot import gnuradio:\n{r.stderr.strip()}")
    log.info(f"Pre-flight OK: demod interpreter {DEMOD_PYTHON} has gnuradio")


def main():
    passes = load_passes(PASSES_FILE)
    preflight_demod_python()
    log.info("═══ LIVE Pass Scheduler — ACS1 B206mini-i ═══")
    if PIPELINE == "direct":
        log.info(f"Live demod {LIVE_UHD} (host, gr-uhd single process) "
                 f"| {len(passes)} supported passes")
    else:
        log.info(f"Recorder {RECORDER} (in sandbox) | Tail demod {TAIL_DEMOD} (host) "
                 f"| {len(passes)} supported passes")

    for p in sorted(passes, key=lambda x: parse_utc(x["aos"])):
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

        log.info("AOS imminent — " + ("live demod now." if PIPELINE == "direct"
                                        else "recording + tail demod now."))
        record_pass_live(p)

    log.info("All scheduled passes complete.")

if __name__ == "__main__":
    main()

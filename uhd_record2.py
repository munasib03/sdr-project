#!/usr/bin/env python3
"""
IQ recorder for USRP B206mini-i (UHD 4.10 Python API).
Records raw interleaved baseband to disk for SatDump offline processing,
with zero per-buffer format conversion in the hot path.

  cf32 (default):  interleaved float32 I,Q  -> SatDump baseband_format cf_32
  cs16:            interleaved int16   I,Q  -> SatDump baseband_format cs_16

Changes vs v1:
  * recv() into a large (~1M sample) buffer instead of one packet
    (~24 loop iterations/s at 24 Msps instead of ~12,000)
  * 16 MB buffered file writes
  * wall-clock deadline: recording always ends on time even if samples
    are dropped (no noise tail past LOS, no infinite loop on a stalled device)
  * bail out after 5 consecutive recv timeouts
  * exit code 1 if any overflows / errors occurred, so the scheduler's
    check=True flags a lossy recording as FAILED

Example:
    python3 uhd_record.py --freq 1701.3e6 --rate 6e6 --gain 35 \
        --duration 60 --output metop_pass.cf32
"""
import argparse
import sys
import time
import numpy as np
import uhd
from shm_ring import ShmRing


def main():
    p = argparse.ArgumentParser(description="Record IQ from USRP B206mini-i")
    p.add_argument("--freq", type=float, required=True, help="Center frequency (Hz)")
    p.add_argument("--rate", type=float, default=6e6, help="Sample rate (Hz)")
    p.add_argument("--gain", type=float, default=35, help="RX gain (dB)")
    p.add_argument("--duration", type=float, required=True, help="Duration (s)")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path (disk mode)")
    p.add_argument("--shm-name", type=str, default=None,
                   help="Shared-memory ring name (RAM mode) -- mutually exclusive with --output")
    p.add_argument("--shm-capacity-bytes", type=int, default=None,
                   help="Ring capacity in bytes; required with --shm-name")
    p.add_argument("--format", choices=["cf32", "cs16"], default="cf32",
                   help="cf32 = interleaved float32, cs16 = interleaved int16")
    p.add_argument("--args", type=str, default="",
                   help="Extra UHD device args, e.g. 'serial=xxxx'")
    p.add_argument("--ant", type=str, default=None,
                   help="RX antenna port, e.g. RX2 or TX/RX")
    p.add_argument("--num-recv-frames", type=int, default=1000,
                   help="Host RX ring-buffer frames (bigger absorbs stalls)")
    p.add_argument("--recv-buf-samps", type=int, default=1 << 20,
                   help="Samples per recv() call (bigger = fewer Python iterations)")
    p.add_argument("--max-timeouts", type=int, default=5,
                   help="Consecutive recv timeouts before aborting")
    args = p.parse_args()

    if not args.output and not args.shm_name:
        sys.exit("Specify at least one of --output (disk) or --shm-name (RAM ring); "
                 "giving both archives to disk AND feeds a live reader from RAM")
    if args.shm_name and not args.shm_capacity_bytes:
        sys.exit("--shm-name requires --shm-capacity-bytes")

    # ---- build device args with a large host-side ring buffer ----
    dev_args = f"num_recv_frames={args.num_recv_frames}"
    if args.args:
        dev_args += "," + args.args

    print(f"[*] Connecting to USRP ({dev_args})...")
    usrp = uhd.usrp.MultiUSRP(dev_args)

    usrp.set_rx_rate(args.rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args.freq))
    usrp.set_rx_gain(args.gain)
    if args.ant:
        usrp.set_rx_antenna(args.ant)
        print(f"[*] RX antenna: {usrp.get_rx_antenna()}")

    actual_rate = usrp.get_rx_rate()
    actual_freq = usrp.get_rx_freq()
    actual_gain = usrp.get_rx_gain()
    print(f"[*] rate {actual_rate/1e6:.3f} Msps | "
          f"freq {actual_freq/1e6:.4f} MHz | gain {actual_gain:.1f} dB")

    # ---- CPU format matches output: no round-trip conversion ----
    if args.format == "cf32":
        cpu_fmt = "fc32"
        buf_dtype = np.complex64                 # memory = [re,im,re,im] float32
        bytes_per_samp = 8
    else:  # cs16
        cpu_fmt = "sc16"
        buf_dtype = np.dtype([("re", np.int16), ("im", np.int16)])  # 4 bytes/item
        bytes_per_samp = 4

    st_args = uhd.usrp.StreamArgs(cpu_fmt, "sc16")
    st_args.channels = [0]
    rx = usrp.get_rx_stream(st_args)

    # ---- large recv buffer: recv() aggregates many packets per call ----
    spb = rx.get_max_num_samps()
    buf_samps = max(spb, args.recv_buf_samps)
    buffer = np.zeros((1, buf_samps), dtype=buf_dtype)
    md = uhd.types.RXMetadata()
    print(f"[*] recv buffer: {buf_samps} samples "
          f"({buf_samps * bytes_per_samp / 1e6:.1f} MB, device max {spb}/pkt)")

    num_total = int(args.duration * actual_rate)
    expected_bytes = num_total * bytes_per_samp
    sink_label = args.output if args.output else f"shm:{args.shm_name}"
    print(f"[*] Recording {args.duration:.0f}s ({num_total} samples, "
          f"~{expected_bytes/1e9:.2f} GB) -> {sink_label}")

    # ---- optional: wait for LO lock, then let the AD9361 settle ----
    try:
        for _ in range(50):
            if usrp.get_rx_sensor("lo_locked").to_bool():
                break
            time.sleep(0.01)
    except Exception:
        pass
    time.sleep(0.2)

    # ---- stream ----
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    rx.issue_stream_cmd(cmd)

    OVERFLOW = uhd.types.RXMetadataErrorCode.overflow
    TIMEOUT = uhd.types.RXMetadataErrorCode.timeout
    NONE = uhd.types.RXMetadataErrorCode.none

    received = 0
    overflows = 0
    timeouts = 0          # consecutive
    total_timeouts = 0
    other_errs = 0
    ring_drops = 0        # chunks refused because the ring was full
    ring_drop_samps = 0
    ring_recovered = False   # logged once, when writes resume after a drop
    peak_backlog = 0         # so a drop can be told apart from a real overrun
    aborted = False
    last_print = 0.0
    rssi_supported = None   # None = untested, True/False once known

    t0 = time.time()
    deadline = t0 + args.duration   # wall clock, not sample count

    # Both sinks may be active at once. The disk file is the ARCHIVE -- it is
    # written unconditionally and is the authoritative capture. The ring is only
    # a live feed for a concurrent demod; if it fills, its newest chunk is
    # dropped rather than blocking recv() (blocking would stall the device and
    # turn a live-reader hiccup into a real, unbounded overflow). With an
    # archive present a ring drop costs nothing permanent: the samples are on
    # disk and the pass can simply be re-demodulated offline.
    f = ring = None
    if args.output:
        f = open(args.output, "wb", buffering=16 * 1024 * 1024)
    if args.shm_name:
        ring = ShmRing(args.shm_name, args.shm_capacity_bytes, create=True)
    print(f"[*] sinks: disk={'yes' if f else 'no'}  ring={'yes' if ring else 'no'}"
          + ("  (ring drops are recoverable from the archive)" if f and ring else ""))

    try:
        first = True
        while time.time() < deadline:
            # generous timeout on the very first packet, tight after
            num_rx = rx.recv(buffer, md, 3.0 if first else 1.0)
            first = False
            ec = md.error_code

            if ec == TIMEOUT:
                timeouts += 1
                total_timeouts += 1
                if timeouts >= args.max_timeouts:
                    print(f"\n[!] {timeouts} consecutive recv timeouts — "
                          f"device stalled, aborting.")
                    aborted = True
                    break
                continue
            timeouts = 0

            if ec == OVERFLOW:
                overflows += 1
                continue
            if ec != NONE:
                other_errs += 1
                continue
            if num_rx == 0:
                continue

            # raw bytes, no conversion -- built once, handed to both sinks
            chunk = np.ascontiguousarray(buffer[0, :num_rx]).tobytes()

            if f is not None:
                f.write(chunk)          # archive: unconditional, never refuses

            if ring is not None and not ring.write_nonblocking(chunk):
                # A refusal is NEVER fatal: keep looping to the wall-clock
                # deadline so the rest of the pass is still captured.
                if ring_drops == 0:
                    print(f"\n[!] t={time.time()-t0:.1f}s ring full — live demod is "
                          f"behind; dropping from the RING only. Recording CONTINUES"
                          + (" and the disk archive is unaffected." if f else "."),
                          flush=True)
                ring_drops += 1
                ring_drop_samps += num_rx
            elif ring is not None and ring_drops and not ring_recovered:
                ring_recovered = True
                print(f"\n[*] t={time.time()-t0:.1f}s ring drained — feeding the live "
                      f"demod again after {ring_drops} dropped chunk(s).", flush=True)

            # counts samples CAPTURED. With an archive that is what landed on
            # disk; ring drops are tracked separately and cost only the live .s8.
            received += num_rx

            now = time.time() - t0
            if now - last_print >= 1.0:          # throttle prints to 1/s
                last_print = now
                pct = 100 * now / args.duration
                if ring:
                    bl = ring.backlog_bytes()
                    peak_backlog = max(peak_backlog, bl)
                    backlog = f"  backlog={bl/1e6:.1f}MB"
                else:
                    backlog = ""
                if ring_drops:
                    backlog += f"  RINGDROPS={ring_drops}"
                rssi = ""
                if rssi_supported is not False:
                    try:
                        rssi_val = usrp.get_rx_sensor("rssi", 0).to_real()
                        rssi = f"  rssi={rssi_val:.1f}dB"
                        rssi_supported = True
                    except Exception:
                        rssi_supported = False   # stop trying every iteration
                # \r overwrite for an interactive terminal; \n when redirected
                # to a log file so each sample is its own greppable line
                end = "" if sys.stdout.isatty() else "\n"
                print(f"\r[*] {pct:5.1f}%  {now:6.1f}s  "
                      f"overflows={overflows}{backlog}{rssi}", end=end, flush=True)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
        aborted = True
    finally:
        rx.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        if f:
            f.close()
        if ring:
            ring.close()   # unlink is the scheduler's job -- reader may still be draining

    wrote = received * bytes_per_samp
    deficit = num_total - received
    print(f"\n[+] Done. {received} samples, {wrote} bytes "
          f"({wrote/1e9:.2f} GB) -> {sink_label}")
    print(f"[+] overflows={overflows}  timeouts={total_timeouts}  "
          f"other_errors={other_errs}  ring_drops={ring_drops}")
    if ring_drops:
        pct = 100.0 * peak_backlog / args.shm_capacity_bytes
        print(f"[!] Ring refused {ring_drops} chunk(s): {ring_drop_samps} samples "
              f"({ring_drop_samps/actual_rate:.2f} s of {args.duration:.0f} s) never "
              f"reached the live demod. Peak backlog was {peak_backlog/1e6:.1f} MB of "
              f"the {args.shm_capacity_bytes/1e9:.2f} GB ring ({pct:.1f}%).")
        if pct < 50.0:
            # Not an overrun: the ring had ample room, so the reader kept up.
            print(f"[!] Peak backlog stayed well under capacity, so this was NOT the "
                  f"demod falling behind. Suspect a spurious refusal -- check that "
                  f"shm_ring.py is the ctypes/counter-guard version.")
        else:
            print(f"[!] Backlog approached capacity: the demod genuinely could not "
                  f"keep up. Reduce competing CPU load during the pass, or raise "
                  f"--shm-capacity-bytes.")
        if f:
            print(f"[+] NO DATA LOST: the disk archive {args.output} is complete "
                  f"({wrote} bytes). Only the live .s8 has gaps -- re-demodulate "
                  f"the archive offline for a gap-free result.")
        else:
            print(f"[!] No archive was written, so those samples are gone; the .s8 "
                  f"covers the full window with {ring_drop_samps/actual_rate:.2f} s of gaps.")
    if deficit > 0:
        print(f"[!] Sample deficit: {deficit} samples "
              f"({deficit/actual_rate:.2f} s of data) missing vs. expected.")
    if overflows:
        print("[!] Nonzero overflows: samples were DROPPED. The bitstream has "
              "discontinuities — expect deframer sync losses in SatDump. "
              "Try local/NVMe output and/or --num-recv-frames 2000.")

    # Exit status, in increasing severity:
    #   0 = pristine
    #   2 = ran to completion, but the live ring feed had gaps. A notice, NOT a
    #       failure -- and when an archive was written it is not even data loss,
    #       only a degraded live .s8 that can be regenerated from the archive.
    #   1 = failed/aborted or device-side loss (overflow, error, abort)
    # Kept distinct so the scheduler can warn about 2 without calling the pass
    # a failure and without conflating it with a stalled or aborted device.
    if aborted or overflows or other_errs:
        sys.exit(1)
    if ring_drops:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()

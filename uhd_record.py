#!/usr/bin/env python3
"""
IQ recorder for USRP B206mini-i (UHD 4.10 Python API).
Records raw interleaved baseband to disk for SatDump offline processing,
with zero per-buffer format conversion in the hot path.

  cf32 (default):  interleaved float32 I,Q  -> baseband_format cf32
  cs16:            interleaved int16   I,Q  -> baseband_format cs16
  s8:              interleaved int8    I,Q  -> baseband_format s8

The --format flag sets the HOST/file format. USB bus load is set separately by
the over-the-wire format (--otw), which defaults to sc8 for s8 and sc16
otherwise. At the highest rates otw is the binding constraint, not the disk:
61.44 Msps costs 245.8 MB/s on the bus as sc16 but only 122.9 MB/s as sc8.

Example:
    python3 uhd_record.py --freq 1701.3e6 --rate 6e6 --gain 35 \
        --duration 60 --output metop_pass.cf32
"""
import argparse
import sys
import time
import numpy as np
import uhd


def main():
    p = argparse.ArgumentParser(description="Record IQ from USRP B206mini-i")
    p.add_argument("--freq", type=float, required=True, help="Center frequency (Hz)")
    p.add_argument("--rate", type=float, default=6e6, help="Sample rate (Hz)")
    p.add_argument("--gain", type=float, default=35, help="RX gain (dB)")
    p.add_argument("--duration", type=float, required=True, help="Duration (s)")
    p.add_argument("--output", type=str, required=True, help="Output file path")
    p.add_argument("--format", choices=["cf32", "cs16", "s8"], default="cf32",
                   help="cf32 = interleaved float32, cs16 = interleaved int16, "
                        "s8 = interleaved int8")
    p.add_argument("--otw", choices=["sc16", "sc12", "sc8"], default=None,
                   help="Over-the-wire sample format; sets USB bus load, NOT "
                        "file size. Default: sc8 for --format s8, else sc16.")
    p.add_argument("--args", type=str, default="",
                   help="Extra UHD device args, e.g. 'serial=xxxx'")
    p.add_argument("--ant", type=str, default=None,
                   help="RX antenna port, e.g. RX2 or TX/RX")
    p.add_argument("--num-recv-frames", type=int, default=1000,
                   help="Host RX ring-buffer frames (bigger absorbs stalls)")
    args = p.parse_args()

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
    elif args.format == "cs16":
        cpu_fmt = "sc16"
        buf_dtype = np.dtype([("re", np.int16), ("im", np.int16)])  # 4 bytes/item
        bytes_per_samp = 4
    else:  # s8
        cpu_fmt = "sc8"
        buf_dtype = np.dtype([("re", np.int8), ("im", np.int8)])    # 2 bytes/item
        bytes_per_samp = 2

    # Over-the-wire format is what the USB bus actually carries, independent of
    # the host/file format above. This is the binding constraint at high rates.
    otw_fmt = args.otw or ("sc8" if args.format == "s8" else "sc16")
    OTW_BYTES = {"sc16": 4, "sc12": 3, "sc8": 2}
    bus_mbs = actual_rate * OTW_BYTES[otw_fmt] / 1e6
    print(f"[*] host {args.format} ({bytes_per_samp} B/samp) | "
          f"otw {otw_fmt} ({OTW_BYTES[otw_fmt]} B/samp) | "
          f"bus ~{bus_mbs:.1f} MB/s | disk ~{actual_rate*bytes_per_samp/1e6:.1f} MB/s")

    st_args = uhd.usrp.StreamArgs(cpu_fmt, otw_fmt)
    st_args.channels = [0]
    rx = usrp.get_rx_stream(st_args)

    spb = rx.get_max_num_samps()
    buffer = np.zeros((1, spb), dtype=buf_dtype)
    md = uhd.types.RXMetadata()

    num_total = int(args.duration * actual_rate)
    expected_bytes = num_total * bytes_per_samp
    print(f"[*] Recording {args.duration:.0f}s ({num_total} samples, "
          f"~{expected_bytes/1e9:.2f} GB) -> {args.output}")

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
    NONE = uhd.types.RXMetadataErrorCode.none

    received = 0
    overflows = 0
    other_errs = 0
    last_print = 0.0
    t0 = time.time()

    try:
        with open(args.output, "wb") as f:
            first = True
            while received < num_total:
                # generous timeout on the very first packet, tight after
                num_rx = rx.recv(buffer, md, 3.0 if first else 1.0)
                first = False
                ec = md.error_code
                if ec == OVERFLOW:
                    overflows += 1
                    continue
                if ec != NONE:
                    other_errs += 1
                    continue
                if num_rx == 0:
                    continue
                # raw write, no conversion
                f.write(np.ascontiguousarray(buffer[0, :num_rx]).tobytes())
                received += num_rx

                now = time.time() - t0
                if now - last_print >= 1.0:          # throttle prints to 1/s
                    last_print = now
                    pct = 100 * received / num_total
                    print(f"\r[*] {pct:5.1f}%  {now:5.1f}s  "
                          f"overflows={overflows}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[!] Interrupted.")
    finally:
        rx.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    wrote = received * bytes_per_samp
    print(f"\n[+] Done. {received} samples, {wrote} bytes "
          f"({wrote/1e9:.2f} GB) -> {args.output}")
    print(f"[+] overflows={overflows}  other_errors={other_errs}")
    if overflows:
        print("[!] Nonzero overflows: samples were DROPPED. "
              "Some 'low SNR' may be dropped data, not RF. "
              "Try local-disk output and/or --num-recv-frames 2000.")


if __name__ == "__main__":
    main()

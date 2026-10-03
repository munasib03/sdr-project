#!/usr/bin/env python3
"""
Single-process live AHRPT/HRD demod: B206mini -> soft symbols (.s8), with an
optional raw-baseband archive written from the same stream.

Per-satellite profiles
----------------------
Metop AHRPT at 7 Msps and NOAA-20 HRD at 24 Msps are the same modulation
(QPSK) but NOT the same engineering problem, so they no longer share one
hard-coded chain. Each satellite gets a DemodProfile subclass that owns both
its transport tuning (UHD frame sizes, stream formats, buffer sizing) and how
its flowgraph is wired. Adding a satellite means adding a subclass and one
PROFILES entry -- no edits to the code paths the working satellites use.

    MetopProfile  -- 7 Msps, 2.3333 Msym/s, sps 3.0. The chain that has run
                     clean for a year. Deliberately left byte-for-byte as it
                     was; see the class docstring before touching it.
    Noaa20Profile -- 24 Msps, 15 Msym/s, sps 1.6. Same DSP, different plumbing:
                     the CPU margin here is ~1.17x real time instead of Metop's
                     ~4.4x, so anything that wastes a core or stalls the USB
                     drain turns into dropped samples. See its docstring.
    SnppProfile   -- Suomi NPP. SAME waveform as NOAA-20 (SatDump covers both
                     with one pipeline, "Suomi NPP / JPSS-1 HRD"), so it
                     inherits Noaa20Profile wholesale and only relabels the
                     log. NOAA turns SNPP HRD off after 2026-11-01 13:00 UTC.

Common chain (shared by both, in build_demod_tail):
    AGC(1e-2, ref 1.0) -> RRC(alpha 0.5, 31 taps) -> Costas(order 4)
      -> clock_recovery_mm_cc -> complex_to_interleaved_char -> file_sink(.s8)

Duration is bounded two ways: a blocks.head caps the exact sample/byte count
(where the profile puts it differs -- see Noaa20Profile), and a watchdog stops
the flowgraph at duration+grace so a stalled device can never hang the pass
past its window.

Examples
--------
  # Metop-B/C: 7 Msps, 2.3333 Msym/s, archive as cf32
  python3 gr_live_uhd.py --profile metop \
      --freq 145.3e6 --rate 7e6 --gain 40 --ant RX2 \
      --duration 740 --symbol-rate 2333333.3333333335 \
      --out metop.s8 --archive metop.cf32 --archive-format cf32

  # NOAA-20: 24 Msps, 15 Msym/s, archive as cs16
  python3 gr_live_uhd.py --profile noaa20 \
      --freq 720e6 --rate 24e6 --gain 30 --ant RX2 \
      --duration 798 --symbol-rate 15e6 \
      --out noaa20.s8 --archive noaa20.cs16 --archive-format cs16

  # Suomi NPP: identical parameters to NOAA-20, only the label differs
  python3 gr_live_uhd.py --profile snpp \
      --freq 720e6 --rate 24e6 --gain 30 --ant RX2 \
      --duration 798 --symbol-rate 15e6 --out snpp.s8

  # Decode either one with SatDump's shared pipeline:
  #   satdump npp_hrd soft <file>.s8 <outdir>

Every profile default below can be overridden from the command line; anything
left off the command line takes the profile's value. --profile auto (the
default) picks by symbol rate so old invocations keep working.
"""
import argparse
import os
import sys
import threading
import time

# Must be set before UHD opens the device. The host conda UHD ships no images;
# the sandbox tree has them and is a plain directory, so just point at it.
DEFAULT_IMAGES_DIR = "/home/ilham/uhd/uhd_sandbox/usr/local/share/uhd/images"
if not os.environ.get("UHD_IMAGES_DIR") and os.path.isdir(DEFAULT_IMAGES_DIR):
    os.environ["UHD_IMAGES_DIR"] = DEFAULT_IMAGES_DIR

# Kernel ceiling on pinned usbfs DMA memory. num_recv_frames x recv_frame_size
# is drawn from this pool; overshooting aborts the process at device open.
# See DemodProfile.device_args.
USBFS_MEMORY_PARAM = "/sys/module/usbcore/parameters/usbfs_memory_mb"
B200_DEFAULT_RECV_FRAME_SIZE = 8176   # UHD's B200_USB_DATA_DEFAULT_FRAME_SIZE
B200_MAX_RECV_FRAME_SIZE     = 16360  # UHD's B200_USB_DATA_MAX_RECV_FRAME_SIZE

from gnuradio import gr, blocks, digital, filter, analog, uhd
from gnuradio.filter import firdes


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)
    sys.stderr.flush()


# ============================================================================
# PROFILES
# ============================================================================

class DemodProfile:
    """
    Per-satellite transport tuning + flowgraph wiring.

    Class attributes are defaults, not policy: main() overlays anything the
    user passed explicitly on the command line before the profile is used, so
    a profile value is only what you get when you say nothing.

    To add a satellite: subclass, override what differs, register in PROFILES.
    Override build_chain only if the graph topology differs; if the satellite
    just needs different constants, override the attributes and inherit.
    """
    name = "base"

    # ---- transport (how samples get from the radio into the flowgraph) -----
    num_recv_frames = 1000   # UHD host-side ring depth, in frames
    recv_frame_size = None   # bytes/frame; None -> UHD's B200 default (8176)
    otw_format      = "sc16" # over-the-wire sample format
    cpu_format      = "fc32" # what usrp_source hands GNU Radio
    buffer_seconds  = 2.0    # elastic buffer on the source's OUTPUT, in seconds
    max_noutput_items = None # cap per work() call; None -> GNU Radio default
    recv_timeout    = 1.0
    one_packet      = False  # see set_recv_timeout note in build_source

    # ---- demod constants ---------------------------------------------------
    costas_bw             = 0.003
    rrc_alpha             = 0.5
    rrc_taps              = 31
    clock_gain_mu         = 8.7e-3
    clock_mu              = 0.5
    clock_omega_rel_limit = 0.005
    agc_rate              = 1e-2
    scale                 = 100.0

    # Fraction of the kernel's usbfs allowance this profile's RX ring may claim.
    # The rest is left for UHD's control/TX buffers, which come out of the SAME
    # budget -- the 2026-09-21 crash surfaced as "usb tx4 submit failed", i.e.
    # the RX ring had already eaten the pool by the time a TX buffer was needed.
    usbfs_budget_fraction = 0.75

    def usbfs_limit_bytes(self):
        """Kernel cap on usbfs DMA buffers, or None if it can't be read."""
        try:
            with open(USBFS_MEMORY_PARAM) as f:
                return int(f.read().strip()) * 1024 * 1024
        except (OSError, ValueError):
            return None

    def device_args(self, extra=""):
        """
        UHD device-address string, with num_recv_frames clamped to what the
        kernel will actually allow.

        WHY THE CLAMP EXISTS: num_recv_frames x recv_frame_size is pinned DMA
        memory allocated through usbfs, and the kernel caps the total at
        /sys/module/usbcore/parameters/usbfs_memory_mb (16 MB by default).
        Asking for more does not degrade gracefully -- libusb returns
        LIBUSB_ERROR_NO_MEM, UHD throws from inside b200_make(), and since that
        happens on a non-main thread gr::terminate_handler aborts the process
        (rc=-6) before the flowgraph is even built. That cost the whole
        2026-09-21 07:23 NOAA-20 pass (47.2 deg, nothing written at all), so
        this is worth a few lines to make impossible.

        Raising the ceiling is a root action and is what actually buys the full
        341 ms cushion:
            sudo sh -c 'echo 64 > /sys/module/usbcore/parameters/usbfs_memory_mb'
        (runtime-writable, resets on reboot; make it permanent with
         'options usbcore usbfs_memory_mb=64' in /etc/modprobe.d/usbcore.conf)
        """
        frames = self.num_recv_frames
        frame_size = self.recv_frame_size or B200_DEFAULT_RECV_FRAME_SIZE
        limit = self.usbfs_limit_bytes()

        if limit:
            budget = int(limit * self.usbfs_budget_fraction)
            max_frames = max(1, budget // frame_size)
            if frames > max_frames:
                eprint(f"[!] num_recv_frames {frames} x {frame_size} B = "
                       f"{frames*frame_size/2**20:.1f} MB exceeds {self.usbfs_budget_fraction:.0%} "
                       f"of the {limit/2**20:.0f} MB usbfs limit -- clamping to "
                       f"{max_frames} frames ({max_frames*frame_size/2**20:.1f} MB). "
                       f"Raise /sys/module/usbcore/parameters/usbfs_memory_mb to "
                       f"use the full ring.")
                frames = max_frames
            eprint(f"[*] usb rx ring: {frames} x {frame_size} B = "
                   f"{frames*frame_size/2**20:.2f} MB of {limit/2**20:.0f} MB usbfs limit")
        else:
            eprint(f"[!] could not read {USBFS_MEMORY_PARAM} -- not validating "
                   f"the {frames*frame_size/2**20:.1f} MB usb ring against the "
                   f"kernel limit")

        parts = [f"num_recv_frames={frames}"]
        if self.recv_frame_size:
            parts.append(f"recv_frame_size={self.recv_frame_size}")
        if extra:
            parts.append(extra)
        return ",".join(parts)

    def build_demod_tail(self, tb, head_block, symbol_rate, actual_rate, outfile):
        """
        AGC -> RRC -> Costas -> M&M -> interleaved char -> .s8.

        Identical for every satellite so far; only the constants change. The
        telemetry poller reaches into tb.costas / tb.clock, so those names are
        part of the contract with poll_telemetry().
        """
        tb.agc = analog.agc_cc(self.agc_rate, 1.0, 1.0)
        tb.agc.set_max_gain(65536)

        rrc = firdes.root_raised_cosine(1.0, actual_rate, symbol_rate,
                                        self.rrc_alpha, self.rrc_taps)
        tb.rrc = filter.fir_filter_ccf(1, rrc)
        tb.costas = digital.costas_loop_cc(self.costas_bw, 4, False)
        tb.clock = digital.clock_recovery_mm_cc(
            actual_rate / symbol_rate, (self.clock_gain_mu ** 2) / 4.0,
            self.clock_mu, self.clock_gain_mu, self.clock_omega_rel_limit)
        tb.c2if = blocks.complex_to_interleaved_char(False, self.scale)

        tb.connect(head_block, tb.agc, tb.rrc, tb.costas, tb.clock, tb.c2if)
        return tb.c2if

    def build_archive(self, tb, source_block, archive, archive_format):
        """Tee the raw baseband off `source_block` into a file."""
        if archive_format == "cs16":
            # UHD's sc16->fc32 conversion normalises by 32768; scaling back
            # by 32767 keeps every value inside int16 without saturating.
            tb.c2s = blocks.complex_to_interleaved_short(False, 32767.0)
            tb.archive_sink = blocks.file_sink(gr.sizeof_short, archive, False)
            tb.connect(source_block, tb.c2s, tb.archive_sink)
        else:
            tb.archive_sink = blocks.file_sink(gr.sizeof_gr_complex, archive, False)
            tb.connect(source_block, tb.archive_sink)
        tb.archive_sink.set_unbuffered(False)
        eprint(f"[*] archiving baseband -> {archive} ({archive_format})")

    def build_chain(self, tb, symbol_rate, actual_rate, duration, outfile,
                    archive, archive_format):
        raise NotImplementedError


class MetopProfile(DemodProfile):
    """
    Metop-B / Metop-C AHRPT: 7 Msps, 2.333333 Msym/s, sps = 3.0 exactly.

    DO NOT "OPTIMISE" THIS. Measured across every pass in recordings/: zero
    overflow events, ever. At 7 Msps the chain runs ~4.4x real time, so the
    wasted full-rate memcpy through blocks.head and the unbounded work() size
    that cost NOAA-20 real samples are, here, free. The graph below is exactly
    what has been flying; the NOAA-20 fixes are deliberately NOT applied to it
    because there is nothing to fix and a regression here is expensive.
    """
    name = "metop"
    costas_bw = 0.003        # AHRPT; matches the offline v5/v6 SatDump mirror

    def build_chain(self, tb, symbol_rate, actual_rate, duration, outfile,
                    archive, archive_format):
        # head on the full-rate complex stream: caps the exact input sample
        # count. Costs a 56 MB/s memcpy at 7 Msps, which this profile can
        # afford and NOAA-20 cannot.
        nsamps = int(duration * actual_rate)
        tb.head = blocks.head(gr.sizeof_gr_complex, nsamps)
        tb.connect(tb.src, tb.head)
        eprint(f"[*] sample cap: blocks.head on the complex stream, "
               f"{nsamps} samples ({duration:.0f}s)")

        if archive:
            self.build_archive(tb, tb.head, archive, archive_format)

        tail = self.build_demod_tail(tb, tb.head, symbol_rate, actual_rate, outfile)
        tb.sink = blocks.file_sink(gr.sizeof_char, outfile, False)
        tb.sink.set_unbuffered(False)
        tb.connect(tail, tb.sink)


class Noaa20Profile(DemodProfile):
    """
    NOAA-20 (JPSS-1) HRD: 24 Msps, 15 Msym/s, sps = 1.6.

    The problem this profile exists to solve
    ----------------------------------------
    The demod chain runs at only ~1.16-1.19x real time at 24 Msps (profiled;
    see live_pass_scheduler.py). Metop's margin is ~4.4x. That 15% headroom is
    the whole story, and the overflow logs show exactly what it does:

        08-12 pass: clean 0-320s, ~15 overflows in a 16s burst, clean for
        345s, second burst at ~680s. Same shape on 08-10, 09-17, 09-19.

    One stall fills the elastic buffer; at 1.17x it then drains at 14.5% of
    real time, so a full 2.0s buffer takes ~14s to empty and overflows the
    entire time. Predicted 14s vs observed 16-30s bursts. That is the mechanism.

    THE COUNTERINTUITIVE PART: raising buffer_seconds makes this WORSE, not
    better -- a 4s buffer gives 27s bursts. There is no buffer size that fixes
    a 15% margin. buffer_seconds stays at 2.0 on purpose.

    What is NOT the problem (checked, so nobody re-investigates):
      - USB bus. 24 Msps sc16 = 96 MB/s = 19% of UHD's own B200_MAX_RATE_USB3
        (500 MB/s), and uhd_record2.py sustained exactly this rate on this
        radio with zero loss.
      - Disk writeback. RETAIN_BASEBAND has always been False, so the only
        disk load during every overflowing pass was ~30 MB/s of .s8.

    So the fixes are all about not wasting CPU and not letting the USB drain
    thread get starved:
      - recv_frame_size at UHD's maximum (16360 B), num_recv_frames sized to
        the usbfs ceiling: the host ring goes from 7.8 MB (85 ms) to 12.0 MB
        (~131 ms) of device-side cushion, and USB completions drop from
        ~11,700/s to ~5,870/s. The ring cannot grow past ~175 ms until root
        raises usbfs_memory_mb -- see device_args().
      - max_noutput_items capped: without it the scheduler can hand
        usrp_source the whole 48M-item output buffer, so one recv() may run
        the full 1.0s timeout before anything reaches the six downstream
        threads -- they idle, then get a one-second slab. ~10 ms chunks keep
        the pipeline fed. (~100 recv/s, nowhere near the 11,700/s of
        one_packet=True that measured 8% more CPU.)
      - blocks.head moved off the full-rate path: head is a plain memcpy of
        every sample, so on the complex stream it costs 192 MB/s read +
        192 MB/s write plus a thread, purely to count samples. On the output
        byte stream it does the same job for ~30 MB/s.

    Sample rate is NOT negotiable downward: 15 Msym/s with alpha 0.5 occupies
    22.5 MHz, so 24 Msps is already near-minimal and 20 Msps would alias.
    """
    name = "noaa20"

    # ---- transport ---------------------------------------------------------
    # 768 x 16360 B = 11.98 MB = ~131 ms of cushion at 96 MB/s. Sized to fit
    # inside 75% of the kernel's default 16 MB usbfs pool, NOT to be optimal:
    # asking for 2000 frames here aborted the 2026-09-21 pass outright with
    # LIBUSB_ERROR_NO_MEM (31.2 MB requested against a 16 MB ceiling).
    # After 'echo 64 > /sys/module/usbcore/parameters/usbfs_memory_mb' as root,
    # raise this to 2000 for the full ~341 ms. device_args() clamps either way,
    # so a too-large value here degrades to a warning rather than a lost pass.
    num_recv_frames   = 768
    recv_frame_size   = B200_MAX_RECV_FRAME_SIZE  # fewer URB completions/s
    buffer_seconds    = 2.0     # do not raise -- see docstring
    max_noutput_items = 0.010   # SECONDS here; converted against the real rate

    # ---- demod -------------------------------------------------------------
    costas_bw = 0.002           # npp_hrd pipeline value (verified offline)

    def build_chain(self, tb, symbol_rate, actual_rate, duration, outfile,
                    archive, archive_format):
        # No head on the complex stream. The watchdog in main() already bounds
        # the pass on the wall clock (it has been stopping runs cleanly -- the
        # 08-12 pass finished at 830.4s for an 825s request), and the cap below
        # bounds the output exactly.
        src_out = tb.src

        if archive:
            # NOTE: with head moved downstream, the archive is no longer exactly
            # capped -- it writes until the flowgraph stops, so the watchdog's
            # grace period (+20s default) lands in the file. Measured overshoot
            # is ~2-3%, inside the scheduler's 10% disk pre-flight margin.
            # RETAIN_BASEBAND is False today, so this costs nothing in practice.
            self.build_archive(tb, src_out, archive, archive_format)

        tail = self.build_demod_tail(tb, src_out, symbol_rate, actual_rate, outfile)

        # head on the OUTPUT byte stream instead: same deterministic cap, at
        # 30 MB/s instead of 192 MB/s. 2 bytes per soft symbol (I,Q int8).
        # Note the semantics shift slightly: this caps SYMBOLS OUT rather than
        # SAMPLES IN. Clock recovery has been running ~0.006% over nominal, so
        # what gets truncated is the tail of the post-LOS buffer -- noise.
        nbytes = int(duration * symbol_rate) * 2
        tb.head = blocks.head(gr.sizeof_char, nbytes)
        tb.sink = blocks.file_sink(gr.sizeof_char, outfile, False)
        tb.sink.set_unbuffered(False)
        tb.connect(tail, tb.head, tb.sink)
        eprint(f"[*] sample cap: blocks.head on the OUTPUT byte stream, "
               f"{nbytes} bytes ({nbytes//2} symbols) -- keeps the memcpy off "
               f"the 24 Msps path")


class SnppProfile(Noaa20Profile):
    """
    Suomi NPP (SNPP / NPP, NORAD 37849) HRD: 7812 MHz -> 720 MHz IF,
    24 Msps, 15 Msym/s, sps = 1.6.

    SNPP AND NOAA-20 ARE THE SAME WAVEFORM. Not "similar" -- the same. So this
    class inherits everything and overrides only the log label, which exists so
    an operator reading a log can tell which spacecraft was up. If SNPP ever
    needs a constant of its own, this is where it goes; today there is none.

    Evidence they are identical:
      - SatDump's pipeline for both is ONE entry, `npp_hrd`, literally named
        "Suomi NPP / JPSS-1 HRD": constellation qpsk, symbolrate 15e6,
        rrc_alpha 0.5, pll_bw 0.002. (NOAA-21/JPSS-2 is a DIFFERENT entry,
        `jpss_hrd`: OQPSK, 25 Msym/s, 40 Msps, 1279-byte CADU.) Noaa20Profile's
        costas_bw = 0.002 was taken from npp_hrd in the first place, i.e. from
        the SNPP-and-NOAA-20 pipeline.
      - NOAA OSPO: "S-NPP and NOAA-20: 15 Mbps (7812 MHz)"; NOAA-21 and beyond
        increase to 25 Mbps.
      - NASA DRL's RT-STPS configs: npp.xml and jpss1.xml are structurally
        identical (frameLength 1024, interleave 4, standard CCSDS RS),
        differing only in spacecraft ID (157 vs 159). jpss2.xml differs for
        real (frameLength 1279, interleave 5).
      - This station already decoded SNPP with these exact numbers:
        recordings/past_recordings/snpp_20260709_1814.log -- 24 Msps, sps
        1.600000, Resample 0, 52x "Viterbi : SYNCED", BER down to 0.002441,
        "NORAD : 37849 / Name : Suomi NPP", VIIRS + OMPS products saved.

    Sample rate: 24 Msps, NOT the 25e6 that SatDump's pipeline defaults to.
    Occupied BW is (1+0.5) x 15 = 22.5 MHz, so 24 Msps leaves 750 kHz of guard
    per side against ~190 kHz of Doppler + transmitter tolerance (~4x margin).
    24 MHz is also a verified-exact master clock rate on this B206mini
    (UHD logs "Actually got clock rate 24.000000 MHz" -> sps 1.600000 exactly);
    25 MHz is not verified, and the extra 4.2% of samples would eat about a
    quarter of the chain's remaining real-time margin for nothing.

    OPERATIONAL NOTE: NOAA disables SNPP HRD Direct Broadcast after
    2026-11-01 13:00 UTC. After that this profile is dead weight -- it is kept
    only so the history above is not lost.
    """
    name = "snpp"


PROFILES = {
    "metop":  MetopProfile,
    "noaa20": Noaa20Profile,
    "snpp":   SnppProfile,
}

# Symbol rate above which --profile auto picks an HRD profile. Metop is
# 2.33 Msym/s and the HRD birds are 15 Msym/s, so anything in between
# separates them; 10 Msym/s leaves room for both to drift without
# reclassifying.
AUTO_PROFILE_SYMRATE_THRESHOLD = 10e6

# ...but symbol rate CANNOT tell SNPP from NOAA-20 -- they are the same
# waveform, which is the whole point of SnppProfile. So 'auto' resolves every
# 15 Msym/s pass to 'noaa20'. That is functionally correct (SnppProfile adds no
# behaviour), it just means the log says 'noaa20' on an SNPP pass. Pass
# --profile snpp explicitly to get the right label; live_pass_scheduler.py
# does this via each satellite's demod_profile key.
AUTO_HRD_PROFILE = "noaa20"


def resolve_profile(requested, symbol_rate):
    if requested != "auto":
        return PROFILES[requested]()
    chosen = (AUTO_HRD_PROFILE if symbol_rate >= AUTO_PROFILE_SYMRATE_THRESHOLD
              else "metop")
    eprint(f"[*] --profile auto -> '{chosen}' (symrate {symbol_rate/1e6:.4f} Msym/s)"
           + ("  [note: SNPP and NOAA-20 share this waveform; pass "
              "--profile snpp for an SNPP-labelled log]"
              if chosen == AUTO_HRD_PROFILE else ""))
    return PROFILES[chosen]()


# ============================================================================
# FLOWGRAPH
# ============================================================================

class live_uhd_demod(gr.top_block):
    def __init__(self, profile, freq, rate, gain, ant, duration, symbol_rate,
                 outfile, archive=None, archive_format="cf32", dev_args=""):
        gr.top_block.__init__(self, f"live UHD -> soft symbols [{profile.name}]")
        self.profile = profile

        # ---- device + streamer -------------------------------------------
        addr = profile.device_args(dev_args)
        dev = uhd.device_addr_t(addr)
        st = uhd.stream_args_t(cpu_format=profile.cpu_format,
                               otw_format=profile.otw_format)
        st.channels = [0]

        eprint(f"[*] profile '{profile.name}' | opening USRP ({addr}) | "
               f"otw={profile.otw_format} cpu={profile.cpu_format}")
        self.src = uhd.usrp_source(dev, st)
        self.src.set_samp_rate(rate)
        self.src.set_center_freq(uhd.tune_request(freq), 0)
        self.src.set_gain(gain, 0)
        if ant:
            self.src.set_antenna(ant, 0)
        # one_packet=True (gr-uhd's default) returns after a single ~2040-sample
        # packet, i.e. ~11,700 recv() calls/s at 24 Msps. The gr-uhd docs call
        # that "lower latency, but higher CPU load" and recommend a high timeout
        # for high-throughput use. False lets one recv() fill the whole output
        # buffer -- the same change uhd_record2.py made over v1. Measured here:
        # 79.7s -> 73.7s CPU over a 30 s capture at 24 Msps.
        self.src.set_recv_timeout(profile.recv_timeout, profile.one_packet)

        actual_rate = self.src.get_samp_rate()
        actual_freq = self.src.get_center_freq(0)
        actual_gain = self.src.get_gain(0)
        info = self.src.get_usrp_info()          # gr-uhd returns a plain dict
        eprint(f"[*] {info.get('mboard_id', '?')} serial={info.get('mboard_serial', '?')}")
        eprint(f"[*] rate {actual_rate/1e6:.4f} Msps | freq {actual_freq/1e6:.4f} MHz | "
               f"gain {actual_gain:.1f} dB | ant {self.src.get_antenna(0)}")
        if abs(actual_rate - rate) > 1.0:
            eprint(f"[!] requested {rate} Hz but device gave {actual_rate} Hz -- "
                   f"sps would be {actual_rate/symbol_rate:.6f}, not "
                   f"{rate/symbol_rate:.6f}")
        eprint(f"[*] sps = {actual_rate/symbol_rate:.6f} | symrate {symbol_rate:.4f} | "
               f"costas_bw {profile.costas_bw} | {duration}s | "
               f"one_packet={profile.one_packet}")

        # ---- bound the per-work() chunk handed to usrp_source --------------
        # Without this the scheduler may offer the entire output buffer, so a
        # single recv() can block for the full recv_timeout before ANY samples
        # reach the downstream threads. See Noaa20Profile's docstring.
        if profile.max_noutput_items:
            n = int(actual_rate * profile.max_noutput_items)
            self.src.set_max_noutput_items(n)
            eprint(f"[*] max_noutput_items: {n} items = "
                   f"{profile.max_noutput_items*1e3:.0f} ms per work() "
                   f"(~{1.0/profile.max_noutput_items:.0f} recv/s)")

        # ---- elastic buffer between the radio and the demod ---------------
        # A GNU Radio stream connection defaults to 8191 items -- 65 kB, which
        # is 0.34 ms at 24 Msps. Add UHD's own host ring and the total cushion
        # is still small, so any stall longer than that backpressures
        # usrp_source and the device drops samples for good.
        #
        # Sizing the source's OUTPUT buffer is what counts: it is the room
        # usrp_source has to keep pulling from UHD while everything downstream
        # is backed up.
        #
        # But note what this can and cannot do: it converts ONE stall into one
        # overflow burst lasting however long the buffer takes to drain at the
        # chain's real-time margin. On NOAA-20 that is ~14s for 2.0s of buffer.
        # Bigger is NOT better here -- read Noaa20Profile's docstring.
        if profile.buffer_seconds > 0:
            buf_items = int(actual_rate * profile.buffer_seconds)
            self.src.set_min_output_buffer(buf_items)
            eprint(f"[*] source buffer: {buf_items} items = "
                   f"{buf_items*8/1e6:.0f} MB = {profile.buffer_seconds:.1f}s of slack "
                   f"(default would be 8191 items = "
                   f"{8191/actual_rate*1e3:.2f} ms)")
        else:
            eprint("[!] buffer_seconds=0 -- using GNU Radio's 8191-item default "
                   "(~0.34 ms at 24 Msps); expect overflows above ~10 Msps")

        # ---- profile wires the rest ---------------------------------------
        profile.build_chain(self, symbol_rate, actual_rate, duration, outfile,
                            archive, archive_format)


def poll_telemetry(tb, interval, t0, stop_event):
    while not stop_event.wait(interval):
        try:
            eprint(f"[telemetry] t={time.time()-t0:6.1f}s  "
                   f"costas_freq={tb.costas.get_frequency():+.6f} rad/samp  "
                   f"clock_omega={tb.clock.omega():.4f}  clock_mu={tb.clock.mu():.4f}")
        except Exception as e:
            eprint(f"[telemetry] poll error: {e}")
            return


def main():
    ap = argparse.ArgumentParser(
        description="Live B206 -> soft symbols in one GNU Radio process (no ring)")
    ap.add_argument("--profile", choices=["auto"] + sorted(PROFILES), default="auto",
                    help="per-satellite transport tuning + flowgraph wiring. "
                         "'auto' picks by symbol rate (>=10 Msym/s -> noaa20).")
    ap.add_argument("--freq", type=float, required=True, help="IF the radio tunes to (Hz)")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--gain", type=float, required=True)
    ap.add_argument("--ant", type=str, default="RX2")
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--symbol-rate", type=float, required=True)
    ap.add_argument("--out", required=True, help="soft-symbol .s8 output")
    ap.add_argument("--archive", default=None,
                    help="also write the raw baseband here (optional but recommended)")
    ap.add_argument("--archive-format", choices=["cf32", "cs16"], default="cf32")

    # Every tunable below defaults to None so the profile's value is used
    # unless the caller actually passed the flag. Do not give these argparse
    # defaults -- that would silently override the profile on every run.
    ap.add_argument("--rrc-alpha", type=float, default=None)
    ap.add_argument("--rrc-taps", type=int, default=None)
    ap.add_argument("--costas-bw", type=float, default=None)
    ap.add_argument("--clock-gain-mu", type=float, default=None)
    ap.add_argument("--clock-mu", type=float, default=None)
    ap.add_argument("--clock-omega-rel-limit", type=float, default=None)
    ap.add_argument("--agc-rate", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--num-recv-frames", type=int, default=None)
    ap.add_argument("--recv-frame-size", type=int, default=None,
                    help="bytes per USB frame; UHD's B200 max is 16360. "
                         "Bigger frames = fewer URB completions per second.")
    ap.add_argument("--otw-format", choices=["sc16", "sc8"], default=None,
                    help="over-the-wire sample format. sc8 halves bus load and "
                         "host conversion cost for ~6 dB of dynamic range -- "
                         "worth A/B testing on NOAA-20, not yet the default.")
    ap.add_argument("--buffer-seconds", type=float, default=None,
                    help="seconds of elastic buffer on the USRP source output "
                         "(0 = GNU Radio's 8191-item default). NOTE: raising "
                         "this LENGTHENS overflow bursts on a CPU-bound chain; "
                         "read Noaa20Profile's docstring before increasing it.")
    ap.add_argument("--max-noutput-items", type=float, default=None,
                    help="seconds of samples per usrp_source work() call "
                         "(0 = GNU Radio's default, effectively unbounded)")
    ap.add_argument("--one-packet", action="store_true", default=None,
                    help="receive one packet per recv() (gr-uhd's default). "
                         "Lower latency, higher CPU -- off here on purpose.")
    ap.add_argument("--args", dest="dev_args", type=str, default="",
                    help="extra UHD device args, e.g. 'serial=xxxx'")
    ap.add_argument("--grace-period", type=float, default=20.0,
                    help="hard stop this many seconds past duration no matter what")
    ap.add_argument("--telemetry-interval", type=float, default=5.0)
    args = ap.parse_args()

    profile = resolve_profile(args.profile, args.symbol_rate)

    # Overlay explicit CLI values onto the profile. Flag name -> attribute.
    overrides = {
        "rrc_alpha": "rrc_alpha", "rrc_taps": "rrc_taps",
        "costas_bw": "costas_bw", "clock_gain_mu": "clock_gain_mu",
        "clock_mu": "clock_mu", "clock_omega_rel_limit": "clock_omega_rel_limit",
        "agc_rate": "agc_rate", "scale": "scale",
        "num_recv_frames": "num_recv_frames", "recv_frame_size": "recv_frame_size",
        "otw_format": "otw_format", "buffer_seconds": "buffer_seconds",
        "max_noutput_items": "max_noutput_items", "one_packet": "one_packet",
    }
    for flag, attr in overrides.items():
        val = getattr(args, flag)
        if val is not None:
            if getattr(profile, attr) != val:
                eprint(f"[*] override: {attr} {getattr(profile, attr)} -> {val}")
            setattr(profile, attr, val)

    # Ask for RT scheduling before building the flowgraph. Needs a nonzero
    # RLIMIT_RTPRIO. /etc/security/limits.d/99-usrp.conf already grants
    # '@usrp - rtprio 99', so the fix is group membership, not another file:
    #     sudo usermod -aG usrp $USER   (then log out and back in)
    # Without it the demod threads AND UHD's libusb transport thread stay
    # SCHED_OTHER and fully preemptible -- which is the leading suspect for the
    # stalls that kick off NOAA-20's overflow bursts.
    rt = gr.enable_realtime_scheduling()
    eprint(f"[*] real-time scheduling: {rt}"
           + ("" if rt == gr.RT_OK else
              "  (run 'sudo usermod -aG usrp $USER' and re-login; "
              "limits.d already grants rtprio 99 to @usrp)"))

    tb = live_uhd_demod(
        profile, args.freq, args.rate, args.gain, args.ant, args.duration,
        args.symbol_rate, args.out, args.archive, args.archive_format,
        args.dev_args)

    t0 = time.time()
    stop_event = threading.Event()
    tel = None
    if args.telemetry_interval > 0:
        tel = threading.Thread(target=poll_telemetry,
                               args=(tb, args.telemetry_interval, t0, stop_event),
                               daemon=True)

    # Watchdog: head caps the sample count, but if the device stalls or drops
    # samples head may never fill. Stop on the wall clock regardless so a pass
    # can never run past its window. On the noaa20 profile this is the PRIMARY
    # duration bound, since its head sits on the output stream.
    def watchdog():
        deadline = t0 + args.duration + args.grace_period
        while not stop_event.wait(0.5):
            if time.time() >= deadline:
                eprint(f"[!] watchdog: {args.duration + args.grace_period:.0f}s "
                       f"elapsed, stopping flowgraph")
                tb.stop()
                return
    wd = threading.Thread(target=watchdog, daemon=True)

    tb.start()
    if tel:
        tel.start()
    wd.start()
    tb.wait()
    stop_event.set()
    if tel:
        tel.join(timeout=2.0)

    elapsed = time.time() - t0
    n_s8 = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    eprint(f"\n[+] Done in {elapsed:.1f}s. Soft symbols -> {args.out} "
           f"({n_s8/1e9:.2f} GB, {n_s8//2} symbols; "
           f"{100.0*(n_s8//2)/max(1, args.duration*args.symbol_rate):.1f}% of nominal)")
    if args.archive and os.path.exists(args.archive):
        n_bb = os.path.getsize(args.archive)
        bps = 8 if args.archive_format == "cf32" else 4
        eprint(f"[+] Baseband archive -> {args.archive} ({n_bb/1e9:.2f} GB, "
               f"{n_bb//bps} samples; "
               f"{100.0*(n_bb//bps)/max(1, args.duration*args.rate):.1f}% of nominal)")
    # The percentages above are NOT loss indicators: an overflow leaves a
    # splice in the stream, not a shortfall, so a pass that dropped 1% of its
    # samples still reports ~100% of nominal. Count the 'O' markers instead.
    eprint("[*] NOTE: gr-uhd prints 'O' to stdout on overflow -- grep the log for it.")


if __name__ == "__main__":
    main()

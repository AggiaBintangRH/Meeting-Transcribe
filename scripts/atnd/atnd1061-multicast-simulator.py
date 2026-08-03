#!/usr/bin/env python3
"""
ATND1061 multicast notification simulator — a fake array for developing
without hardware. Companion to atnd1061-tcp-simulator.py (TCP is the control
side; this is the notification side).

Sends both UDP multicast notifications from IP Control spec v9.0 section 5,
in the spec's exact wire format (ASCII, space-delimited, CR-terminated):

    MD camera_control_notice 0000 00 NC <status>,<ch>,<angle>,<rotate>,<cam> CR
    MD level_meter_notice    0000 00 NC <42 comma-separated fields> CR

Two speakers can be active at once, because that is the whole point here: the
array's own talker notification names only ONE speaker even during overlap
(it drives a camera), while the level meter shows every hot beam. Overlap is
therefore only visible in level_meter_notice — this simulator reproduces that
asymmetry rather than hiding it.

Keys 1-6 TOGGLE each beam, so holding two on is genuine simultaneous speech.

Usage:
    python3 scripts/atnd/atnd1061-multicast-simulator.py
    python3 scripts/atnd/atnd1061-multicast-simulator.py --group 239.0.0.100 --port 17000
    python3 scripts/atnd/atnd1061-multicast-simulator.py --source-ip 192.168.1.50
    python3 scripts/atnd/atnd1061-multicast-simulator.py --switch-jitter 8   # settle after switch
    python3 scripts/atnd/atnd1061-multicast-simulator.py --beam-override 2:33:200  # move a beam
"""
from __future__ import annotations

import argparse
import contextlib
import random
import select
import socket
import sys
import time
from dataclasses import dataclass, replace

CR = b"\x0d"

LEVEL_FIELDS = 42          # spec section 5.2.1, verified against its example
BEAM_LEVEL_MAX = 61        # post-fader meters are 0..61
GAIN_SHARE_MAX = 15        # gain-share meters are 0..15

# Level meter field map (0-indexed), from the spec's table:
#   0-5   Beam Channel 1-6 (post fader)     12  Analog Out
#   6     Analog Input                      13  Auto Mix
#   7-11  Reserved                          14  Voice Lift
#   15-21 Reserved                          22  AEC (ERL)
#   23-31 Reserved                          32-37 Gain Share Beam 1-6
#   38-41 Reserved
BEAM_SLOTS = range(0, 6)
ANALOG_IN_SLOT = 6
AUTO_MIX_SLOT = 13
GAIN_SHARE_SLOTS = range(32, 38)


@dataclass(frozen=True)
class Beam:
    label: str
    channel: int   # 0-5 on the wire = Beam Channel 1-6
    angle: int     # 0-90 elevation
    rotate: int    # 0-360
    camera: int    # camera area 1-15


# Six beams — the array has exactly six, no more.
BEAMS: dict[str, Beam] = {
    "1": Beam("Speaker A", 0, 48, 156, 1),
    "2": Beam("Speaker B", 1, 33, 195, 2),
    "3": Beam("Speaker C", 2, 60, 225, 3),
    "4": Beam("Speaker D", 3, 25, 110, 4),
    "5": Beam("Speaker E", 4, 75, 300, 5),
    "6": Beam("Speaker F", 5, 42, 20, 6),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fake ATND1061 multicast notifications.")
    p.add_argument("--group", default="239.0.0.100", help="Multicast group (array default).")
    p.add_argument("--port", type=int, default=17000, help="UDP port (array default).")
    p.add_argument("--source-ip", default="",
                   help="Local interface IP to send from. Empty = system default.")
    p.add_argument("--interval", type=float, default=0.1,
                   help="Seconds between notifications (array default 0.1).")
    p.add_argument("--ttl", type=int, default=1, help="Multicast TTL.")
    p.add_argument("--no-levels", action="store_true",
                   help="Send only camera_control_notice, as if Audio Level Notification were 0.")
    p.add_argument("--no-talker", action="store_true",
                   help="Send only level_meter_notice, as if Camera Control Notification were 0.")
    p.add_argument("--beams", default="",
                   help="Beams active at startup, e.g. 1 or 1,2 (two = overlap). "
                        "Needed when there is no terminal to press keys in.")
    p.add_argument("--switch-jitter", type=int, default=0, metavar="N",
                   help="For N samples after the active beam set changes, widen the "
                        "talker-direction noise (models the array settling after a "
                        "talker switch) so the diarizer's smoothing can be exercised.")
    p.add_argument("--beam-override", action="append", default=[], metavar="CH:ANGLE:ROTATE",
                   help="Redefine a beam's direction, e.g. 2:33:200. Repeatable. "
                        "Use it to place two beams at nearly the same bearing and test "
                        "the merge threshold.")
    return p.parse_args()


def apply_overrides(specs: list[str]) -> None:
    """Rewrite BEAMS[key] angle/rotate from CH:ANGLE:ROTATE specs."""
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--beam-override: {spec!r} must be CH:ANGLE:ROTATE")
        key, angle_s, rotate_s = (p.strip() for p in parts)
        if key not in BEAMS:
            raise SystemExit(f"--beam-override: beam {key!r} is not 1-6")
        try:
            angle, rotate = int(angle_s), int(rotate_s)
        except ValueError:
            raise SystemExit(f"--beam-override: {spec!r} angle/rotate must be integers")
        if not (0 <= angle <= 90 and 0 <= rotate <= 360):
            raise SystemExit(f"--beam-override: {spec!r} angle 0-90, rotate 0-360")
        BEAMS[key] = replace(BEAMS[key], angle=angle, rotate=rotate)


def clamp(value: int, lo: int, hi: int) -> int:
    return min(hi, max(lo, value))


def camera_notice(active: list[Beam], jitter: bool = False) -> bytes:
    """Talker position. Reports ONE speaker even when several are active —
    the real array picks a single target for the camera to point at.

    `jitter=True` widens the per-sample noise, modelling the array settling
    for a few samples right after a talker switch."""
    if not active:
        payload = "0,,,,"
    else:
        b = active[0]  # dominant talker
        a_amp, r_amp = (15, 20) if jitter else (2, 3)
        angle = clamp(b.angle + random.randint(-a_amp, a_amp), 0, 90)
        rotate = (b.rotate + random.randint(-r_amp, r_amp)) % 360
        payload = f"1,{b.channel},{angle},{rotate},{b.camera}"
    # Trailing SPACE before the CR — the real array sends it (measured: the
    # silence notice is 44 bytes for 42 bytes of text). The simulator omitted it
    # and so agreed with a parser that the device did not, hiding a total
    # receive failure. Keep it: a faithful simulator is the only thing that
    # makes these tests mean anything.
    return f"MD camera_control_notice 0000 00 NC {payload} ".encode() + CR


def level_notice(active: list[Beam]) -> bytes:
    """Per-beam levels. Every active beam reads hot — this is the only
    notification in which simultaneous speech is visible."""
    fields = [""] * LEVEL_FIELDS
    hot = {b.channel for b in active}

    for ch in BEAM_SLOTS:
        if ch in hot:
            level = clamp(random.randint(42, 55), 0, BEAM_LEVEL_MAX)
        else:
            level = random.randint(0, 6)  # room noise floor
        fields[ch] = str(level)

    for slot, ch in zip(GAIN_SHARE_SLOTS, BEAM_SLOTS):
        fields[slot] = str(random.randint(9, GAIN_SHARE_MAX) if ch in hot else 0)

    beam_levels = [int(fields[ch]) for ch in BEAM_SLOTS]
    fields[ANALOG_IN_SLOT] = str(max(beam_levels))
    fields[AUTO_MIX_SLOT] = str(max(beam_levels))  # the mixed-down channel

    return f"MD level_meter_notice 0000 00 NC {','.join(fields)} ".encode() + CR


def send(sock: socket.socket, payload: bytes, destination, state: dict) -> None:
    """Send one notification, surviving transient routing errors.

    macOS clones the multicast route on first use, so the first send can fail
    with EHOSTUNREACH ("No route to host") and succeed a moment later. A real
    array would not vanish because of that, so neither should this — log once
    and keep going rather than dying mid-session.
    """
    try:
        sock.sendto(payload, destination)
        if state.get("failing"):
            print("  ... sending again", flush=True)
            state["failing"] = False
    except OSError as exc:
        if not state.get("failing"):
            print(f"  cannot send: {exc}", flush=True)
            print("    Retrying. If this never clears, pass --source-ip with this "
                  "Mac's IP on the array's network.", flush=True)
            state["failing"] = True


def make_socket(source_ip: str, ttl: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    if source_ip:
        sock.bind((source_ip, 0))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(source_ip))
    return sock


@contextlib.contextmanager
def key_reader():
    """Live keys when run from a terminal; a no-op reader otherwise, so the
    simulator still runs in the background or under a test harness."""
    import termios
    import tty
    if not sys.stdin.isatty():
        yield lambda: None
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield lambda: sys.stdin.read(1).lower() if select.select([sys.stdin], [], [], 0)[0] else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def status_line(active: list[Beam]) -> str:
    if not active:
        return "silence"
    names = " + ".join(f"{b.label} (CH{b.channel + 1})" for b in active)
    tag = "  <-- OVERLAP" if len(active) > 1 else ""
    return names + tag


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than 0")
    if args.no_levels and args.no_talker:
        raise SystemExit("--no-levels and --no-talker together would send nothing")

    if args.switch_jitter < 0:
        raise SystemExit("--switch-jitter must be 0 or more")
    apply_overrides(args.beam_override)

    sock = make_socket(args.source_ip, args.ttl)
    destination = (args.group, args.port)

    active: list[Beam] = []
    for key in filter(None, (k.strip() for k in args.beams.split(","))):
        if key not in BEAMS:
            raise SystemExit(f"--beams: {key!r} is not a beam (use 1-6)")
        active.append(BEAMS[key])

    print("ATND1061 multicast simulator")
    print(f"  destination : {args.group}:{args.port}")
    print(f"  source IP   : {args.source_ip or 'system default'}")
    print(f"  interval    : {args.interval * 1000:.0f} ms")
    print(f"  sending     : "
          + ", ".join(filter(None, [
              None if args.no_talker else "camera_control_notice",
              None if args.no_levels else "level_meter_notice"])))
    print()
    if sys.stdin.isatty():
        print("  Keys : 1-6 toggle a beam (turn on two = overlap), 0 silence, q quit")
    else:
        print("  No terminal — keys are off; use --beams to choose who speaks")
    print(f"  {status_line(active)}")
    print()

    next_send = 0.0
    send_state: dict = {}
    jitter_left = 0   # samples of widened noise still owed after a switch
    with key_reader() as read_key:
        while True:
            now = time.monotonic()
            key = read_key()
            if key == "q":
                print("\nstopped")
                return 0
            if key == "0":
                active = []
                jitter_left = args.switch_jitter
                print(f"  {status_line(active)}")
            elif key in BEAMS:
                beam = BEAMS[key]
                if beam in active:
                    active.remove(beam)
                else:
                    active.append(beam)
                jitter_left = args.switch_jitter
                print(f"  {status_line(active)}")

            if now >= next_send:
                if not args.no_talker:
                    send(sock, camera_notice(active, jitter=jitter_left > 0),
                         destination, send_state)
                    if jitter_left > 0:
                        jitter_left -= 1
                if not args.no_levels:
                    send(sock, level_notice(active), destination, send_state)
                next_send = now + args.interval

            time.sleep(0.01)


if __name__ == "__main__":
    raise SystemExit(main())

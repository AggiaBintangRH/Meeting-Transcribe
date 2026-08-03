#!/usr/bin/env python3
"""
ATND1061 TCP control simulator — a fake array for developing without hardware.

Speaks the IP Control spec v9.0 TCP protocol (section 2.2): ASCII messages,
space-delimited, CR-terminated (single 0x0d, never CRLF).

    action : <command> <handshake> <model_id> <unit_no> <continue> [params] CR
    ACK    : <command> ACK CR
    NAK    : <command> NAK <error_code> CR
    answer : <command> <model_id> <unit_no> <continue> <params> CR

Handshake is S (expects ACK/NAK) or O (Get, expects an Answer). Error codes
(Table 2-6): 01 syntax, 02 invalid command, 03 split error, 04 parameter,
05 timeout, 90/92/93 busy, 99 other.

Only the commands the app actually needs are implemented; anything else in the
spec's 111-command list gets NAK 02, which is what a real array does for a
command it does not know.

Sparse sets are honoured everywhere: an empty field means "leave unchanged"
(spec section 2, "When you send a command from the host, you can omit its
parameters").

Deliberately NOT implemented: s_network_dante and s_powersave — the spec says
they reboot the device, so they are out of scope for this app.

Usage:
    python3 scripts/atnd/atnd1061-tcp-simulator.py                 # 127.0.0.1:17300
    python3 scripts/atnd/atnd1061-tcp-simulator.py --port 17300 --host 0.0.0.0
    python3 scripts/atnd/atnd1061-tcp-simulator.py --reject        # refuse commands
    python3 scripts/atnd/atnd1061-tcp-simulator.py --silent        # accept, never reply
    python3 scripts/atnd/atnd1061-tcp-simulator.py --delay-ms 4000 # late replies
    python3 scripts/atnd/atnd1061-tcp-simulator.py --aec-odd       # spec's own broken AEC example
"""
from __future__ import annotations

import argparse
import socketserver
import sys
import threading
import time

CR = b"\x0d"

# Device state the simulator hands back, in the spec's own field order.
# Mirrors the ATND1061 factory defaults (Table 5-1: 239.0.0.100 : 17000).
DEVICE = {
    "ip_config_mode": "1",
    "ip_address": "192.168.033.102",
    "subnet_mask": "255.255.000.000",
    "gateway": "",
    "mac_address": "0005CDC102FA",
    "allow_discovery": "1",
    "control_port": "17300",
    "notification": "1",
    "audio_level_notification": "1",
    "multicast_address": "239.000.000.100",
    "multicast_port": "17000",
    "camera_control_notification": "1",
    "talker_interval": "100",
    "level_interval": "100",
    # Table 4-46 answer: Device ID is a hex byte 00–FF (the spec's example is 08),
    # not a friendly name.
    "device_id": "08",
    # s_mute / g_mute (Table 4-124/4-126): 0 not muted, 1 muted.
    "mute": "0",
    # s_roomtype / g_roomtype (Table 4-136/4-138): 0 Dry, 1 Live, 2 Reverberant.
    "roomtype": "1",
    # SVAD (Table 4-17) is Set-only — there is no GVAD, so a real array never
    # reports this back. Stored purely so the log shows the last value.
    "vad": "0",
}

# s_output_mute / g_output_mute (Table 4-98/4-100): <channel>,<mute>.
# Channel 0 Analog Out, 1 Auto Mix.
OUTPUT_MUTE = {"0": "0", "1": "0"}

# s_agc / g_agc (Table 4-37/4-39): 0 Enable(0/1), 1 Reserved, 2 Target Level(-10..10).
AGC = ["1", "", "2"]

# s_aec_general / g_aec_general (Table 4-34/4-36) — the FULL 12-field leaf list:
#  0 AEC Enable(0/1), 1 AEC Reference(0/1), 2-5 Reserved,
#  6 Reserved (Noise Canceling group), 7 NC Attenuation Level(0..20),
#  8 NLP Enable(0/1), 9 NLP Sensitivity(0..2), 10 Reserved, 11 NC Enable(0/1).
# NOTE: the spec's own example for this command ("2,1,,,0,,20,20,1,1,") has only
# ELEVEN fields and a field 0 of "2", which is outside AEC Enable's documented
# 0/1 domain — the table and the example contradict each other. --aec-odd serves
# that example verbatim so the app's AEC gate can be tested against it.
AEC = ["1", "0", "", "", "", "", "", "20", "1", "1", "", "0"]
AEC_ODD = "2,1,,,0,,20,20,1,1,"

# s_audio_system / g_audio_system (Table 4-120/4-122): 0-3 Reserved, 4 Tx6(0/1),
# 5 Beam Sensitivity(0 Low/1 Mid/2 High), 6 Auto Attenuation(0/1),
# 7 Attenuation Level(0..28), 8 Hold Time(0..100).
AUDIO_SYSTEM = ["", "", "", "", "1", "2", "1", "28", "100"]

# s_smart_mix / g_smart_mix (Table 4-31/4-33), per beam channel 0..5:
# 0 Input Channel(0..5), 1 Reserved, 2 Gain Share Weight(0..60 => -15.0..+15.0 dB),
# 3-6 Reserved. Only the weight is state; the rest is echoed back.
SMART_MIX = {str(ch): "30" for ch in range(6)}

_lock = threading.Lock()


def log(direction: str, text: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    arrow = "<--" if direction == "rx" else "-->"
    print(f"[{stamp}] {arrow} {text!r}", flush=True)


def ack(cmd: str) -> bytes:
    return f"{cmd} ACK".encode() + CR


def nak(cmd: str, code: str) -> bytes:
    return f"{cmd} NAK {code}".encode() + CR


def answer(cmd: str, params: str) -> bytes:
    # Unit No is the Device ID (00–FF), not a fixed 00 — Tables 2-7, 4-72 etc.
    return f"{cmd} 0000 {DEVICE['device_id']} NC {params}".encode() + CR


# MARK: - validation helpers


def _int_in(value: str, low: int, high: int) -> bool:
    try:
        return low <= int(value) <= high
    except ValueError:
        return False


def _apply_sparse(target: list[str], fields: list[str], rules: dict[int, tuple[int, int]]) -> bool:
    """Write non-empty fields onto `target` in place. False = out of range (NAK 04).

    Empty field = unchanged (spec section 2). Fields with no rule are stored
    unvalidated — they are Reserved and the array's own domain is undocumented.
    """
    for idx, value in enumerate(fields):
        if idx >= len(target) or not value:
            continue
        rule = rules.get(idx)
        if rule and not _int_in(value, rule[0], rule[1]):
            return False
    for idx, value in enumerate(fields):
        if idx < len(target) and value:
            target[idx] = value
    return True


def g_network_params() -> str:
    """Answer for g_network — exactly 21 fields (Table 4-72).

    Get and Set layouts DIFFER: the MAC address exists only in the Get answer
    (index 4), which shifts everything after it by one relative to s_network.
    """
    d = DEVICE
    fields = [""] * 21
    fields[0] = d["ip_config_mode"]
    fields[1] = d["ip_address"]
    fields[2] = d["subnet_mask"]
    fields[3] = d["gateway"]
    fields[4] = d["mac_address"]            # Get only — absent from s_network
    fields[5] = d["allow_discovery"]
    fields[6] = d["control_port"]
    fields[7] = d["notification"]
    fields[8] = d["audio_level_notification"]
    fields[9] = d["multicast_address"]
    fields[10] = d["multicast_port"]
    # 11–19 Reserved (9 fields)
    fields[20] = d["camera_control_notification"]
    return ",".join(fields)


def handle_message(raw: str, opts) -> bytes | None:
    """Map one CR-terminated message to the array's reply (None = stay silent)."""
    parts = raw.split(" ")
    cmd = parts[0] if parts else ""

    if not cmd:
        return None
    if opts.reject:
        return nak(cmd, "99")

    # <command> <handshake> <model> <unit> <continue> [params]
    if len(parts) < 5:
        return nak(cmd, "01")

    handshake, model_id, unit_no, cont = parts[1], parts[2], parts[3], parts[4]
    params = parts[5] if len(parts) > 5 else ""

    if handshake not in ("S", "O", "H"):
        return nak(cmd, "01")
    if model_id != "0000" or cont not in ("NC", "CS", "CM", "CE"):
        return nak(cmd, "01")

    # MARK: network

    if cmd == "g_network":
        return answer(cmd, g_network_params())

    if cmd == "s_network":
        # SET layout — 20 fields, no MAC (Table 4-70). Do not reuse the Get
        # indices here: they are off by one from index 4 onward.
        fields = params.split(",")
        if len(fields) < 20:
            return nak(cmd, "04")
        for idx, low, high in ((5, 1, 65535), (6, 0, 1), (7, 0, 1), (9, 1, 65535), (19, 0, 1)):
            if fields[idx] and not _int_in(fields[idx], low, high):
                return nak(cmd, "04")
        with _lock:
            for key, idx in (
                ("ip_config_mode", 0), ("ip_address", 1), ("subnet_mask", 2),
                ("gateway", 3), ("allow_discovery", 4), ("control_port", 5),
                ("notification", 6), ("audio_level_notification", 7),
                ("multicast_address", 8), ("multicast_port", 9),
                ("camera_control_notification", 19),
            ):
                if fields[idx]:
                    DEVICE[key] = fields[idx]
        return ack(cmd)

    # MARK: mute

    if cmd == "g_mute":
        return answer(cmd, DEVICE["mute"])

    if cmd == "s_mute":
        if params not in ("0", "1"):
            return nak(cmd, "04")
        with _lock:
            DEVICE["mute"] = params
        return ack(cmd)

    if cmd == "g_output_mute":
        # The Get takes the channel; without it the array has nothing to answer.
        if params not in OUTPUT_MUTE:
            return nak(cmd, "04")
        with _lock:
            return answer(cmd, f"{params},{OUTPUT_MUTE[params]}")

    if cmd == "s_output_mute":
        fields = params.split(",")
        if len(fields) != 2 or fields[0] not in OUTPUT_MUTE or fields[1] not in ("0", "1", ""):
            return nak(cmd, "04")
        with _lock:
            if fields[1]:
                OUTPUT_MUTE[fields[0]] = fields[1]
        return ack(cmd)

    # MARK: audio processing

    if cmd == "g_agc":
        with _lock:
            return answer(cmd, ",".join(AGC))

    if cmd == "s_agc":
        fields = params.split(",")
        if len(fields) != 3:
            return nak(cmd, "04")
        with _lock:
            if not _apply_sparse(AGC, fields, {0: (0, 1), 2: (-10, 10)}):
                return nak(cmd, "04")
        return ack(cmd)

    if cmd == "g_aec_general":
        if opts.aec_odd:
            return answer(cmd, AEC_ODD)
        with _lock:
            return answer(cmd, ",".join(AEC))

    if cmd == "s_aec_general":
        fields = params.split(",")
        if len(fields) != 12:
            return nak(cmd, "04")
        with _lock:
            if not _apply_sparse(AEC, fields,
                                 {0: (0, 1), 1: (0, 1), 7: (0, 20), 8: (0, 1), 9: (0, 2), 11: (0, 1)}):
                return nak(cmd, "04")
        return ack(cmd)

    if cmd == "g_smart_mix":
        if params not in SMART_MIX:
            return nak(cmd, "04")
        with _lock:
            return answer(cmd, f"{params},,{SMART_MIX[params]},,,,")

    if cmd == "s_smart_mix":
        fields = params.split(",")
        if len(fields) != 7 or fields[0] not in SMART_MIX:
            return nak(cmd, "04")
        if fields[2] and not _int_in(fields[2], 0, 60):
            return nak(cmd, "04")
        with _lock:
            if fields[2]:
                SMART_MIX[fields[0]] = fields[2]
        return ack(cmd)

    if cmd == "g_audio_system":
        with _lock:
            return answer(cmd, ",".join(AUDIO_SYSTEM))

    if cmd == "s_audio_system":
        fields = params.split(",")
        if len(fields) != 9:
            return nak(cmd, "04")
        with _lock:
            if not _apply_sparse(AUDIO_SYSTEM, fields,
                                 {4: (0, 1), 5: (0, 2), 6: (0, 1), 7: (0, 28), 8: (0, 100)}):
                return nak(cmd, "04")
        return ack(cmd)

    if cmd == "g_roomtype":
        return answer(cmd, DEVICE["roomtype"])

    if cmd == "s_roomtype":
        if params not in ("0", "1", "2"):
            return nak(cmd, "04")
        with _lock:
            DEVICE["roomtype"] = params
        return ack(cmd)

    # MARK: notification intervals

    if cmd == "s_camera_control_interval":
        if not _int_in(params, 100, 300000):
            return nak(cmd, "04")
        with _lock:
            DEVICE["talker_interval"] = params
        return ack(cmd)

    if cmd == "g_camera_control_interval":
        return answer(cmd, DEVICE["talker_interval"])

    if cmd == "s_level_meter_interval":
        if not _int_in(params, 100, 300000):
            return nak(cmd, "04")
        with _lock:
            DEVICE["level_interval"] = params
        return ack(cmd)

    if cmd == "g_level_meter_interval":
        return answer(cmd, DEVICE["level_interval"])

    # MARK: misc

    if cmd == "SVAD":
        # Set-only: there is no GVAD, so the value is never readable back.
        if params not in ("0", "1"):
            return nak(cmd, "04")
        with _lock:
            DEVICE["vad"] = params
        return ack(cmd)

    if cmd == "g_deviceid":
        return answer(cmd, DEVICE["device_id"])

    return nak(cmd, "02")


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"*** connected: {peer}", flush=True)
        buf = b""
        try:
            while True:
                chunk = self.request.recv(4096)
                if not chunk:
                    break
                buf += chunk
                # The spec caps a message at 287 bytes including the CR; past
                # that without one, the peer is not speaking this protocol.
                if len(buf) > 287 and CR not in buf:
                    log("tx", "NAK 01 (message too long)")
                    self.request.sendall(nak("", "01"))
                    buf = b""
                    continue
                while CR in buf:
                    line, buf = buf.split(CR, 1)
                    raw = line.decode("utf-8", "replace")
                    log("rx", repr(raw))
                    if self.server.opts.silent:
                        continue
                    # The spec's wire format ALWAYS ends with a space before the
                    # CR — every example does, parameter or not (GDID␣O␣0000␣00␣
                    # NC␣↲ and MUTE␣S␣0000␣00␣NC␣1␣↲). A real ATND1061 rejects a
                    # line missing it with NAK 04; mirror that here so this double
                    # is FAITHFUL. The old .strip() hid exactly this: a Set that
                    # dropped its trailing parameter space passed the simulator
                    # but the real device NAK 04'd it.
                    if not raw.endswith(" "):
                        cmd = raw.split(" ", 1)[0]
                        log("tx", "NAK 04 (missing trailing space before CR)")
                        self.request.sendall(nak(cmd, "04"))
                        continue
                    text = raw[:-1]   # drop exactly the one protocol trailing space
                    reply = handle_message(text, self.server.opts)
                    if reply is None:
                        continue
                    # A late reply is the interesting case: past the app's
                    # request timeout it arrives with nothing waiting for it,
                    # which is exactly the desync the app must survive.
                    if self.server.opts.delay_ms:
                        time.sleep(self.server.opts.delay_ms / 1000)
                    log("tx", reply.rstrip(CR).decode())
                    self.request.sendall(reply)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            print(f"*** disconnected: {peer}", flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    p = argparse.ArgumentParser(description="Fake ATND1061 TCP control port.")
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Use 0.0.0.0 to accept from other machines.")
    p.add_argument("--port", type=int, default=17300, help="TCP port (array default 17300).")
    p.add_argument("--reject", action="store_true",
                   help="Answer every command with NAK 99 — tests the app's error path.")
    p.add_argument("--silent", action="store_true",
                   help="Accept the connection but never reply — tests the app's timeout path.")
    p.add_argument("--delay-ms", type=int, default=0,
                   help="Wait this long before every reply. Above 3000 the app's request timeout "
                        "fires first and the reply lands late, with nothing waiting for it.")
    p.add_argument("--aec-odd", action="store_true",
                   help="Serve the spec's own contradictory 11-field AEC example "
                        "('2,1,,,0,,20,20,1,1,') instead of the Table 4-34 layout.")
    opts = p.parse_args()

    server = Server((opts.host, opts.port), Handler)
    server.opts = opts

    modes = []
    if opts.reject:
        modes.append("reject")
    if opts.silent:
        modes.append("silent")
    if opts.delay_ms:
        modes.append(f"delay {opts.delay_ms}ms")
    if opts.aec_odd:
        modes.append("aec-odd")
    mode = f" [{', '.join(modes)}]" if modes else ""

    print(f"ATND1061 TCP simulator on {opts.host}:{opts.port}{mode}", flush=True)
    print(f"  device id : {DEVICE['device_id']}", flush=True)
    print(f"  multicast : {DEVICE['multicast_address']} : {DEVICE['multicast_port']}", flush=True)
    print("  Ctrl-C to stop\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import socket
import struct


DEFAULT_MCAST_GRP = "239.0.0.100"
DEFAULT_MCAST_PORT = 17000
# This Mac's own address on the array's network. 0.0.0.0 = let the kernel pick.
DEFAULT_INTERFACE_IP = "0.0.0.0"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Receive ATND1061 camera_control_notice multicast packets.")
    parser.add_argument("--group", default=DEFAULT_MCAST_GRP, help="Multicast group IP.")
    parser.add_argument("--port", type=int, default=DEFAULT_MCAST_PORT, help="Multicast UDP port.")
    # THIS MAC's address, not the array's. The old name (--device-ip) said the
    # opposite of what the value is used for and cost real debugging time: it is
    # passed to IP_ADD_MEMBERSHIP as imr_interface, i.e. which of this machine's
    # interfaces joins the group. Putting the ARRAY's IP here makes the join fail.
    # --device-ip still works so existing habits and notes keep running.
    parser.add_argument("--interface-ip", "--device-ip", dest="interface_ip",
                        default=DEFAULT_INTERFACE_IP,
                        help="THIS MAC's IP on the array's network (not the array's). "
                             "Use 0.0.0.0 to let the kernel choose.")
    return parser


def make_socket(group, port, interface_ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # On Mac, SO_REUSEPORT allows multiple listeners on same port
    if hasattr(socket, "SO_REUSEPORT"):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

    sock.bind(("", port))

    # Join multicast group on the specified interface
    mreq = struct.pack("4s4s", socket.inet_aton(group), socket.inet_aton(interface_ip))
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    return sock


def main():
    args = build_parser().parse_args()

    sock = make_socket(args.group, args.port, args.interface_ip)

    print("ATND1061 multicast receiver (Mac)")
    print(f"Group     : {args.group}")
    print(f"Port      : {args.port}")
    print(f"Interface : {args.interface_ip if args.interface_ip != '0.0.0.0' else '0.0.0.0 (auto)'}  (this Mac)")
    print("Waiting for packets... (Ctrl+C to stop)")
    print()

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            message = data.decode("utf-8").strip()
            print(f"[RECV from {addr[0]}:{addr[1]}] {message}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

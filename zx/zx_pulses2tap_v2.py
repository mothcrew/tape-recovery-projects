#!/usr/bin/env python3
"""
zx_pulses2tap_v2.py

Recover ordinary ZX Spectrum ROM-loader TAP blocks from tape2pulses output.

This version is intentionally based on PAIR SUMS.

For this recording, individual pulse lengths can vary substantially:
    790 + 790
    948 + 632
    711 + 869
    1975 + 1343
etc.

What identifies a ZX bit is the TOTAL duration of the two half-pulses:
    bit 0 ~= 2 * 855 T-states
    bit 1 ~= 2 * 1710 T-states

The recording is allowed to have a timing scale and quantization/wow.

Input:
    tape2pulses output text

Output:
    standard TAP file

Usage:
    python3 zx_pulses2tap_v2.py DATA3_OUT.pulses DATA3_OUT.tap --debug
"""

import argparse
import re
import struct
from pathlib import Path
import numpy as np


BIT0 = 855.0
BIT1 = 1710.0
PILOT = 2168.0


def read_pulses(filename):
    rx = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*[01]\s*$")
    values = []
    with open(filename, "r", errors="replace") as f:
        for line in f:
            m = rx.match(line)
            if m:
                values.append(float(m.group(1)))
    return np.asarray(values, dtype=np.float64)


def find_pilot_runs(pulses, min_count=100):
    """
    Find the long, regular pilot runs.

    1850..2600 is deliberately used here. 1800 was too broad for this
    particular tape because some data pulses immediately before/after
    blocks are around 1817 T-states.
    """
    mask = (pulses >= 1850) & (pulses <= 2600)

    runs = []
    start = None

    for i, good in enumerate(mask):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start >= min_count:
                runs.append((start, i))
            start = None

    if start is not None and len(mask) - start >= min_count:
        runs.append((start, len(mask)))

    return runs


def learn_timing(pulses, pilot_runs):
    """
    Estimate tape speed from the pilot, then derive the boundary between
    a 0-bit pair and a 1-bit pair.

    We do NOT require individual half-pulses to equal each other.
    """
    pilot_values = np.concatenate([
        pulses[a:b] for a, b in pilot_runs
    ])

    pilot = float(np.median(pilot_values))
    scale = pilot / PILOT

    pair0 = 2.0 * BIT0 * scale
    pair1 = 2.0 * BIT1 * scale

    # Midpoint between ideal 0 and 1 pair durations.
    threshold = (pair0 + pair1) / 2.0

    return pilot, scale, pair0, pair1, threshold


def pulse_pair_bit(a, b, threshold):
    """
    Decode purely from the sum of the two half-pulses.

    This is important: the two halves do NOT have to be the same length.
    """
    total = a + b
    return 1 if total >= threshold else 0


def decode_bytes(pulses, start, count, threshold):
    """
    Decode exactly 'count' bytes starting at pulse index 'start'.
    """
    need = count * 16

    if start < 0 or start + need > len(pulses):
        return None

    bits = []
    i = start

    for _ in range(count * 8):
        bits.append(pulse_pair_bit(pulses[i], pulses[i + 1], threshold))
        i += 2

    out = bytearray()

    for p in range(0, len(bits), 8):
        value = 0
        for bit in bits[p:p + 8]:
            value = (value << 1) | bit
        out.append(value)

    return bytes(out)


def xor_checksum(block):
    x = 0
    for b in block:
        x ^= b
    return x


def valid_header(block):
    if len(block) != 19:
        return False
    if block[0] != 0x00:
        return False
    if block[1] not in (0, 1, 2, 3):
        return False
    return xor_checksum(block[:-1]) == block[-1]


def valid_data(block):
    if len(block) < 3:
        return False
    if block[0] != 0xFF:
        return False
    return xor_checksum(block[:-1]) == block[-1]


def header_info(block):
    types = {
        0: "BASIC",
        1: "NUMBER ARRAY",
        2: "CHAR ARRAY",
        3: "CODE",
    }

    name = block[2:12].decode("ascii", errors="replace").rstrip()
    length = block[12] | (block[13] << 8)

    return types.get(block[1], str(block[1])), name, length


def find_header(pulses, pilot_end, threshold, search=40):
    """
    Search for a valid 19-byte header immediately following a pilot.

    The first one or two pulses can be sync/transition material, so try
    multiple possible pair boundaries.
    """
    for offset in range(search):
        start = pilot_end + offset
        block = decode_bytes(pulses, start, 19, threshold)

        if block is not None and valid_header(block):
            return block, start

    return None


def find_data(pulses, pilot_end, payload_length, threshold, search=40):
    """
    Search for the data block belonging to the preceding header.
    """
    total = payload_length + 2

    for offset in range(search):
        start = pilot_end + offset
        block = decode_bytes(pulses, start, total, threshold)

        if block is not None and valid_data(block):
            return block, start

    return None


def write_tap(filename, blocks):
    with open(filename, "wb") as f:
        for block in blocks:
            if len(block) > 65535:
                raise ValueError(
                    f"TAP block is too large: {len(block)} bytes"
                )
            f.write(struct.pack("<H", len(block)))
            f.write(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pulses")
    ap.add_argument("tap")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--min-pilot", type=int, default=100)
    args = ap.parse_args()

    pulses = read_pulses(args.pulses)

    print(f"Read {len(pulses):,} pulses from {args.pulses}")

    if len(pulses) < 1000:
        raise SystemExit("Too few pulses.")

    pilots = find_pilot_runs(pulses, args.min_pilot)

    if not pilots:
        raise SystemExit("No ZX Spectrum pilot runs found.")

    pilot, scale, pair0, pair1, threshold = learn_timing(
        pulses, pilots
    )

    print(f"Estimated pilot timing: {pilot:.1f} T-states")
    print(f"Timing scale:           {scale:.4f}")
    print(f"Expected bit-0 pair:    {pair0:.1f} T-states")
    print(f"Expected bit-1 pair:    {pair1:.1f} T-states")
    print(f"Pair classification:    {threshold:.1f} T-states")

    print(f"Pilot runs found: {len(pilots)}")

    if args.debug:
        for n, (a, b) in enumerate(pilots, 1):
            print(
                f"  pilot {n}: pulses {a}-{b-1}, "
                f"count={b-a}, "
                f"median={np.median(pulses[a:b]):.1f}"
            )

    blocks = []
    pending_payload = None

    for n, (a, b) in enumerate(pilots, 1):
        print(f"\nSearching after pilot {n}...")

        # If the preceding block was a header, the next pilot should
        # contain its data block. Try data first.
        if pending_payload is not None:
            result = find_data(
                pulses, b, pending_payload, threshold
            )

            if result:
                block, start = result

                print(
                    f"Data: {len(block)} bytes total, "
                    f"{len(block)-2} payload bytes | CHECKSUM OK"
                )

                if args.debug:
                    print(
                        f"  start pulse: {start} "
                        f"(offset +{start-b})"
                    )
                    print(
                        f"  first bytes: "
                        f"{block[:16].hex(' ')}"
                    )
                    print(
                        f"  last bytes: "
                        f"{block[-8:].hex(' ')}"
                    )

                blocks.append(block)
                pending_payload = None
                continue

            print(
                f"Data block not found for expected "
                f"{pending_payload} payload bytes."
            )

        # Otherwise look for a header.
        result = find_header(
            pulses, b, threshold
        )

        if result:
            block, start = result
            typ, name, length = header_info(block)

            print(
                f"Header: {typ} '{name}' "
                f"length={length} | CHECKSUM OK"
            )

            if args.debug:
                print(
                    f"  start pulse: {start} "
                    f"(offset +{start-b})"
                )
                print(f"  bytes: {block.hex(' ')}")

            blocks.append(block)
            pending_payload = length
            continue

        print("No checksum-valid Spectrum header/data block found.")

    if pending_payload is not None:
        print(
            f"\nWarning: final header declared "
            f"{pending_payload} payload bytes, "
            f"but its data block was not recovered."
        )

    if not blocks:
        raise SystemExit("No valid Spectrum blocks recovered.")

    write_tap(args.tap, blocks)

    print(
        f"\nRecovered {len(blocks)} valid Spectrum block(s)."
    )
    print(f"Wrote TAP: {args.tap}")


if __name__ == "__main__":
    main()

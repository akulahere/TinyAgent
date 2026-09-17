"""Generate our own small PNG fixture for Chapter 9, using only the standard library."""

from pathlib import Path
import struct
import zlib


def make_png() -> bytes:
    width, height = 600, 240
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: none
        for x in range(width):
            color = (255, 255, 255)
            if 40 <= x <= 160 and 60 <= y <= 180:
                color = (225, 35, 35)  # red square
            elif (x - 300) ** 2 + (y - 120) ** 2 <= 60 ** 2:
                color = (30, 90, 225)  # blue circle
            elif 50 <= y <= 180 and abs(x - 490) <= (y - 50) * 65 / 130:
                color = (20, 160, 70)  # green triangle
            rows.extend(color)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "examples" / "vision" / "shapes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_png())
    print(path)

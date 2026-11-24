"""Shared test utilities that are not pytest fixtures."""

import base64

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
)
_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def jpeg_size(jpeg_bytes: bytes) -> tuple[int, int]:
    """Return `(width, height)` parsed from a baseline/extended JPEG SOF marker."""
    index = 2
    while index < len(jpeg_bytes):
        if jpeg_bytes[index] != 0xFF:
            msg = "Malformed JPEG segment"
            raise ValueError(msg)
        marker = jpeg_bytes[index + 1]
        if marker in _SOF_MARKERS:
            height = int.from_bytes(jpeg_bytes[index + 5 : index + 7], "big")
            width = int.from_bytes(jpeg_bytes[index + 7 : index + 9], "big")
            return width, height
        index += 2 + int.from_bytes(jpeg_bytes[index + 2 : index + 4], "big")
    msg = "JPEG holds no SOF marker"
    raise ValueError(msg)

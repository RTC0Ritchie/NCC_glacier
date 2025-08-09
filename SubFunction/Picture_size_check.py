import os
import struct
from typing import Tuple, Dict, Optional, List

# --------------------------------------------------
# 1. Fast width/height parsers (read only header bytes)
# --------------------------------------------------
def _jpeg_size(path: str) -> Optional[Tuple[int, int]]:
    """JPEG: extract width/height from SOF0–SOF15 segment (big-endian)."""
    try:
        with open(path, 'rb') as f:
            if f.read(2) != b'\xFF\xD8':
                return None
            while True:
                marker = f.read(2)
                if len(marker) != 2:
                    return None
                length = struct.unpack('>H', f.read(2))[0]
                if 0xC0 <= marker[1] <= 0xCF and marker[0] == 0xFF:
                    buf = f.read(5)
                    if len(buf) < 5:
                        return None
                    h, w = struct.unpack('>HH', buf[1:5])
                    return w, h
                f.seek(length - 2, 1)
    except Exception:
        return None

def _png_size(path: str) -> Optional[Tuple[int, int]]:
    """PNG: IHDR block has width/height at bytes 16–24 (big-endian)."""
    with open(path, 'rb') as f:
        if f.read(8) != b'\x89PNG\r\n\x1A\n':
            return None
        f.seek(16)
        w, h = struct.unpack('>II', f.read(8))
        return w, h

def _gif_size(path: str) -> Optional[Tuple[int, int]]:
    """GIF: logical-screen descriptor holds width/height at bytes 6–10 (little-endian)."""
    with open(path, 'rb') as f:
        sig = f.read(6)
        if sig not in (b'GIF87a', b'GIF89a'):
            return None
        w, h = struct.unpack('<HH', f.read(4))
        return w, h

def _bmp_size(path: str) -> Optional[Tuple[int, int]]:
    """BMP: DIB header has width/height at bytes 18–26 (little-endian)."""
    with open(path, 'rb') as f:
        if f.read(2) != b'BM':
            return None
        f.seek(18)
        w, h = struct.unpack('<II', f.read(8))
        return w, abs(h)

def _webp_size(path: str) -> Optional[Tuple[int, int]]:
    """WebP: simple VP8 key-frame width/height extraction."""
    with open(path, 'rb') as f:
        if f.read(4) != b'RIFF':
            return None
        f.seek(12)
        fourcc = f.read(4)
        if fourcc == b'VP8 ':
            f.seek(20)
            b0, b1, b2 = struct.unpack('<BBB', f.read(3))
            w = b0 | ((b1 & 0x3F) << 8)
            h = (b2 << 8) | ((b1 & 0xC0) << 6)
            return w, h
    return None

# --------------------------------------------------
# 2. Unified dispatcher for size & format
# --------------------------------------------------
_SIGNATURES = {
    b'\xFF\xD8': ('jpg', _jpeg_size),
    b'\x89PNG':  ('png', _png_size),
    b'GIF':      ('gif', _gif_size),
    b'BM':       ('bmp', _bmp_size),
    b'RIFF':     ('webp', _webp_size),
}

def get_image_info(path: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    """Return (format, (width, height)) or None if unsupported."""
    try:
        with open(path, 'rb') as f:
            head = f.read(2)
        for sig, (fmt, parser) in _SIGNATURES.items():
            if head.startswith(sig):
                sz = parser(path)
                return (fmt, sz) if sz else None
    except Exception:
        pass
    return None

# --------------------------------------------------
# 3. Check uniform size & format in folder
# --------------------------------------------------
def check_uniform(folder: str):
    """Print results: uniform size, uniform format."""
    infos: Dict[str, Tuple[str, Tuple[int, int]]] = {}
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        info = get_image_info(full)
        if info is None:
            continue
        infos[name] = info

    if not infos:
        raise ValueError('No image files found!')

    # --- format check ---
    formats = [fmt for fmt, _ in infos.values()]
    first_fmt = formats[0]
    format_ok = all(fmt == first_fmt for fmt in formats)
    if not format_ok:
        for fname, (fmt, _) in infos.items():
            print(f'{fname}: {fmt}')
        raise ValueError('Image formats differ!')
    # print(f'All images share the same format: {first_fmt}')

    # --- size check ---
    sizes = [size for _, size in infos.values()]
    first_size = sizes[0]
    size_ok = all(sz == first_size for sz in sizes)
    if size_ok:
        print(f'All images have the same size: {first_size}')
    else:
        for fname, (_, size) in infos.items():
            print(f'{fname}: {size}')
        raise ValueError('Image sizes differ!')

# --------------------------------------------------
# 4. CLI
# --------------------------------------------------
if __name__ == '__main__':
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else '.'
    check_uniform(target)
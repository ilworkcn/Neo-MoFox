"""base64 编码和解码的辅助函数。"""

from __future__ import annotations

import base64


def base64_encode_bytes(data: bytes, encoding: str = "utf-8") -> str:
    """将字节编码为 base64 文本。"""
    return base64.b64encode(data).decode(encoding)


def base64_decode_to_bytes(data: str | bytes) -> bytes:
    """将 base64 文本解码为字节。"""
    return base64.b64decode(data)


def base64_decode_to_text(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """将 base64 文本解码为字节后，再解码为文本。"""
    return base64.b64decode(data).decode(encoding, errors=errors)

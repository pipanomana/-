import os
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def to_grayscale(bgr):
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def prewitt_gradients(gray):
    kx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]], dtype=np.float32)

    gx = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kx)
    gy = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=ky)
    g = np.abs(gx) + np.abs(gy)
    return gx, gy, g


def normalize_to_u8(mat):
    min_val = float(np.min(mat))
    max_val = float(np.max(mat))
    if max_val - min_val < 1e-8:
        return np.zeros_like(mat, dtype=np.uint8)
    norm = (mat - min_val) / (max_val - min_val)
    return (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def binarize(mat_u8, threshold):
    return np.where(mat_u8 >= threshold, 255, 0).astype(np.uint8)


def main():
    # Явный путь к входному изображению
    input_path = Path(__file__).resolve().parent / "src" / "img0.png"

    # Порог бинаризации
    threshold = 90

    # Папка для результатов
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(input_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {input_path}")

    gray = to_grayscale(img)
    gx, gy, g = prewitt_gradients(gray)

    gx_norm = normalize_to_u8(gx)
    gy_norm = normalize_to_u8(gy)
    g_norm = normalize_to_u8(g)
    g_binary = binarize(g_norm, threshold)

    # Сохранение результатов
    cv2.imwrite(str(out_dir / "source_color.png"), img)
    cv2.imwrite(str(out_dir / "grayscale.bmp"), gray.round().astype(np.uint8))
    cv2.imwrite(str(out_dir / "gx_norm.bmp"), gx_norm)
    cv2.imwrite(str(out_dir / "gy_norm.bmp"), gy_norm)
    cv2.imwrite(str(out_dir / "g_norm.bmp"), g_norm)
    cv2.imwrite(str(out_dir / "g_binary.bmp"), g_binary)


if __name__ == "__main__":
    main()

import math
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_gray_weighted(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def wan_threshold(gray, window_size=3):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    pad = window_size // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")

    shifts = []
    for dy in range(window_size):
        for dx in range(window_size):
            shifts.append(padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]])

    local_min = np.minimum.reduce(shifts)
    local_max = np.maximum.reduce(shifts)
    return 0.5 * (local_min + local_max)


def save_gray_bmp(path, arr):
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    img.save(path, format="BMP")


def save_binary_bmp(path, arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(path, format="BMP")


def main():
    root = Path(__file__).resolve().parent
    src_dir = root / "src"

    images = sorted(src_dir.glob("img*.png"))
    if not images:
        raise FileNotFoundError(f"No input images found in {src_dir}")

    results = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        rgb = np.array(img, dtype=np.float32) / 255.0
        gray = rgb_to_gray_weighted(rgb)

        gray_name = img_path.stem + "_grayscale.bmp"
        gray_path = src_dir / gray_name
        save_gray_bmp(gray_path, gray)

        t = wan_threshold(gray, window_size=3)
        binary = np.where(gray > t, 255, 0)

        bin_name = img_path.stem + "_binary_wan_w3.bmp"
        bin_path = src_dir / bin_name
        save_binary_bmp(bin_path, binary)

        results.append(
            {
                "source": img_path.name,
                "gray": gray_name,
                "binary": bin_name,
                "size": img.size,
            }
        )

    report_path = root / "report_lab2.md"
    report_path.write_text(build_report(results))


def build_report(results):
    lines = []
    lines.append("# Лабораторная работа №2\n")
    lines.append("## Обесцвечивание и бинаризация растровых изображений\n\n")
    lines.append("### Вариант 8: адаптивная бинаризация WAN\n")
    lines.append("### Окно: 3x3\n\n")
    lines.append("### Исходные данные\n")
    lines.append(f"- Количество изображений: {len(results)}\n")
    lines.append("- Метод: WAN (локальный порог по min/max)\n")
    lines.append("- Размер окна: `3x3`\n")
    lines.append("- Формат исходных изображений: PNG (`src/img*.png`)\n")
    lines.append("- Формат полутоновых и бинарных изображений: BMP\n\n")
    lines.append("### Формулы\n\n")
    lines.append("Обесцвечивание (взвешенное усреднение RGB):\n\n")
    lines.append("```text\n")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)\n")
    lines.append("```\n\n")
    lines.append("Адаптивный порог WAN (локально по min/max):\n\n")
    lines.append("```text\n")
    lines.append("min(x, y) = min(I(i, j)), (i, j) in W(x, y)\n")
    lines.append("max(x, y) = max(I(i, j)), (i, j) in W(x, y)\n")
    lines.append("T(x, y)   = (min(x, y) + max(x, y)) / 2\n")
    lines.append("B(x, y)   = 255, если I(x, y) > T(x, y), иначе 0\n")
    lines.append("```\n\n")
    lines.append("### 1. Приведение полноцветного изображения к полутоновому\n\n")

    for idx, item in enumerate(results, 1):
        lines.append(f"#### 1.{idx} Изображение {idx}\n")
        lines.append(f"Источник: `{item['source']}`\n\n")
        lines.append("| Исходное (RGB, PNG) | Полутоновое (BMP) |\n")
        lines.append("|:-------------------:|:-----------------:|\n")
        lines.append(
            f"| ![source](src/{item['source']}) | ![gray](src/{item['gray']}) |\n\n"
        )

    lines.append("### 2. Бинаризация полутонового изображения методом WAN\n\n")

    for idx, item in enumerate(results, 1):
        lines.append(f"#### 2.{idx} Изображение {idx}\n\n")
        lines.append("| Полутоновое | WAN 3x3 |\n")
        lines.append("|:-----------:|:-------:|\n")
        lines.append(
            f"| ![gray](src/{item['gray']}) | ![wan](src/{item['binary']}) |\n\n"
        )

    lines.append("### Результаты выполнения\n\n")
    lines.append("| Изображение | Размер | Бинарный файл |\n")
    lines.append("|:------------|-------:|:--------------|\n")
    for idx, item in enumerate(results, 1):
        w, h = item["size"]
        lines.append(
            f"| №{idx} ({item['source']}) | {w}x{h} | `{item['binary']}` |\n"
        )

    lines.append("\n### Выводы\n\n")
    lines.append("1. Реализовано обесцвечивание RGB-изображений без библиотечных функций grayscale.\n")
    lines.append("2. Для варианта 8 реализована адаптивная бинаризация WAN с окном 3x3 без библиотечных функций бинаризации.\n")
    lines.append("3. В отчете показаны результаты каждой операции (до и после) на нескольких изображениях.\n")

    return "".join(lines)


if __name__ == "__main__":
    main()

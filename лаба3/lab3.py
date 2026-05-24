from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_gray_weighted(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rank_filter(gray, window_size=3, rank=7):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    pad = window_size // 2
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode="edge")
    shifts = []
    for dy in range(window_size):
        for dx in range(window_size):
            shifts.append(padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]])

    stack = np.stack(shifts, axis=0)
    k = max(0, min(rank - 1, stack.shape[0] - 1))
    return np.partition(stack, k, axis=0)[k]


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

        gray_name = img_path.stem + "_gray.bmp"
        gray_path = src_dir / gray_name
        save_gray_bmp(gray_path, gray)

        gray_filtered = rank_filter(gray, window_size=3, rank=7)
        gray_f_name = img_path.stem + "_gray_rank7_w3.bmp"
        gray_f_path = src_dir / gray_f_name
        save_gray_bmp(gray_f_path, gray_filtered)

        diff_abs = np.abs(gray - gray_filtered)
        diff_name = img_path.stem + "_gray_diff_abs.bmp"
        diff_path = src_dir / diff_name
        save_gray_bmp(diff_path, diff_abs)

        diff_x4 = np.clip(diff_abs * 4.0, 0.0, 1.0)
        diff_x4_name = img_path.stem + "_gray_diff_abs_x4.bmp"
        diff_x4_path = src_dir / diff_x4_name
        save_gray_bmp(diff_x4_path, diff_x4)

        mono = np.where(gray >= (128.0 / 255.0), 255, 0).astype(np.uint8)
        mono_name = img_path.stem + "_mono.bmp"
        mono_path = src_dir / mono_name
        save_binary_bmp(mono_path, mono)

        mono_norm = mono.astype(np.float32) / 255.0
        mono_filtered = rank_filter(mono_norm, window_size=3, rank=7)
        mono_filtered_u8 = (mono_filtered >= 0.5).astype(np.uint8) * 255
        mono_f_name = img_path.stem + "_mono_rank7_w3.bmp"
        mono_f_path = src_dir / mono_f_name
        save_binary_bmp(mono_f_path, mono_filtered_u8)

        xor = np.bitwise_xor(mono, mono_filtered_u8)
        xor_name = img_path.stem + "_mono_diff_xor.bmp"
        xor_path = src_dir / xor_name
        save_binary_bmp(xor_path, xor)

        results.append(
            {
                "source": img_path.name,
                "gray": gray_name,
                "gray_filtered": gray_f_name,
                "gray_diff": diff_name,
                "gray_diff_x4": diff_x4_name,
                "mono": mono_name,
                "mono_filtered": mono_f_name,
                "mono_xor": xor_name,
                "size": img.size,
            }
        )

    report_path = root / "report_lab3.md"
    report_path.write_text(build_report(results))


def build_report(results):
    lines = []
    lines.append("# Лабораторная работа №3\n")
    lines.append("## Фильтрация изображений и морфологические операции\n\n")
    lines.append("### Вариант 8: Ранговый фильтр, маска равнина, окно 3x3\n")
    lines.append("### Ранг: 7/9\n\n")
    lines.append("### Исходные данные\n")
    lines.append(f"- Количество изображений: {len(results)}\n")
    lines.append("- Использованы полноцветные изображения, переведенные в полутон и монохром.\n")
    lines.append("- Монохром создается порогом 128.\n")
    lines.append("- Разностные изображения:\n")
    lines.append("  - полутон: модуль разности `|I - F|` (дополнительно показана версия с усилением контраста);\n")
    lines.append("  - монохром: `XOR(I, F)`.\n\n")
    lines.append("### Формулы\n\n")
    lines.append("Перевод в полутон:\n\n")
    lines.append("```text\n")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)\n")
    lines.append("```\n\n")
    lines.append("Ранговый фильтр (окно 3x3, ранг 7/9):\n\n")
    lines.append("```text\n")
    lines.append("F(x, y) = rank_7( W(x, y) )\n")
    lines.append("```\n\n")
    lines.append("Разностные изображения:\n\n")
    lines.append("```text\n")
    lines.append("D_gray(x, y) = |I(x, y) - F(x, y)|\n")
    lines.append("D_mono(x, y) = I_mono(x, y) XOR F_mono(x, y)\n")
    lines.append("```\n\n")
    lines.append("### 1. Полутоновая обработка\n\n")

    for idx, item in enumerate(results, 1):
        lines.append(f"#### 1.{idx} Изображение {idx}\n")
        lines.append(f"Источник: `{item['source']}`\n\n")
        lines.append("| Исходное RGB | Полутоновое | После фильтра | Разностное `|I-F|` | Разностное (x4) |\n")
        lines.append("|:------------:|:-----------:|:-------------:|:------------------:|:---------------:|\n")
        lines.append(
            "| "
            f"![src](src/{item['source']}) | "
            f"![gray](src/{item['gray']}) | "
            f"![grayf](src/{item['gray_filtered']}) | "
            f"![diff](src/{item['gray_diff']}) | "
            f"![diffv](src/{item['gray_diff_x4']}) |\n\n"
        )

    lines.append("### 2. Монохромная обработка\n\n")

    for idx, item in enumerate(results, 1):
        lines.append(f"#### 2.{idx} Изображение {idx}\n\n")
        lines.append("| Исходное монохромное | После фильтра | XOR-разность |\n")
        lines.append("|:--------------------:|:-------------:|:------------:|\n")
        lines.append(
            f"| ![mono](src/{item['mono']}) | ![monof](src/{item['mono_filtered']}) | ![xord](src/{item['mono_xor']}) |\n\n"
        )

    lines.append("### Результаты выполнения\n\n")
    lines.append("| Изображение | Размер | Фильтр |\n")
    lines.append("|:------------|-------:|:-------|\n")
    for idx, item in enumerate(results, 1):
        w, h = item["size"]
        lines.append(
            f"| №{idx} ({item['source']}) | {w}x{h} | Ранговый фильтр, ранг 7/9, окно 3x3 |\n"
        )

    lines.append("\n### Выводы\n\n")
    lines.append("1. Реализован ранговый фильтр (вариант 8) без библиотечных функций фильтрации.\n")
    lines.append("2. Получены отфильтрованные изображения в полутоне и монохроме.\n")
    lines.append("3. Сформированы разностные изображения: модуль разности для полутона и XOR для монохрома.\n")

    return "".join(lines)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_gray_weighted(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def conv2d(gray, kernel):
    kh, kw = kernel.shape
    pad_y = kh // 2
    pad_x = kw // 2
    padded = np.pad(gray, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros_like(gray, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            out += kernel[y, x] * padded[y : y + gray.shape[0], x : x + gray.shape[1]]
    return out


def normalize_to_u8(arr):
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - amin) / (amax - amin)
    return (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def save_gray_bmp(path, arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(path, format="BMP")


def main():
    root = Path(__file__).resolve().parent
    src_dir = root / "src"

    images = sorted(src_dir.glob("img*.png"))
    if not images:
        raise FileNotFoundError(f"No input images found in {src_dir}")

    kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    threshold = 90
    results = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        rgb = np.array(img, dtype=np.float32) / 255.0
        gray = rgb_to_gray_weighted(rgb)
        gray_u8 = (np.clip(gray, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        gray_name = img_path.stem + "_gray.bmp"
        gray_path = src_dir / gray_name
        save_gray_bmp(gray_path, gray_u8)

        gx = conv2d(gray, kx)
        gy = conv2d(gray, ky)
        g = np.abs(gx) + np.abs(gy)

        gx_norm = normalize_to_u8(gx)
        gy_norm = normalize_to_u8(gy)
        g_norm = normalize_to_u8(g)

        gx_name = img_path.stem + "_gx_norm.bmp"
        gy_name = img_path.stem + "_gy_norm.bmp"
        g_name = img_path.stem + "_g_norm.bmp"

        save_gray_bmp(src_dir / gx_name, gx_norm)
        save_gray_bmp(src_dir / gy_name, gy_norm)
        save_gray_bmp(src_dir / g_name, g_norm)

        binary = np.where(g_norm >= threshold, 255, 0).astype(np.uint8)
        binary_name = img_path.stem + "_g_binary.bmp"
        save_gray_bmp(src_dir / binary_name, binary)

        results.append(
            {
                "source": img_path.name,
                "gray": gray_name,
                "gx": gx_name,
                "gy": gy_name,
                "g": g_name,
                "binary": binary_name,
                "size": img.size,
            }
        )

    report_path = root / "report_lab4.md"
    report_path.write_text(build_report(results, threshold))


def build_report(results, threshold):
    lines = []
    lines.append("# Лабораторная работа №4\n")
    lines.append("## Выделение контуров на изображении\n\n")
    lines.append("### Вариант 8\n")
    lines.append("- Оператор: Прюитт 3x3\n")
    lines.append("- Формула градиента: `G = |Gx| + |Gy|`\n")
    lines.append(f"- Порог бинаризации градиентной матрицы: `T={threshold}` (подобран опытным путем)\n\n")
    lines.append("### Формулы\n\n")
    lines.append("Перевод цветного изображения в полутоновое:\n\n")
    lines.append("```text\n")
    lines.append("I(x, y) = 0.299 * R(x, y) + 0.587 * G(x, y) + 0.114 * B(x, y)\n")
    lines.append("```\n\n")
    lines.append("Градиенты по оператору Прюитта (ядра 3x3):\n\n")
    lines.append("```text\n")
    lines.append("Kx = [[ 1,  0, -1],\n")
    lines.append("      [ 1,  0, -1],\n")
    lines.append("      [ 1,  0, -1]]\n\n")
    lines.append("Ky = [[ 1,  1,  1],\n")
    lines.append("      [ 0,  0,  0],\n")
    lines.append("      [-1, -1, -1]]\n")
    lines.append("```\n\n")
    lines.append("```text\n")
    lines.append("Gx = I * Kx\n")
    lines.append("Gy = I * Ky\n")
    lines.append("G  = |Gx| + |Gy|\n")
    lines.append("```\n\n")
    lines.append("Бинаризация градиентной матрицы:\n\n")
    lines.append("```text\n")
    lines.append("B(x, y) = 255, если G(x, y) >= T, иначе 0\n")
    lines.append("```\n\n")

    lines.append("### Результаты\n\n")

    for idx, item in enumerate(results, 1):
        w, h = item["size"]
        lines.append(f"#### {idx}. Изображение {idx} (размер {w}x{h})\n\n")
        lines.append("**1. Исходное цветное изображение**\n\n")
        lines.append(f"![source](src/{item['source']})\n\n")
        lines.append("**2. Полутоновое изображение**\n\n")
        lines.append(f"![gray](src/{item['gray']})\n\n")
        lines.append("**3. Градиентные матрицы (нормализованные 0..255)**\n\n")
        lines.append("| Gx | Gy | G |\n")
        lines.append("|:--:|:--:|:--:|\n")
        lines.append(
            f"| ![gx](src/{item['gx']}) | ![gy](src/{item['gy']}) | ![g](src/{item['g']}) |\n\n"
        )
        lines.append("**4. Бинаризованная градиентная матрица G**\n\n")
        lines.append(f"![binary](src/{item['binary']})\n\n")

    lines.append("### Таблица файлов\n\n")
    lines.append("| Операция | Файл |\n")
    lines.append("|:---------|:-----|\n")
    lines.append("| Исходное цветное | `src/img*_source.png` |\n")
    lines.append("| Полутоновое | `src/img*_gray.bmp` |\n")
    lines.append("| Нормализованная матрица Gx | `src/img*_gx_norm.bmp` |\n")
    lines.append("| Нормализованная матрица Gy | `src/img*_gy_norm.bmp` |\n")
    lines.append("| Нормализованная матрица G | `src/img*_g_norm.bmp` |\n")
    lines.append("| Бинаризация G | `src/img*_g_binary.bmp` |\n\n")

    lines.append("### Вывод\n")
    lines.append(
        "Для варианта 8 реализовано выделение контуров оператором Прюитта 3x3 с формулой `G = |Gx| + |Gy|`. "
        "Получены требуемые матрицы `Gx`, `Gy`, `G` и итоговая бинаризованная карта контуров.\n"
    )

    return "".join(lines)


if __name__ == "__main__":
    main()

import math
from pathlib import Path

import numpy as np
from PIL import Image


def rgb_to_hsi(rgb):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    i = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    s = np.zeros_like(i)
    mask_i = i > 1e-8
    s[mask_i] = 1.0 - (min_rgb[mask_i] / i[mask_i])

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    h = np.zeros_like(i)
    mask_den = den > 1e-8
    theta = np.zeros_like(i)
    theta[mask_den] = np.arccos(np.clip(num[mask_den] / den[mask_den], -1.0, 1.0))
    h[mask_den] = np.where(b[mask_den] <= g[mask_den], theta[mask_den], 2.0 * np.pi - theta[mask_den])

    return h, s, i


def hsi_to_rgb(h, s, i):
    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    mask_zero = s <= 1e-8
    r[mask_zero] = i[mask_zero]
    g[mask_zero] = i[mask_zero]
    b[mask_zero] = i[mask_zero]

    mask = ~mask_zero
    if not np.any(mask):
        return np.stack([r, g, b], axis=-1)

    h_m = h[mask]
    s_m = s[mask]
    i_m = i[mask]

    r_m = np.empty_like(h_m)
    g_m = np.empty_like(h_m)
    b_m = np.empty_like(h_m)

    region1 = h_m < 2.0 * np.pi / 3.0
    region2 = (h_m >= 2.0 * np.pi / 3.0) & (h_m < 4.0 * np.pi / 3.0)
    region3 = h_m >= 4.0 * np.pi / 3.0

    if np.any(region1):
        h1 = h_m[region1]
        s1 = s_m[region1]
        i1 = i_m[region1]
        b1 = i1 * (1.0 - s1)
        r1 = i1 * (1.0 + (s1 * np.cos(h1) / np.cos(np.pi / 3.0 - h1)))
        g1 = 3.0 * i1 - (r1 + b1)
        r_m[region1] = r1
        g_m[region1] = g1
        b_m[region1] = b1

    if np.any(region2):
        h2 = h_m[region2] - 2.0 * np.pi / 3.0
        s2 = s_m[region2]
        i2 = i_m[region2]
        r2 = i2 * (1.0 - s2)
        g2 = i2 * (1.0 + (s2 * np.cos(h2) / np.cos(np.pi / 3.0 - h2)))
        b2 = 3.0 * i2 - (r2 + g2)
        r_m[region2] = r2
        g_m[region2] = g2
        b_m[region2] = b2

    if np.any(region3):
        h3 = h_m[region3] - 4.0 * np.pi / 3.0
        s3 = s_m[region3]
        i3 = i_m[region3]
        g3 = i3 * (1.0 - s3)
        b3 = i3 * (1.0 + (s3 * np.cos(h3) / np.cos(np.pi / 3.0 - h3)))
        r3 = 3.0 * i3 - (g3 + b3)
        r_m[region3] = r3
        g_m[region3] = g3
        b_m[region3] = b3

    r[mask] = r_m
    g[mask] = g_m
    b[mask] = b_m

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def bilinear_resize(src, out_h, out_w):
    h, w = src.shape[:2]
    if out_h < 1 or out_w < 1:
        raise ValueError("Invalid output size")

    if out_h == 1:
        ys = np.array([0.0], dtype=np.float32)
    else:
        ys = np.linspace(0, h - 1, out_h, dtype=np.float32)

    if out_w == 1:
        xs = np.array([0.0], dtype=np.float32)
    else:
        xs = np.linspace(0, w - 1, out_w, dtype=np.float32)

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    dy = (ys - y0)[:, None, None]
    dx = (xs - x0)[None, :, None]

    ia = src[y0[:, None], x0[None, :]]
    ib = src[y0[:, None], x1[None, :]]
    ic = src[y1[:, None], x0[None, :]]
    id_ = src[y1[:, None], x1[None, :]]

    top = ia * (1.0 - dx) + ib * dx
    bottom = ic * (1.0 - dx) + id_ * dx
    out = top * (1.0 - dy) + bottom * dy
    return out


def decimate(src, factor):
    return src[::factor, ::factor]


def save_rgb(path, arr):
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8))
    img.save(path)


def save_gray(path, arr):
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
    img.save(path)


def main():
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    src_dir.mkdir(exist_ok=True)

    input_path = root / "картинка кота.jpg"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input image: {input_path}")

    img = Image.open(input_path).convert("RGB")
    source_path = src_dir / "source.png"
    img.save(source_path)

    rgb = np.array(img, dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]

    save_gray(src_dir / "r_channel.png", rgb[..., 0])
    save_gray(src_dir / "g_channel.png", rgb[..., 1])
    save_gray(src_dir / "b_channel.png", rgb[..., 2])

    hsi_h, hsi_s, hsi_i = rgb_to_hsi(rgb)
    save_gray(src_dir / "intensity_channel.png", hsi_i)

    inv_i = 1.0 - hsi_i
    inv_rgb = hsi_to_rgb(hsi_h, hsi_s, inv_i)
    save_rgb(src_dir / "inverted_intensity.png", inv_rgb)

    m = 3
    n = 2
    k = m / n

    upscaled = bilinear_resize(rgb, h * m, w * m)
    save_rgb(src_dir / "upscaled.png", upscaled)

    downscaled = decimate(rgb, n)
    save_rgb(src_dir / "downscaled.png", downscaled)

    two_pass = decimate(upscaled, n)
    save_rgb(src_dir / "two_pass.png", two_pass)

    out_h = int(math.ceil(h * k))
    out_w = int(math.ceil(w * k))
    one_pass = bilinear_resize(rgb, out_h, out_w)
    save_rgb(src_dir / "one_pass.png", one_pass)

    sizes = {
        "source": (w, h),
        "upscaled": (upscaled.shape[1], upscaled.shape[0]),
        "downscaled": (downscaled.shape[1], downscaled.shape[0]),
        "two_pass": (two_pass.shape[1], two_pass.shape[0]),
        "one_pass": (one_pass.shape[1], one_pass.shape[0]),
    }

    report_path = root / "report.md"
    report_path.write_text(
        "# Лабораторная работа №1\n"
        "## Цветовые модели и передискретизация изображений\n\n"
        "### Исходное изображение\n\n"
        "![Исходное изображение](src/source.png)\n\n"
        "### 1. Цветовые модели\n\n"
        "#### 1.1 Компоненты R, G, B\n\n"
        "|             Красный канал              |             Зеленый канал              |              Синий канал               |\n"
        "|:--------------------------------------:|:--------------------------------------:|:--------------------------------------:|\n"
        "| ![R](src/r_channel.png) | ![G](src/g_channel.png) | ![B](src/b_channel.png) |\n\n"
        "#### 1.2 Яркостная компонента HSI\n\n"
        "![Яркостная компонента HSI](src/intensity_channel.png)\n\n"
        "#### 1.3 Инвертирование яркостной компоненты\n\n"
        "|            Исходное изображение            |                  С инвертированной яркостью                   |\n"
        "|:------------------------------------------:|:-------------------------------------------------------------:|\n"
        "| ![Исходное](src/source.png) | ![Инвертированное](src/inverted_intensity.png) |\n\n"
        "### 2. Передискретизация (M=3, N=2, K=1.500)\n\n"
        "#### 2.1 Растяжение в M раз (метод билинейной интерполяции)\n\n"
        "|                  Исходное                  |                   Растянутое                   |\n"
        "|:------------------------------------------:|:----------------------------------------------:|\n"
        "| ![Исходное](src/source.png) | ![Растянутое](src/upscaled.png) |\n\n"
        "#### 2.2 Сжатие в N раз (метод прореживания)\n\n"
        "|                  Исходное                  |                    Сжатое                    |\n"
        "|:------------------------------------------:|:--------------------------------------------:|\n"
        "| ![Исходное](src/source.png) | ![Сжатое](src/downscaled.png) |\n\n"
        "#### 2.3 Двухпроходная передискретизация (растяжение + сжатие)\n\n"
        "|                  Исходное                  |             Результат двух проходов             |\n"
        "|:------------------------------------------:|:-----------------------------------------------:|\n"
        "| ![Исходное](src/source.png) | ![Два прохода](src/two_pass.png) |\n\n"
        "#### 2.4 Однопроходная передискретизация (прямое масштабирование)\n\n"
        "|                  Исходное                  |            Результат одного прохода             |\n"
        "|:------------------------------------------:|:-----------------------------------------------:|\n"
        "| ![Исходное](src/source.png) | ![Один проход](src/one_pass.png) |\n\n"
        "### Результаты выполнения\n\n"
        "| Операция                          | Размер изображения |\n"
        "|:----------------------------------|-------------------:|\n"
        f"| Исходное изображение              | {sizes['source'][0]}x{sizes['source'][1]} |\n"
        f"| Растяжение (M=3)                | {sizes['upscaled'][0]}x{sizes['upscaled'][1]} |\n"
        f"| Сжатие (N=2)                    | {sizes['downscaled'][0]}x{sizes['downscaled'][1]} |\n"
        f"| Двухпроходная (M=3 + N=2)     | {sizes['two_pass'][0]}x{sizes['two_pass'][1]} |\n"
        f"| Однопроходная (K=1.500)         | {sizes['one_pass'][0]}x{sizes['one_pass'][1]} |\n\n"
        "### Выводы\n\n"
        "В ходе выполнения лабораторной работы были изучены:\n\n"
        "1. **Цветовые модели RGB и HSI**:\n"
        "   - Выделены компоненты красного, зеленого и синего каналов.\n"
        "   - Выполнено преобразование RGB -> HSI.\n"
        "   - Произведено инвертирование яркостной компоненты.\n\n"
        "2. **Методы передискретизации**:\n"
        "   - Реализован метод билинейной интерполяции для растяжения изображения.\n"
        "   - Реализован метод прореживания для сжатия изображения.\n"
        "   - Выполнена двухпроходная передискретизация: растяжение в M раз с последующим сжатием в N раз.\n"
        "   - Реализована однопроходная передискретизация: прямое масштабирование в K=M/N раз.\n"
    )


if __name__ == "__main__":
    main()

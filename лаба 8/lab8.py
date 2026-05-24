from pathlib import Path
import csv
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
RESULTS_DIR = BASE_DIR / "results"

CASES = [
    ("case1", Path("../лаба 4/src/img0.png")),
    ("case2", Path("../лаба 3/src/img1.png")),
    ("case3", Path("../лаба2/src/img2.png")),
]

LEVELS = 16
DIRECTIONS = [
    (1, 0, "0"),
    (0, 1, "90"),
    (1, 1, "45"),
    (1, -1, "135"),
]


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for case, _ in CASES:
        (SRC_DIR / case).mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    return Image.open(path).convert("RGB")


def rgb_to_hsl(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    delta = maxc - minc
    lightness = (maxc + minc) / 2.0

    saturation = np.zeros_like(lightness)
    nonzero = delta > 1e-8
    saturation[nonzero] = delta[nonzero] / (1.0 - np.abs(2.0 * lightness[nonzero] - 1.0))

    hue = np.zeros_like(lightness)
    red_max = nonzero & (maxc == r)
    green_max = nonzero & (maxc == g)
    blue_max = nonzero & (maxc == b)
    hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6.0
    hue[green_max] = ((b[green_max] - r[green_max]) / delta[green_max]) + 2.0
    hue[blue_max] = ((r[blue_max] - g[blue_max]) / delta[blue_max]) + 4.0
    hue /= 6.0

    return hue, saturation, lightness


def hsl_to_rgb(hue, saturation, lightness):
    c = (1.0 - np.abs(2.0 * lightness - 1.0)) * saturation
    h6 = (hue * 6.0) % 6.0
    x = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = lightness - c / 2.0

    rp = np.zeros_like(lightness)
    gp = np.zeros_like(lightness)
    bp = np.zeros_like(lightness)

    masks = [
        (0 <= h6) & (h6 < 1),
        (1 <= h6) & (h6 < 2),
        (2 <= h6) & (h6 < 3),
        (3 <= h6) & (h6 < 4),
        (4 <= h6) & (h6 < 5),
        (5 <= h6) & (h6 < 6),
    ]
    values = [
        (c, x, 0),
        (x, c, 0),
        (0, c, x),
        (0, x, c),
        (x, 0, c),
        (c, 0, x),
    ]

    for mask, (rv, gv, bv) in zip(masks, values):
        rp[mask] = rv[mask] if hasattr(rv, "__getitem__") else rv
        gp[mask] = gv[mask] if hasattr(gv, "__getitem__") else gv
        bp[mask] = bv[mask] if hasattr(bv, "__getitem__") else bv

    rgb = np.stack([rp + m, gp + m, bp + m], axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def linear_contrast_lightness(lightness):
    l_min = float(lightness.min())
    l_max = float(lightness.max())
    if l_max - l_min < 1e-8:
        return np.zeros_like(lightness), l_min, l_max, 0.0, 0.0

    a = 1.0 / (l_max - l_min)
    b = -l_min / (l_max - l_min)
    out = np.clip(a * lightness + b, 0.0, 1.0)
    return out, l_min, l_max, a, b


def save_gray(lightness, path):
    gray = (np.clip(lightness, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(gray, "L").save(path)
    return gray


def quantize(gray_u8, levels=LEVELS):
    return np.minimum((gray_u8.astype(np.int32) * levels) // 256, levels - 1).astype(np.uint8)


def iter_lines(height, width, dx, dy):
    starts = []
    if dx == 1 and dy == 0:
        starts = [(0, y) for y in range(height)]
    elif dx == 0 and dy == 1:
        starts = [(x, 0) for x in range(width)]
    elif dx == 1 and dy == 1:
        starts = [(x, 0) for x in range(width)] + [(0, y) for y in range(1, height)]
    elif dx == 1 and dy == -1:
        starts = [(x, height - 1) for x in range(width)] + [(0, y) for y in range(height - 1)]
    else:
        raise ValueError("Unsupported direction")

    for x, y in starts:
        coords = []
        while 0 <= x < width and 0 <= y < height:
            coords.append((x, y))
            x += dx
            y += dy
        yield coords


def glrlm_matrix(quantized, levels=LEVELS):
    height, width = quantized.shape
    max_run = max(height, width)
    matrix = np.zeros((levels, max_run), dtype=np.float64)

    for dx, dy, _ in DIRECTIONS:
        for coords in iter_lines(height, width, dx, dy):
            if not coords:
                continue
            x0, y0 = coords[0]
            current = int(quantized[y0, x0])
            run_len = 1
            for x, y in coords[1:]:
                value = int(quantized[y, x])
                if value == current:
                    run_len += 1
                else:
                    matrix[current, run_len - 1] += 1.0
                    current = value
                    run_len = 1
            matrix[current, run_len - 1] += 1.0

    nonzero_cols = np.where(matrix.sum(axis=0) > 0)[0]
    if nonzero_cols.size:
        matrix = matrix[:, : int(nonzero_cols.max()) + 1]
    return matrix


def glrlm_features(matrix):
    k = float(matrix.sum())
    if k == 0:
        return {"K": 0.0, "GLNU": 0.0, "RLNU": 0.0, "GLNU_norm": 0.0, "RLNU_norm": 0.0}

    grey_profile = matrix.sum(axis=1)
    run_profile = matrix.sum(axis=0)
    glnu = float(np.sum(grey_profile ** 2) / k)
    rlnu = float(np.sum(run_profile ** 2) / k)
    return {
        "K": k,
        "GLNU": glnu,
        "RLNU": rlnu,
        "GLNU_norm": glnu / k,
        "RLNU_norm": rlnu / k,
    }


def save_matrix_csv(matrix, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["grey_level/run_length", *range(1, matrix.shape[1] + 1)])
        for level, row in enumerate(matrix):
            writer.writerow([level, *(int(value) for value in row)])


def save_features_csv(features_before, features_after, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["feature", "before", "after", "abs_diff", "rel_diff"])
        for key in ["K", "GLNU", "RLNU", "GLNU_norm", "RLNU_norm"]:
            before = features_before[key]
            after = features_after[key]
            abs_diff = abs(after - before)
            rel_diff = abs_diff / abs(before) if abs(before) > 1e-12 else 0.0
            writer.writerow([key, f"{before:.8f}", f"{after:.8f}", f"{abs_diff:.8f}", f"{rel_diff:.8f}"])


def save_matrix_image(matrix, path):
    log_matrix = np.log1p(matrix)
    max_value = float(log_matrix.max())
    if max_value < 1e-8:
        img = np.zeros_like(log_matrix, dtype=np.uint8)
    else:
        img = (log_matrix / max_value * 255.0).round().astype(np.uint8)

    image = Image.fromarray(img, "L")
    scale_x = max(1, min(8, 900 // max(image.width, 1)))
    scale_y = max(8, 240 // max(image.height, 1))
    image = image.resize((image.width * scale_x, image.height * scale_y), Image.Resampling.NEAREST)
    image.save(path)


def histogram(lightness):
    values = (np.clip(lightness, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.bincount(values.ravel(), minlength=256)


def draw_hist_compare(hist_before, hist_after, path):
    width = 620
    height = 320
    margin_left = 48
    margin_bottom = 34
    margin_top = 28
    plot_w = 540
    plot_h = 230
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    max_value = max(int(hist_before.max()), int(hist_after.max()), 1)
    origin_x = margin_left
    origin_y = margin_top + plot_h
    draw.line((origin_x, margin_top, origin_x, origin_y), fill="black")
    draw.line((origin_x, origin_y, origin_x + plot_w, origin_y), fill="black")
    draw.text((margin_left, 6), "Brightness histograms: before / after", fill="black", font=font)

    for i in range(256):
        x = origin_x + int(i / 255 * (plot_w - 1))
        y_before = origin_y - int(hist_before[i] / max_value * plot_h)
        y_after = origin_y - int(hist_after[i] / max_value * plot_h)
        draw.line((x, origin_y, x, y_before), fill=(40, 90, 170))
        draw.line((x, origin_y, x, y_after), fill=(190, 65, 45))

    draw.text((origin_x, origin_y + 12), "0", fill="black", font=font)
    draw.text((origin_x + plot_w - 18, origin_y + 12), "255", fill="black", font=font)
    draw.rectangle((420, 12, 432, 24), fill=(40, 90, 170))
    draw.text((438, 13), "before", fill="black", font=font)
    draw.rectangle((500, 12, 512, 24), fill=(190, 65, 45))
    draw.text((518, 13), "after", fill="black", font=font)
    image.save(path)


def process_case(case_name, source_rel_path):
    case_dir = SRC_DIR / case_name
    source_path = (BASE_DIR / source_rel_path).resolve()
    image = load_rgb(source_path)
    rgb = np.array(image)
    hue, saturation, lightness_before = rgb_to_hsl(rgb)
    lightness_after, l_min, l_max, a, b = linear_contrast_lightness(lightness_before)
    rgb_after = hsl_to_rgb(hue, saturation, lightness_after)

    shutil.copyfile(source_path, case_dir / "source_color.png")
    Image.fromarray(rgb_after, "RGB").save(case_dir / "contrast_color.png")
    gray_before = save_gray(lightness_before, case_dir / "gray_before.bmp")
    gray_after = save_gray(lightness_after, case_dir / "gray_after.bmp")

    draw_hist_compare(histogram(lightness_before), histogram(lightness_after), case_dir / "hist_compare.png")

    matrix_before = glrlm_matrix(quantize(gray_before))
    matrix_after = glrlm_matrix(quantize(gray_after))
    features_before = glrlm_features(matrix_before)
    features_after = glrlm_features(matrix_after)

    save_matrix_csv(matrix_before, case_dir / "glrlm_before.csv")
    save_matrix_csv(matrix_after, case_dir / "glrlm_after.csv")
    save_features_csv(features_before, features_after, case_dir / "features_compare.csv")
    save_matrix_image(matrix_before, case_dir / "glrlm_before.png")
    save_matrix_image(matrix_after, case_dir / "glrlm_after.png")

    return {
        "case": case_name,
        "source": source_rel_path.as_posix(),
        "width": image.width,
        "height": image.height,
        "Lmin": l_min,
        "Lmax": l_max,
        "a": a,
        "b": b,
        "matrix_before_shape": f"{matrix_before.shape[0]}x{matrix_before.shape[1]}",
        "matrix_after_shape": f"{matrix_after.shape[0]}x{matrix_after.shape[1]}",
        "features_before": features_before,
        "features_after": features_after,
    }


def write_summary(rows):
    csv_path = SRC_DIR / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "case", "source", "size", "matrix_before", "matrix_after",
            "GLNU_before", "GLNU_after", "RLNU_before", "RLNU_after",
            "GLNU_norm_before", "GLNU_norm_after", "RLNU_norm_before", "RLNU_norm_after",
        ])
        for row in rows:
            before = row["features_before"]
            after = row["features_after"]
            writer.writerow([
                row["case"],
                row["source"],
                f"{row['width']}x{row['height']}",
                row["matrix_before_shape"],
                row["matrix_after_shape"],
                f"{before['GLNU']:.6f}",
                f"{after['GLNU']:.6f}",
                f"{before['RLNU']:.6f}",
                f"{after['RLNU']:.6f}",
                f"{before['GLNU_norm']:.8f}",
                f"{after['GLNU_norm']:.8f}",
                f"{before['RLNU_norm']:.8f}",
                f"{after['RLNU_norm']:.8f}",
            ])

    json_path = SRC_DIR / "summary.json"
    serializable = []
    for row in rows:
        copy = dict(row)
        copy["features_before"] = {k: float(v) for k, v in row["features_before"].items()}
        copy["features_after"] = {k: float(v) for k, v in row["features_after"].items()}
        serializable.append(copy)
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(rows):
    lines = [
        "# Лабораторная работа №8",
        "## Текстурный анализ и контрастирование",
        "",
        "### Вариант 8: GLRLM",
        "- Матрица: GLRLM, матрица длин серий.",
        "- Признаки: `GLNU`, `RLNU`.",
        "- Преобразование яркости: линейное преобразование яркости.",
        f"- Количество уровней квантования яркости для GLRLM: `{LEVELS}`.",
        "- Направления серий: `0`, `45`, `90`, `135` градусов.",
        "",
        "### Формулы",
        "",
        "Линейное преобразование яркости в канале `L` модели HSL:",
        "",
        "```text",
        "L'(x, y) = a * L(x, y) + b",
        "a = 1 / (Lmax - Lmin)",
        "b = -Lmin / (Lmax - Lmin)",
        "```",
        "",
        "Матрица длин серий GLRLM:",
        "",
        "```text",
        "n(a, r) = количество серий длины r с уровнем яркости a",
        "K = sum_a sum_r n(a, r)",
        "N_a = sum_r n(a, r)",
        "N_r = sum_a n(a, r)",
        "GLNU = (1 / K) * sum_a N_a^2",
        "RLNU = (1 / K) * sum_r N_r^2",
        "```",
        "",
        "### Исходные данные",
        "",
        "| Случай | Источник | Размер |",
        "|:------:|:---------|:------:|",
    ]

    for row in rows:
        lines.append(f"| {row['case']} | `{row['source']}` | `{row['width']}x{row['height']}` |")

    lines.append("")
    lines.append("### Результаты по изображениям")
    lines.append("")

    for row in rows:
        case = row["case"]
        before = row["features_before"]
        after = row["features_after"]
        lines.extend([
            f"#### {case}",
            f"- Источник: `{row['source']}`",
            f"- Параметры линейного преобразования: `Lmin={row['Lmin']:.6f}`, `Lmax={row['Lmax']:.6f}`, `a={row['a']:.6f}`, `b={row['b']:.6f}`",
            "",
            "| Исходное RGB | Полутоновое L до | Полутоновое L после |",
            "|:------------:|:----------------:|:-------------------:|",
            f"| ![{case}_src](src/{case}/source_color.png) | ![{case}_gray_before](src/{case}/gray_before.bmp) | ![{case}_gray_after](src/{case}/gray_after.bmp) |",
            "",
            "| Контрастированное RGB | Гистограммы яркости до/после |",
            "|:---------------------:|:----------------------------:|",
            f"| ![{case}_contrast](src/{case}/contrast_color.png) | ![{case}_hist](src/{case}/hist_compare.png) |",
            "",
            "| GLRLM до | GLRLM после |",
            "|:--------:|:-----------:|",
            f"| ![{case}_glrlm_before](src/{case}/glrlm_before.png) | ![{case}_glrlm_after](src/{case}/glrlm_after.png) |",
            "",
            "| Метрика | До | После |",
            "|:--------|---:|------:|",
            f"| K, число серий | `{before['K']:.0f}` | `{after['K']:.0f}` |",
            f"| GLNU | `{before['GLNU']:.6f}` | `{after['GLNU']:.6f}` |",
            f"| RLNU | `{before['RLNU']:.6f}` | `{after['RLNU']:.6f}` |",
            f"| GLNU_norm | `{before['GLNU_norm']:.8f}` | `{after['GLNU_norm']:.8f}` |",
            f"| RLNU_norm | `{before['RLNU_norm']:.8f}` | `{after['RLNU_norm']:.8f}` |",
            "",
            f"CSV: `src/{case}/glrlm_before.csv`, `src/{case}/glrlm_after.csv`, `src/{case}/features_compare.csv`",
            "",
        ])

    lines.extend([
        "### Сводные результаты",
        "",
        "| Случай | Размер | GLRLM до | GLRLM после | GLNU до | GLNU после | RLNU до | RLNU после |",
        "|:------:|:------:|:--------:|:-----------:|--------:|-----------:|--------:|-----------:|",
    ])

    for row in rows:
        before = row["features_before"]
        after = row["features_after"]
        lines.append(
            f"| {row['case']} | `{row['width']}x{row['height']}` | `{row['matrix_before_shape']}` | `{row['matrix_after_shape']}` | "
            f"{before['GLNU']:.6f} | {after['GLNU']:.6f} | {before['RLNU']:.6f} | {after['RLNU']:.6f} |"
        )

    lines.extend([
        "",
        "Дополнительно: `src/summary.csv`, `src/summary.json`.",
        "",
        "### Вывод",
        "Для варианта 8 построены матрицы GLRLM для исходных и линейно контрастированных изображений. Рассчитаны признаки неоднородности яркости серий `GLNU` и неоднородности длин серий `RLNU`; сравнение до/после показывает изменение текстурных признаков после преобразования яркости.",
        "",
    ])

    (BASE_DIR / "report_lab8.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    rows = [process_case(case, path) for case, path in CASES]
    write_summary(rows)
    write_report(rows)


if __name__ == "__main__":
    main()

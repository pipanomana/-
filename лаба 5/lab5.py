from pathlib import Path
import csv
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "src" / "templates"
PROFILES_DIR = BASE_DIR / "src" / "profiles"
RESULTS_DIR = BASE_DIR / "results"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/NotoSansOsmanya-Regular.ttf")
FONT_SIZE = 96
PADDING = 24
THRESHOLD = 128

# Variant 8: Osmanya alphabet letters. U+104A0..U+104A9 are digits, not letters.
ALPHABET = [chr(codepoint) for codepoint in range(0x10480, 0x1049E)]


def ensure_dirs():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def symbol_id(index, symbol):
    return f"sym_{index:02d}_U{ord(symbol):04X}"


def render_symbol(symbol, font):
    canvas_size = FONT_SIZE * 3
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    x = (canvas_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)

    arr = np.array(image)
    rows, cols = np.where(arr < THRESHOLD)
    if rows.size == 0:
        raise ValueError(f"Символ {symbol!r} не отрисован выбранным шрифтом")

    left = max(int(cols.min()) - 1, 0)
    upper = max(int(rows.min()) - 1, 0)
    right = min(int(cols.max()) + 2, image.width)
    lower = min(int(rows.max()) + 2, image.height)
    return image.crop((left, upper, right, lower))


def binary_black(image):
    return (np.array(image.convert("L")) < THRESHOLD).astype(np.uint8)


def quarter_features(binary):
    height, width = binary.shape
    x_mid = width // 2
    y_mid = height // 2
    quarters = [
        binary[:y_mid, :x_mid],
        binary[:y_mid, x_mid:],
        binary[y_mid:, :x_mid],
        binary[y_mid:, x_mid:],
    ]

    masses = [int(q.sum()) for q in quarters]
    densities = [float(m / q.size) if q.size else 0.0 for m, q in zip(masses, quarters)]
    return masses, densities


def scalar_features(binary):
    height, width = binary.shape
    mass = int(binary.sum())
    if mass == 0:
        raise ValueError("Нельзя рассчитать признаки для пустого символа")

    y_coords, x_coords = np.indices(binary.shape)
    xc = float((x_coords * binary).sum() / mass)
    yc = float((y_coords * binary).sum() / mass)
    xc_norm = float(xc / (width - 1)) if width > 1 else 0.0
    yc_norm = float(yc / (height - 1)) if height > 1 else 0.0

    ix = float((((y_coords - yc) ** 2) * binary).sum())
    iy = float((((x_coords - xc) ** 2) * binary).sum())
    ix_norm = float(ix / (mass * height * height))
    iy_norm = float(iy / (mass * width * width))

    q_masses, q_densities = quarter_features(binary)
    profile_x = binary.sum(axis=0).astype(int)
    profile_y = binary.sum(axis=1).astype(int)

    return {
        "width": width,
        "height": height,
        "mass": mass,
        "q_masses": q_masses,
        "q_densities": q_densities,
        "xc": xc,
        "yc": yc,
        "xc_norm": xc_norm,
        "yc_norm": yc_norm,
        "ix": ix,
        "iy": iy,
        "ix_norm": ix_norm,
        "iy_norm": iy_norm,
        "profile_x": profile_x,
        "profile_y": profile_y,
    }


def nice_ticks(max_value):
    if max_value <= 0:
        return [0]
    count = min(max_value, 5)
    return sorted(set(int(round(max_value * i / count)) for i in range(count + 1)))


def draw_profile(profile, path, title, orientation):
    profile = np.asarray(profile, dtype=int)
    max_value = int(profile.max()) if profile.size else 0
    tick_values = nice_ticks(max_value)
    font = ImageFont.load_default()

    margin_left = 48
    margin_right = 18
    margin_top = 28
    margin_bottom = 38

    if orientation == "x":
        bar_w = 6
        gap = 2
        plot_w = max(280, int(profile.size) * (bar_w + gap))
        plot_h = 190
        image = Image.new("RGB", (margin_left + plot_w + margin_right, margin_top + plot_h + margin_bottom), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 6), title, fill="black", font=font)
        origin_x = margin_left
        origin_y = margin_top + plot_h
        draw.line((origin_x, margin_top, origin_x, origin_y), fill="black")
        draw.line((origin_x, origin_y, origin_x + plot_w, origin_y), fill="black")

        for value in tick_values:
            y = origin_y - int(value / max(max_value, 1) * plot_h)
            draw.line((origin_x - 4, y, origin_x, y), fill="black")
            draw.text((4, y - 5), str(value), fill="black", font=font)

        for i, value in enumerate(profile):
            x0 = origin_x + i * (bar_w + gap)
            x1 = x0 + bar_w
            y0 = origin_y - int(value / max(max_value, 1) * plot_h)
            draw.rectangle((x0, y0, x1, origin_y), fill=(40, 85, 160))

        draw.text((origin_x, origin_y + 12), "x", fill="black", font=font)
        draw.text((origin_x + plot_w - 18, origin_y + 12), str(profile.size - 1), fill="black", font=font)
    else:
        bar_h = 6
        gap = 2
        plot_w = 240
        plot_h = max(220, int(profile.size) * (bar_h + gap))
        image = Image.new("RGB", (margin_left + plot_w + margin_right, margin_top + plot_h + margin_bottom), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 6), title, fill="black", font=font)
        origin_x = margin_left
        origin_y = margin_top
        draw.line((origin_x, origin_y, origin_x, origin_y + plot_h), fill="black")
        draw.line((origin_x, origin_y + plot_h, origin_x + plot_w, origin_y + plot_h), fill="black")

        for value in tick_values:
            x = origin_x + int(value / max(max_value, 1) * plot_w)
            draw.line((x, origin_y + plot_h, x, origin_y + plot_h + 4), fill="black")
            draw.text((x - 5, origin_y + plot_h + 8), str(value), fill="black", font=font)

        for i, value in enumerate(profile):
            y0 = origin_y + i * (bar_h + gap)
            y1 = y0 + bar_h
            x1 = origin_x + int(value / max(max_value, 1) * plot_w)
            draw.rectangle((origin_x, y0, x1, y1), fill=(155, 65, 55))

        draw.text((8, origin_y), "y=0", fill="black", font=font)
        draw.text((8, origin_y + plot_h - 12), str(profile.size - 1), fill="black", font=font)

    image.save(path)


def format_float(value):
    return f"{value:.6f}"


def save_csv(rows):
    csv_path = RESULTS_DIR / "features.csv"
    header = [
        "index", "symbol", "unicode", "name", "width", "height", "mass",
        "M1", "M2", "M3", "M4",
        "w1", "w2", "w3", "w4",
        "xc", "yc", "xc_norm", "yc_norm",
        "Ix", "Iy", "Ix_norm", "Iy_norm",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(header)
        for row in rows:
            features = row["features"]
            writer.writerow([
                row["index"],
                row["symbol"],
                f"U+{ord(row['symbol']):04X}",
                row["name"],
                features["width"],
                features["height"],
                features["mass"],
                *features["q_masses"],
                *(format_float(v) for v in features["q_densities"]),
                format_float(features["xc"]),
                format_float(features["yc"]),
                format_float(features["xc_norm"]),
                format_float(features["yc_norm"]),
                format_float(features["ix"]),
                format_float(features["iy"]),
                format_float(features["ix_norm"]),
                format_float(features["iy_norm"]),
            ])
    return csv_path


def save_report(rows):
    report_path = BASE_DIR / "report_lab5.md"
    lines = [
        "# Лабораторная работа №5",
        "## Выделение признаков символов",
        "",
        "### Вариант 8: алфавит Османья",
        "",
        "### Параметры генерации",
        f"- Шрифт: `{FONT_PATH}`",
        f"- Размер шрифта: `{FONT_SIZE}`",
        f"- Количество символов: `{len(ALPHABET)}`",
        "- Принцип: `1 символ = 1 файл`, белые поля обрезаны.",
        "",
        "### Формулы признаков",
        "",
        "```text",
        "I_b(x, y) = 1 для черного пикселя и 0 для белого",
        "M_q = sum I_b(x, y), где (x, y) принадлежит четверти q",
        "w_q = M_q / S_q",
        "M = sum I_b(x, y)",
        "xc = (1/M) * sum x * I_b(x, y)",
        "yc = (1/M) * sum y * I_b(x, y)",
        "xc_norm = xc / (W - 1)",
        "yc_norm = yc / (H - 1)",
        "Ix = sum (y - yc)^2 * I_b(x, y)",
        "Iy = sum (x - xc)^2 * I_b(x, y)",
        "Ix_norm = Ix / (M * H^2)",
        "Iy_norm = Iy / (M * W^2)",
        "profile_X(x) = sum_y I_b(x, y)",
        "profile_Y(y) = sum_x I_b(x, y)",
        "```",
        "",
        "### Результаты генерации символов и профилей",
        "",
        "| № | Символ | Unicode | Название | Эталон | Профиль X | Профиль Y |",
        "|:--:|:------:|:-------:|:---------|:------:|:---------:|:---------:|",
    ]

    for row in rows:
        code = f"U+{ord(row['symbol']):04X}"
        lines.append(
            f"| {row['index']} | {row['symbol']} | `{code}` | {row['name']} | "
            f"![t]({row['template_path']}) | ![px]({row['profile_x_path']}) | "
            f"![py]({row['profile_y_path']}) |"
        )

    lines.extend([
        "",
        "### Табличные признаки",
        "",
        "- Скалярные признаки сохранены в CSV (`;`-разделитель): `results/features.csv`.",
        "- Профили сохранены в PNG с целочисленными подписями осей.",
        "",
        "### Вывод",
        "Для алфавита Османья по варианту 8 сгенерированы эталонные изображения букв, рассчитаны массы четвертей, удельные веса, координаты центра тяжести, нормированные координаты, осевые моменты инерции, нормированные моменты и профили X/Y.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    ensure_dirs()
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Не найден шрифт: {FONT_PATH}")

    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    rows = []

    for index, symbol in enumerate(ALPHABET, start=1):
        sid = symbol_id(index, symbol)
        template = render_symbol(symbol, font)
        template_path = TEMPLATES_DIR / f"{sid}.bmp"
        template.save(template_path)

        binary = binary_black(template)
        features = scalar_features(binary)

        profile_x_path = PROFILES_DIR / f"{sid}_profile_x.png"
        profile_y_path = PROFILES_DIR / f"{sid}_profile_y.png"
        draw_profile(features["profile_x"], profile_x_path, f"{sid} profile X", "x")
        draw_profile(features["profile_y"], profile_y_path, f"{sid} profile Y", "y")

        rows.append({
            "index": index,
            "symbol": symbol,
            "name": unicodedata.name(symbol),
            "features": features,
            "template_path": template_path.relative_to(BASE_DIR).as_posix(),
            "profile_x_path": profile_x_path.relative_to(BASE_DIR).as_posix(),
            "profile_y_path": profile_y_path.relative_to(BASE_DIR).as_posix(),
        })

    save_csv(rows)
    save_report(rows)


if __name__ == "__main__":
    main()

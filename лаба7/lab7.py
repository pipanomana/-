from pathlib import Path
import csv
import math
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "src" / "templates"
CLASSIFICATION_DIR = BASE_DIR / "src" / "classification"
RESULTS_DIR = BASE_DIR / "results"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/NotoSansOsmanya-Regular.ttf")
BASE_FONT_SIZE = 92
EXPERIMENT_FONT_SIZE = 98
THRESHOLD = 128
PROFILE_THRESHOLD = 1
MIN_GAP_TO_SPLIT = 5
MIN_SEGMENT_WIDTH = 3
LETTER_SPACING = 8
WORD_SPACING = 34

# Somali phrase "waan ku jeclahay" ("I love you") written with Osmanya letters.
PHRASE_LATIN = "waan ku jeclahay"
PHRASE = "𐒓𐒛𐒒 𐒏𐒚 𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕"
EXPECTED_SEQUENCE = PHRASE.replace(" ", "")

# Variant 8: Osmanya alphabet letters. U+104A0..U+104A9 are digits, not letters.
ALPHABET = [chr(codepoint) for codepoint in range(0x10480, 0x1049E)]
FEATURE_NAMES = ["mass_norm", "xc_norm", "yc_norm", "ix_norm", "iy_norm"]


def ensure_dirs():
    for path in [TEMPLATES_DIR, CLASSIFICATION_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def symbol_id(index, symbol):
    return f"sym_{index:02d}_U{ord(symbol):04X}"


def binary_black(image):
    return (np.array(image.convert("L")) < THRESHOLD).astype(np.uint8)


def crop_to_black(image, pad=0):
    arr = np.array(image.convert("L"))
    rows, cols = np.where(arr < THRESHOLD)
    if rows.size == 0:
        raise ValueError("На изображении нет черных пикселей")

    left = max(int(cols.min()) - pad, 0)
    upper = max(int(rows.min()) - pad, 0)
    right = min(int(cols.max()) + pad + 1, image.width)
    lower = min(int(rows.max()) + pad + 1, image.height)
    return image.crop((left, upper, right, lower))


def render_symbol(symbol, font):
    canvas_size = font.size * 3
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    x = (canvas_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)
    return crop_to_black(image, pad=0)


def render_text(text, font):
    dummy = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy)
    char_boxes = [(char, draw.textbbox((0, 0), char, font=font)) for char in text if char != " "]
    min_top = min(bbox[1] for _, bbox in char_boxes)
    max_bottom = max(bbox[3] for _, bbox in char_boxes)
    text_height = max_bottom - min_top

    text_width = 0
    for char in text:
        if char == " ":
            text_width += WORD_SPACING
            continue
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width += bbox[2] - bbox[0] + LETTER_SPACING

    image = Image.new("L", (text_width + 40, text_height + 40), 255)
    draw = ImageDraw.Draw(image)
    x = 20
    y = 20 - min_top
    for char in text:
        if char == " ":
            x += WORD_SPACING
            continue
        bbox = draw.textbbox((0, 0), char, font=font)
        draw.text((x - bbox[0], y), char, font=font, fill=0)
        x += bbox[2] - bbox[0] + LETTER_SPACING

    return crop_to_black(image, pad=0)


def vertical_profile(binary):
    return binary.sum(axis=0).astype(int)


def runs_from_mask(mask):
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def merge_close_runs(runs, min_gap):
    if not runs:
        return []

    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap = start - prev_end - 1
        if gap < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def segment_characters(binary):
    profile = vertical_profile(binary)
    active_columns = profile > PROFILE_THRESHOLD
    runs = merge_close_runs(runs_from_mask(active_columns), MIN_GAP_TO_SPLIT)

    boxes = []
    for x0, x1 in runs:
        if x1 - x0 + 1 < MIN_SEGMENT_WIDTH:
            continue
        segment = binary[:, x0:x1 + 1]
        rows = np.where(segment.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        boxes.append((int(x0), int(rows.min()), int(x1), int(rows.max())))
    return boxes


def feature_vector(binary):
    height, width = binary.shape
    mass = int(binary.sum())
    if mass == 0:
        raise ValueError("Нельзя рассчитать признаки для пустого символа")

    y_coords, x_coords = np.indices(binary.shape)
    xc = float((x_coords * binary).sum() / mass)
    yc = float((y_coords * binary).sum() / mass)

    mass_norm = float(mass / (width * height))
    xc_norm = float(xc / (width - 1)) if width > 1 else 0.0
    yc_norm = float(yc / (height - 1)) if height > 1 else 0.0
    ix = float((((y_coords - yc) ** 2) * binary).sum())
    iy = float((((x_coords - xc) ** 2) * binary).sum())
    ix_norm = float(ix / (mass * height * height))
    iy_norm = float(iy / (mass * width * width))
    return np.array([mass_norm, xc_norm, yc_norm, ix_norm, iy_norm], dtype=np.float64)


def euclidean_distance(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def similarity(a, b):
    distance = euclidean_distance(a, b)
    return 1.0 / (1.0 + distance), distance


def draw_segmentation_boxes(image, boxes, path):
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    font = ImageFont.load_default()
    for index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        draw.rectangle((x0, y0, x1, y1), outline=(220, 40, 35), width=2)
        draw.text((x0, max(y0 - 12, 0)), str(index), fill=(220, 40, 35), font=font)
    rgb.save(path)


def save_boxes_csv(boxes, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["idx", "x0", "y0", "x1", "y1", "w", "h"])
        for index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            writer.writerow([index, x0, y0, x1, y1, x1 - x0 + 1, y1 - y0 + 1])


def build_templates():
    font = ImageFont.truetype(str(FONT_PATH), BASE_FONT_SIZE)
    rows = []

    for index, symbol in enumerate(ALPHABET, start=1):
        sid = symbol_id(index, symbol)
        template = render_symbol(symbol, font)
        template_path = TEMPLATES_DIR / f"{sid}.bmp"
        template.save(template_path)
        features = feature_vector(binary_black(template))
        rows.append({
            "index": index,
            "symbol": symbol,
            "unicode": f"U+{ord(symbol):04X}",
            "name": unicodedata.name(symbol),
            "path": template_path,
            "features": features,
        })

    save_template_features(rows)
    return rows


def save_template_features(template_rows):
    path = BASE_DIR / "src" / "template_features.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index", "symbol", "unicode", "name", *FEATURE_NAMES])
        for row in template_rows:
            writer.writerow([
                row["index"],
                row["symbol"],
                row["unicode"],
                row["name"],
                *(f"{value:.8f}" for value in row["features"]),
            ])
    return path


def classify_segment(binary, template_rows):
    features = feature_vector(binary)
    hypotheses = []
    for row in template_rows:
        score, distance = similarity(features, row["features"])
        hypotheses.append({
            "symbol": row["symbol"],
            "unicode": row["unicode"],
            "score": score,
            "distance": distance,
        })
    hypotheses.sort(key=lambda item: item["score"], reverse=True)
    return hypotheses


def save_hypotheses(path, all_hypotheses):
    with path.open("w", encoding="utf-8") as file:
        for index, hypotheses in enumerate(all_hypotheses, start=1):
            pairs = ", ".join(f"(\"{h['symbol']}\", {h['score']:.6f})" for h in hypotheses)
            file.write(f"{index}: [{pairs}]\n")


def count_errors(expected, recognized):
    substitutions = sum(1 for a, b in zip(expected, recognized) if a != b)
    return substitutions + abs(len(expected) - len(recognized))


def classify_phrase(mode, font_size, template_rows):
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    phrase_image = render_text(PHRASE, font)
    phrase_path = CLASSIFICATION_DIR / f"{mode}_phrase_mono.bmp"
    phrase_image.save(phrase_path)

    binary = binary_black(phrase_image)
    boxes = segment_characters(binary)
    boxes_path = CLASSIFICATION_DIR / f"{mode}_boxes.png"
    draw_segmentation_boxes(phrase_image, boxes, boxes_path)
    save_boxes_csv(boxes, CLASSIFICATION_DIR / f"{mode}_boxes.csv")

    all_hypotheses = []
    recognized_symbols = []
    for x0, y0, x1, y1 in boxes:
        segment = phrase_image.crop((x0, y0, x1 + 1, y1 + 1))
        hypotheses = classify_segment(binary_black(segment), template_rows)
        all_hypotheses.append(hypotheses)
        recognized_symbols.append(hypotheses[0]["symbol"])

    hypotheses_path = CLASSIFICATION_DIR / f"{mode}_hypotheses.txt"
    save_hypotheses(hypotheses_path, all_hypotheses)

    recognized = "".join(recognized_symbols)
    errors = count_errors(EXPECTED_SEQUENCE, recognized)
    accuracy = (1.0 - errors / len(EXPECTED_SEQUENCE)) * 100.0
    return {
        "mode": mode,
        "font_size": font_size,
        "phrase_path": phrase_path,
        "boxes_path": boxes_path,
        "boxes_count": len(boxes),
        "hypotheses_path": hypotheses_path,
        "boxes_csv": CLASSIFICATION_DIR / f"{mode}_boxes.csv",
        "expected": EXPECTED_SEQUENCE,
        "recognized": recognized,
        "errors": errors,
        "accuracy": accuracy,
    }


def save_summary_csv(results):
    path = RESULTS_DIR / "classification_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["mode", "font_size", "segments", "errors", "accuracy"])
        for result in results:
            writer.writerow([
                result["mode"],
                result["font_size"],
                result["boxes_count"],
                result["errors"],
                f"{result['accuracy']:.2f}",
            ])
    return path


def rel(path):
    return path.relative_to(BASE_DIR).as_posix()


def save_report(base_result, experiment_result):
    path = BASE_DIR / "report_lab7.md"
    results = [base_result, experiment_result]
    lines = [
        "# Лабораторная работа №7",
        "## Классификация на основе признаков, анализ профилей",
        "",
        "### Вариант 8: алфавит Османья",
        "",
        "### Исходные данные",
        f"- Шрифт: `{FONT_PATH}`",
        f"- Базовый размер: `{BASE_FONT_SIZE}`",
        f"- Экспериментальный размер: `{EXPERIMENT_FONT_SIZE}`",
        f"- Фраза: `{PHRASE}`",
        f"- Транслитерация: `{PHRASE_LATIN}`",
        f"- Алфавит: `{len(ALPHABET)}` символов",
        "",
        "### Формулы",
        "",
        "Евклидово расстояние в пространстве нормализованных признаков:",
        "",
        "```text",
        "d(a, b) = sqrt(sum_i (a_i - b_i)^2)",
        "```",
        "",
        "Мера близости:",
        "",
        "```text",
        "S(a, b) = 1 / (1 + d(a, b))",
        "```",
        "",
        "Процент правильного распознавания:",
        "",
        "```text",
        "P = (1 - E / N) * 100%",
        "```",
        "",
        "Использованные признаки: нормированная масса, нормированные координаты центра тяжести, нормированные осевые моменты инерции.",
        "",
        "### 1. Эталонные символы и признаки",
        "",
        "- Эталоны: `src/templates/`",
        "- Таблица признаков: `src/template_features.csv` (`;`-разделитель)",
        "",
        "### 2. Классификация базовой строки",
        "",
        "| Монохромная строка | Сегментация (прямоугольники) |",
        "|:------------------:|:----------------------------:|",
        f"| ![base_phrase]({rel(base_result['phrase_path'])}) | ![base_boxes]({rel(base_result['boxes_path'])}) |",
        "",
        f"- Гипотезы: `{rel(base_result['hypotheses_path'])}`",
        f"- Координаты сегментов: `{rel(base_result['boxes_csv'])}`",
        f"- Эталонная последовательность (без пробелов): `{base_result['expected']}`",
        f"- Распознанная последовательность: `{base_result['recognized']}`",
        f"- Ошибок: `{base_result['errors']}`",
        f"- Точность: `{base_result['accuracy']:.2f}%`",
        "",
        "### 3. Эксперимент с другим размером шрифта",
        "",
        "| Монохромная строка | Сегментация (прямоугольники) |",
        "|:------------------:|:----------------------------:|",
        f"| ![exp_phrase]({rel(experiment_result['phrase_path'])}) | ![exp_boxes]({rel(experiment_result['boxes_path'])}) |",
        "",
        f"- Гипотезы: `{rel(experiment_result['hypotheses_path'])}`",
        f"- Координаты сегментов: `{rel(experiment_result['boxes_csv'])}`",
        f"- Эталонная последовательность (без пробелов): `{experiment_result['expected']}`",
        f"- Распознанная последовательность: `{experiment_result['recognized']}`",
        f"- Ошибок: `{experiment_result['errors']}`",
        f"- Точность: `{experiment_result['accuracy']:.2f}%`",
        "",
        "### 4. Сравнение результатов",
        "",
        "| Режим | Размер шрифта | Сегментов | Ошибок | Точность (%) |",
        "|:------|--------------:|----------:|-------:|-------------:|",
    ]

    labels = {"base": "Базовый", "experiment": "Эксперимент"}
    for result in results:
        lines.append(
            f"| {labels[result['mode']]} | {result['font_size']} | {result['boxes_count']} | "
            f"{result['errors']} | {result['accuracy']:.2f} |"
        )

    lines.extend([
        "",
        "Сводная таблица сохранена в `results/classification_summary.csv`.",
        "",
        "### Вывод",
        "Реализована классификация символов по евклидовой мере близости нормализованных признаков. Для каждого сегмента получены ранжированные гипотезы, построена распознанная строка, рассчитаны ошибки и процент верного распознавания. Проведен эксперимент с изменением размера шрифта.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ensure_dirs()
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Не найден шрифт: {FONT_PATH}")

    template_rows = build_templates()
    base_result = classify_phrase("base", BASE_FONT_SIZE, template_rows)
    experiment_result = classify_phrase("experiment", EXPERIMENT_FONT_SIZE, template_rows)
    save_summary_csv([base_result, experiment_result])
    save_report(base_result, experiment_result)


if __name__ == "__main__":
    main()

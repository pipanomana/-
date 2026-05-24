from pathlib import Path
import csv
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "src" / "input"
PROFILES_DIR = BASE_DIR / "src" / "profiles"
SEGMENTS_DIR = BASE_DIR / "src" / "segments"
ALPHABET_TEMPLATES_DIR = BASE_DIR / "src" / "alphabet" / "templates"
ALPHABET_PROFILES_DIR = BASE_DIR / "src" / "alphabet" / "profiles"
RESULTS_DIR = BASE_DIR / "results"

FONT_PATH = Path("/System/Library/Fonts/Supplemental/NotoSansOsmanya-Regular.ttf")
FONT_SIZE = 92
THRESHOLD = 128
PROFILE_THRESHOLD = 1
MIN_GAP_TO_SPLIT = 5
MIN_SEGMENT_WIDTH = 3
LETTER_SPACING = 8
WORD_SPACING = 34

# Somali phrase "waan ku jeclahay" ("I love you") written with Osmanya letters.
PHRASE_LATIN = "waan ku jeclahay"
PHRASE = "𐒓𐒛𐒒 𐒏𐒚 𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕"
ALPHABET = [chr(codepoint) for codepoint in range(0x10480, 0x1049E)]


def ensure_dirs():
    for path in [
        INPUT_DIR,
        PROFILES_DIR,
        SEGMENTS_DIR,
        ALPHABET_TEMPLATES_DIR,
        ALPHABET_PROFILES_DIR,
        RESULTS_DIR,
    ]:
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


def render_symbol(symbol, font):
    canvas_size = FONT_SIZE * 3
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), symbol, font=font)
    x = (canvas_size - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (canvas_size - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), symbol, font=font, fill=0)
    return crop_to_black(image, pad=1)


def horizontal_profile(binary):
    return binary.sum(axis=1).astype(int)


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
    v_profile = vertical_profile(binary)
    active_columns = v_profile > PROFILE_THRESHOLD
    runs = runs_from_mask(active_columns)
    runs = merge_close_runs(runs, MIN_GAP_TO_SPLIT)

    boxes = []
    for x0, x1 in runs:
        if x1 - x0 + 1 < MIN_SEGMENT_WIDTH:
            continue
        segment = binary[:, x0:x1 + 1]
        rows = np.where(segment.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        y0 = int(rows.min())
        y1 = int(rows.max())
        boxes.append((int(x0), y0, int(x1), y1))
    return boxes


def nice_ticks(max_value):
    if max_value <= 0:
        return [0]
    count = min(max_value, 5)
    return sorted(set(int(round(max_value * i / count)) for i in range(count + 1)))


def draw_profile(profile, path, title, orientation):
    profile = np.asarray(profile, dtype=int)
    max_value = int(profile.max()) if profile.size else 0
    ticks = nice_ticks(max_value)
    font = ImageFont.load_default()

    margin_left = 48
    margin_right = 18
    margin_top = 28
    margin_bottom = 38

    if orientation == "vertical":
        bar_w = 5
        gap = 1
        plot_w = max(320, int(profile.size) * (bar_w + gap))
        plot_h = 200
        image = Image.new("RGB", (margin_left + plot_w + margin_right, margin_top + plot_h + margin_bottom), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 6), title, fill="black", font=font)
        origin_x = margin_left
        origin_y = margin_top + plot_h
        draw.line((origin_x, margin_top, origin_x, origin_y), fill="black")
        draw.line((origin_x, origin_y, origin_x + plot_w, origin_y), fill="black")
        for value in ticks:
            y = origin_y - int(value / max(max_value, 1) * plot_h)
            draw.line((origin_x - 4, y, origin_x, y), fill="black")
            draw.text((4, y - 5), str(value), fill="black", font=font)
        for i, value in enumerate(profile):
            x0 = origin_x + i * (bar_w + gap)
            x1 = x0 + bar_w
            y0 = origin_y - int(value / max(max_value, 1) * plot_h)
            draw.rectangle((x0, y0, x1, origin_y), fill=(42, 92, 165))
        draw.text((origin_x, origin_y + 12), "x=0", fill="black", font=font)
        draw.text((origin_x + plot_w - 28, origin_y + 12), str(profile.size - 1), fill="black", font=font)
    else:
        bar_h = 5
        gap = 1
        plot_w = 260
        plot_h = max(260, int(profile.size) * (bar_h + gap))
        image = Image.new("RGB", (margin_left + plot_w + margin_right, margin_top + plot_h + margin_bottom), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin_left, 6), title, fill="black", font=font)
        origin_x = margin_left
        origin_y = margin_top
        draw.line((origin_x, origin_y, origin_x, origin_y + plot_h), fill="black")
        draw.line((origin_x, origin_y + plot_h, origin_x + plot_w, origin_y + plot_h), fill="black")
        for value in ticks:
            x = origin_x + int(value / max(max_value, 1) * plot_w)
            draw.line((x, origin_y + plot_h, x, origin_y + plot_h + 4), fill="black")
            draw.text((x - 5, origin_y + plot_h + 8), str(value), fill="black", font=font)
        for i, value in enumerate(profile):
            y0 = origin_y + i * (bar_h + gap)
            y1 = y0 + bar_h
            x1 = origin_x + int(value / max(max_value, 1) * plot_w)
            draw.rectangle((origin_x, y0, x1, y1), fill=(156, 66, 54))
        draw.text((8, origin_y), "y=0", fill="black", font=font)
        draw.text((8, origin_y + plot_h - 12), str(profile.size - 1), fill="black", font=font)

    image.save(path)


def draw_segmentation_boxes(image, boxes, path):
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    for index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        draw.rectangle((x0, y0, x1, y1), outline=(220, 40, 35), width=2)
        draw.text((x0, max(y0 - 12, 0)), str(index), fill=(220, 40, 35), font=ImageFont.load_default())
    rgb.save(path)


def save_segments(image, boxes):
    paths = []
    for index, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = box
        segment = image.crop((x0, y0, x1 + 1, y1 + 1))
        path = SEGMENTS_DIR / f"segment_{index:02d}.bmp"
        segment.save(path)
        paths.append(path)
    return paths


def save_boxes_csv(boxes):
    path = RESULTS_DIR / "segments_boxes.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["idx", "x0", "y0", "x1", "y1", "w", "h"])
        for index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            writer.writerow([index, x0, y0, x1, y1, x1 - x0 + 1, y1 - y0 + 1])
    return path


def build_alphabet_profiles(font):
    rows = []
    for index, symbol in enumerate(ALPHABET, start=1):
        sid = symbol_id(index, symbol)
        template = render_symbol(symbol, font)
        template_path = ALPHABET_TEMPLATES_DIR / f"{sid}.bmp"
        template.save(template_path)

        binary = binary_black(template)
        profile_x_path = ALPHABET_PROFILES_DIR / f"{sid}_profile_x.png"
        profile_y_path = ALPHABET_PROFILES_DIR / f"{sid}_profile_y.png"
        draw_profile(vertical_profile(binary), profile_x_path, f"{sid} profile X", "vertical")
        draw_profile(horizontal_profile(binary), profile_y_path, f"{sid} profile Y", "horizontal")

        rows.append({
            "index": index,
            "symbol": symbol,
            "code": f"U+{ord(symbol):04X}",
            "name": unicodedata.name(symbol),
            "template": template_path.relative_to(BASE_DIR).as_posix(),
            "profile_x": profile_x_path.relative_to(BASE_DIR).as_posix(),
            "profile_y": profile_y_path.relative_to(BASE_DIR).as_posix(),
        })
    return rows


def save_report(image, boxes, segment_paths, alphabet_rows):
    path = BASE_DIR / "report_lab6.md"
    lines = [
        "# Лабораторная работа №6",
        "## Сегментация текста",
        "",
        "### Вариант 8: алфавит Османья",
        "",
        "### Исходные данные",
        f"- Фраза: `{PHRASE}`",
        f"- Транслитерация: `{PHRASE_LATIN}`",
        f"- Шрифт: `{FONT_PATH}`, размер `{FONT_SIZE}`",
        f"- Межбуквенный интервал при подготовке строки: `{LETTER_SPACING}` px",
        f"- Размер монохромного изображения: `{image.width}x{image.height}`",
        f"- Количество найденных символов: `{len(boxes)}`",
        "",
        "### Формулы профилей",
        "",
        "```text",
        "H(y) = sum_x I_b(x, y)",
        "V(x) = sum_y I_b(x, y)",
        "```",
        "",
        "Где `I_b(x,y)=1` для черного пикселя и `0` для белого.",
        "",
        "### 1. Подготовка строки",
        "",
        "#### 1.1 Монохромное изображение фразы",
        "![input](src/input/phrase_mono.bmp)",
        "",
        "### 2. Профили изображения",
        "",
        "| Горизонтальный профиль | Вертикальный профиль |",
        "|:----------------------:|:--------------------:|",
        "| ![h](src/profiles/horizontal_profile.png) | ![v](src/profiles/vertical_profile.png) |",
        "",
        "### 3. Сегментация символов по вертикальному профилю с прореживанием",
        "",
        "#### 3.1 Обрамляющие прямоугольники",
        "![boxes](src/segments/segmentation_boxes.png)",
        "",
        "#### 3.2 Вырезанные сегменты",
        "",
    ]

    for index, segment_path in enumerate(segment_paths, start=1):
        rel_path = segment_path.relative_to(BASE_DIR).as_posix()
        lines.append(f"- Сегмент {index}: `[segment_{index:02d}]` -> ![s{index}]({rel_path})")

    lines.extend([
        "",
        "#### 3.3 Массив координат прямоугольников",
        "",
        "| idx | x0 | y0 | x1 | y1 | w | h |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        lines.append(f"| {index} | {x0} | {y0} | {x1} | {y1} | {x1 - x0 + 1} | {y1 - y0 + 1} |")

    lines.extend([
        "",
        "CSV с координатами (`;`-разделитель): `results/segments_boxes.csv`.",
        "",
        "### 4. Профили символов выбранного алфавита",
        "",
        "- Эталоны символов: `src/alphabet/templates/`.",
        "- Профили X/Y: `src/alphabet/profiles/`.",
        f"- Построены для всех {len(alphabet_rows)} букв алфавита Османья варианта 8.",
        "",
        "Пример (первые 6 символов):",
        "",
        "| Символ | Unicode | Эталон | Профиль X | Профиль Y |",
        "|:------:|:-------:|:------:|:---------:|:---------:|",
    ])

    for row in alphabet_rows[:6]:
        lines.append(
            f"| {row['symbol']} | `{row['code']}` | ![t]({row['template']}) | "
            f"![px]({row['profile_x']}) | ![py]({row['profile_y']}) |"
        )

    lines.extend([
        "",
        "### Вывод",
        "Реализованы расчет горизонтального и вертикального профилей, сегментация символов строки на основе вертикального профиля с прореживанием, сохранение массива координат прямоугольников и построение профилей всех букв алфавита Османья.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    ensure_dirs()
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Не найден шрифт: {FONT_PATH}")

    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    phrase_image = render_text(PHRASE, font)
    phrase_path = INPUT_DIR / "phrase_mono.bmp"
    phrase_image.save(phrase_path)

    binary = binary_black(phrase_image)
    h_profile = horizontal_profile(binary)
    v_profile = vertical_profile(binary)
    draw_profile(h_profile, PROFILES_DIR / "horizontal_profile.png", "Horizontal profile H(y)", "horizontal")
    draw_profile(v_profile, PROFILES_DIR / "vertical_profile.png", "Vertical profile V(x)", "vertical")

    boxes = segment_characters(binary)
    draw_segmentation_boxes(phrase_image, boxes, SEGMENTS_DIR / "segmentation_boxes.png")
    segment_paths = save_segments(phrase_image, boxes)
    save_boxes_csv(boxes)

    alphabet_rows = build_alphabet_profiles(font)
    save_report(phrase_image, boxes, segment_paths, alphabet_rows)


if __name__ == "__main__":
    main()

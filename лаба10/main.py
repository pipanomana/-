from pathlib import Path
import argparse
import csv
import math
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "output"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
SPECTROGRAMS_DIR = OUTPUT_DIR / "spectrograms"
SEGMENTATION_DIR = OUTPUT_DIR / "segmentation"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

RAW_ALPHABET = {
    "+": BASE_DIR / "+.wav",
    "0": BASE_DIR / "0.wav",
    "1": BASE_DIR / "1.wav",
    "2": BASE_DIR / "2.wav",
    "3": BASE_DIR / "3.wav",
    "4": BASE_DIR / "4.wav",
    "5": BASE_DIR / "5.wav",
    "6": BASE_DIR / "6.wav",
    "7": BASE_DIR / "7.wav",
    "8": BASE_DIR / "8.wav",
    "9": BASE_DIR / "9.wav",
}
RAW_PHONE = BASE_DIR / "12.wav"

STFT_WINDOW_MS = 30
STFT_OVERLAP = 0.67
SEGMENT_WINDOW_MS = 25
SEGMENT_HOP_MS = 10
BAND_COUNT = 24
MIN_FREQ = 80.0
MAX_FREQ = 5500.0
MERGE_GAP_SECONDS = 0.18
MIN_SEGMENT_SECONDS = 0.10
PAD_SECONDS = 0.04


def ensure_dirs():
    for path in [
        SAMPLES_DIR,
        OUTPUT_DIR,
        SEGMENTS_DIR,
        SPECTROGRAMS_DIR,
        SEGMENTATION_DIR,
        TABLES_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 3:
        bytes_data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_data[:, 0].astype(np.int32)
            | (bytes_data[:, 1].astype(np.int32) << 8)
            | (bytes_data[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float64) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return sample_rate, data, channels, sample_width


def write_wav(path, sample_rate, signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def normalize(signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    max_abs = float(np.max(np.abs(signal))) if signal.size else 0.0
    if max_abs < 1e-12:
        return signal
    return signal / max_abs * 0.95


def frame_rms(signal, sample_rate, window_ms=SEGMENT_WINDOW_MS, hop_ms=SEGMENT_HOP_MS):
    win = max(1, int(sample_rate * window_ms / 1000))
    hop = max(1, int(sample_rate * hop_ms / 1000))
    if len(signal) < win:
        signal = np.pad(signal, (0, win - len(signal)))

    values = []
    starts = []
    for start in range(0, len(signal) - win + 1, hop):
        frame = signal[start:start + win]
        values.append(float(np.sqrt(np.mean(frame ** 2))))
        starts.append(start)
    return np.array(values), np.array(starts), win, hop


def trim_silence(signal, sample_rate):
    rms, starts, win, _ = frame_rms(signal, sample_rate)
    if rms.size == 0:
        return signal
    threshold = max(np.percentile(rms, 20) * 2.0, np.max(rms) * 0.08)
    active = np.where(rms >= threshold)[0]
    if active.size == 0:
        return signal
    pad = int(PAD_SECONDS * sample_rate)
    start = max(int(starts[active[0]]) - pad, 0)
    end = min(int(starts[active[-1]] + win) + pad, len(signal))
    return signal[start:end]


def segment_speech(signal, sample_rate):
    rms, starts, win, hop = frame_rms(signal, sample_rate)
    if rms.size == 0:
        return [], rms, starts, 0.0

    noise_floor = np.percentile(rms, 20)
    high = np.percentile(rms, 90)
    threshold = max(noise_floor * 2.5, high * 0.18)
    active = rms >= threshold

    runs = []
    start_idx = None
    for index, value in enumerate(active):
        if value and start_idx is None:
            start_idx = index
        elif not value and start_idx is not None:
            runs.append((start_idx, index - 1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(active) - 1))

    max_gap = int(MERGE_GAP_SECONDS / (hop / sample_rate))
    merged = []
    for run in runs:
        if not merged:
            merged.append(run)
            continue
        prev_start, prev_end = merged[-1]
        gap = run[0] - prev_end - 1
        if gap <= max_gap:
            merged[-1] = (prev_start, run[1])
        else:
            merged.append(run)

    pad = int(PAD_SECONDS * sample_rate)
    min_len = int(MIN_SEGMENT_SECONDS * sample_rate)
    segments = []
    for left, right in merged:
        start = max(int(starts[left]) - pad, 0)
        end = min(int(starts[right] + win) + pad, len(signal))
        if end - start >= min_len:
            segments.append((start, end))
    return segments, rms, starts, threshold


def stft(signal, sample_rate):
    n_fft = max(256, int(sample_rate * STFT_WINDOW_MS / 1000))
    n_fft = 1 << int(math.ceil(math.log2(n_fft)))
    hop = max(1, int(n_fft * (1.0 - STFT_OVERLAP)))
    window = np.hanning(n_fft)
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    frames = []
    for start in range(0, len(signal) - n_fft + 1, hop):
        frames.append(np.fft.rfft(signal[start:start + n_fft] * window))
    if not frames:
        frames.append(np.fft.rfft(np.pad(signal, (0, n_fft - len(signal))) * window))
    return np.array(frames).T, n_fft, hop


def log_band_features(signal, sample_rate):
    signal = trim_silence(normalize(signal), sample_rate)
    spec, n_fft, _ = stft(signal, sample_rate)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    edges = np.geomspace(MIN_FREQ, min(MAX_FREQ, sample_rate / 2), BAND_COUNT + 1)
    features = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            features.append(np.zeros(spec.shape[1]))
        else:
            features.append(np.mean(np.log(mag[mask, :]), axis=0))
    feat = np.array(features).T
    feat = feat - feat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norm, 1e-9)


def dtw_distance(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(ai - b[j - 1]))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / (n + m))


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def draw_spectrogram(signal, sample_rate, path):
    spec, n_fft, _ = stft(normalize(signal), sample_rate)
    power = 20.0 * np.log10(np.abs(spec) + 1e-8)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    max_freq = min(MAX_FREQ, sample_rate / 2)
    mask = freqs <= max_freq
    power = power[mask, :]
    vmin = np.percentile(power, 5)
    vmax = np.percentile(power, 99)
    norm = np.clip((power - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)

    width, height = 900, 520
    margin_l, margin_t = 62, 34
    plot_w, plot_h = 800, 430
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    gray = Image.fromarray((norm[::-1, :] * 255).astype(np.uint8), "L")
    gray = gray.resize((plot_w, plot_h), Image.Resampling.BILINEAR)
    color = Image.merge("RGB", (gray, Image.fromarray(np.zeros((plot_h, plot_w), dtype=np.uint8) + 45, "L"), Image.fromarray(255 - np.array(gray), "L")))
    image.paste(color, (margin_l, margin_t))
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline="black")
    draw.text((margin_l, 12), "Phone STFT spectrogram, Hann window", fill="black", font=font)
    draw.text((margin_l, margin_t + plot_h + 12), "time", fill="black", font=font)
    draw.text((10, margin_t), f"{int(max_freq)} Hz", fill="black", font=font)
    draw.text((26, margin_t + plot_h - 10), "0 Hz", fill="black", font=font)
    image.save(path)


def draw_segmentation(signal, sample_rate, segments, rms, starts, threshold, path):
    width, height = 1200, 520
    margin_l, margin_t = 60, 30
    plot_w, wave_h, rms_h = 1100, 250, 160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_l, 10), "Phone waveform and RMS segmentation", fill="black", font=font)
    center = margin_t + wave_h // 2
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + wave_h), outline=(210, 210, 210))
    draw.line((margin_l, center, margin_l + plot_w, center), fill=(180, 180, 180))
    step = max(1, len(signal) // plot_w)
    reduced = signal[:plot_w * step].reshape(-1, step)
    max_vals = reduced.max(axis=1)
    min_vals = reduced.min(axis=1)
    scale = (wave_h - 20) / 2
    for x, (mn, mx) in enumerate(zip(min_vals, max_vals)):
        xx = margin_l + x
        draw.line((xx, center - int(mx * scale), xx, center - int(mn * scale)), fill=(45, 95, 170))

    for start, end in segments:
        x0 = margin_l + int(start / len(signal) * plot_w)
        x1 = margin_l + int(end / len(signal) * plot_w)
        draw.rectangle((x0, margin_t, x1, margin_t + wave_h), outline=(210, 50, 45), width=2)

    rms_top = margin_t + wave_h + 45
    draw.rectangle((margin_l, rms_top, margin_l + plot_w, rms_top + rms_h), outline=(210, 210, 210))
    max_rms = max(float(rms.max()) if rms.size else 0.0, threshold, 1e-9)
    prev = None
    for value, start in zip(rms, starts):
        x = margin_l + int(start / len(signal) * plot_w)
        y = rms_top + rms_h - int(value / max_rms * rms_h)
        if prev is not None:
            draw.line((prev[0], prev[1], x, y), fill=(60, 150, 80), width=2)
        prev = (x, y)
    y_thr = rms_top + rms_h - int(threshold / max_rms * rms_h)
    draw.line((margin_l, y_thr, margin_l + plot_w, y_thr), fill=(200, 60, 50), width=1)
    draw.text((margin_l, rms_top - 16), "RMS and adaptive threshold", fill="black", font=font)
    image.save(path)


def prepare_samples():
    sample_rate = None
    templates = {}
    for symbol, path in RAW_ALPHABET.items():
        sr, signal, _, _ = read_wav(path)
        if sample_rate is None:
            sample_rate = sr
        if sr != sample_rate:
            raise ValueError("Все WAV-файлы должны иметь одинаковую частоту дискретизации")
        prepared = trim_silence(normalize(signal), sr)
        write_wav(SAMPLES_DIR / f"{'plus' if symbol == '+' else symbol}.wav", sr, prepared)
        templates[symbol] = log_band_features(prepared, sr)

    sr, phone, channels, _ = read_wav(RAW_PHONE)
    if sr != sample_rate:
        raise ValueError("Телефонная дорожка имеет другую частоту дискретизации")
    phone = normalize(phone)
    write_wav(SAMPLES_DIR / "phone.wav", sr, phone)
    return sample_rate, phone, channels, templates


def recognize(phone, sample_rate, templates):
    segments, rms, starts, threshold = segment_speech(phone, sample_rate)
    rows = []
    recognized = []
    confidences = []

    for index, (start, end) in enumerate(segments, start=1):
        segment = phone[start:end]
        write_wav(SEGMENTS_DIR / f"segment_{index:02d}.wav", sample_rate, segment)
        features = log_band_features(segment, sample_rate)
        distances = []
        for symbol, template in templates.items():
            distances.append((symbol, dtw_distance(features, template)))
        distances.sort(key=lambda item: item[1])
        best_symbol, best_distance = distances[0]
        second_distance = distances[1][1] if len(distances) > 1 else best_distance
        confidence = (second_distance - best_distance) / second_distance if second_distance > 1e-12 else 1.0
        confidence = max(0.0, min(1.0, confidence))
        recognized.append(best_symbol)
        confidences.append(confidence)
        rows.append({
            "index": index,
            "start": start / sample_rate,
            "end": end / sample_rate,
            "duration": (end - start) / sample_rate,
            "symbol": best_symbol,
            "confidence": confidence,
            "best_distance": best_distance,
            "second_distance": second_distance,
            "hypotheses": distances,
        })
    return rows, "".join(recognized), float(np.mean(confidences)) if confidences else 0.0, segments, rms, starts, threshold


def save_recognition_csv(rows):
    path = TABLES_DIR / "recognition.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["idx", "start_s", "end_s", "duration_s", "recognized", "confidence", "best_distance", "second_distance", "top_hypotheses"])
        for row in rows:
            hypotheses = " ".join(f"{symbol}:{distance:.5f}" for symbol, distance in row["hypotheses"][:5])
            writer.writerow([
                row["index"],
                f"{row['start']:.3f}",
                f"{row['end']:.3f}",
                f"{row['duration']:.3f}",
                row["symbol"],
                f"{row['confidence']:.6f}",
                f"{row['best_distance']:.6f}",
                f"{row['second_distance']:.6f}",
                hypotheses,
            ])


def save_report(sample_rate, phone, channels, rows, recognized, expected, errors, confidence):
    report = [
        "# Лабораторная работа №10: Обработка голоса",
        "",
        "Вариант 8: анализатор речи.",
        "",
        "## Что сделано",
        "1. Подготовлены записи в `samples` в WAV-формате: `PCM 16 bit`, `mono`.",
        "2. Загружен алфавит: цифры `0-9` и слово `плюс`.",
        "3. Для `samples/phone.wav` построена спектрограмма с оконным преобразованием Фурье и окном Ханна.",
        "4. Запись телефонного номера сегментирована по коротковременной RMS-энергии.",
        "5. Каждый сегмент сопоставлен с образцами алфавита методом DTW по логарифмическим спектральным полосам.",
        "6. Телефонный номер распознан как цепочка символов.",
        "7. Посчитаны число ошибок и оценка достоверности, если передана эталонная цепочка.",
        "",
        "## Метод",
        "",
        "STFT:",
        "",
        "```text",
        "STFT{x[n]}(m, w) = sum x[n] w[n-m] exp(-jwn)",
        "spectrogram{x[n]} = |STFT{x[n]}|^2",
        "```",
        "",
        "Окно Ханна:",
        "",
        "```text",
        "w(n) = 0.5 - 0.5 * cos(2*pi*n / (N - 1))",
        "```",
        "",
        "Сегментация:",
        "",
        "```text",
        "RMS = sqrt(mean(x^2))",
        "```",
        "",
        "Оценка достоверности сегмента:",
        "",
        "```text",
        "confidence = (second_distance - best_distance) / second_distance",
        "```",
        "",
        "## Структура",
        "- `main.py` — код лабораторной.",
        "- `samples/*.wav` — подготовленные WAV-записи алфавита и телефонного номера.",
        "- `output/spectrograms/phone_spectrogram.png` — спектрограмма телефонного номера.",
        "- `output/segmentation/phone_segments.png` — график сегментации.",
        "- `output/segments/*.wav` — вырезанные сегменты.",
        "- `output/tables/recognition.csv` — таблица распознавания.",
        "- `output/reports/report.txt` — краткий текстовый отчет.",
        "",
        "## Полученные результаты",
        "",
        "Файл `samples/phone.wav`:",
        "",
        "```text",
        f"частота дискретизации = {sample_rate} Гц",
        f"количество каналов исходной записи = {channels}",
        f"длительность = {len(phone) / sample_rate:.3f} с",
        "```",
        "",
        f"Сегментация нашла `{len(rows)}` фрагментов.",
        "",
        "Распознанная цепочка:",
        "",
        "```text",
        recognized,
        "```",
        "",
    ]

    if expected:
        report.extend([
            "Эталонная цепочка:",
            "",
            "```text",
            expected,
            "```",
            "",
            "Число ошибок по расстоянию Левенштейна:",
            "",
            "```text",
            str(errors),
            "```",
            "",
        ])
    else:
        report.append("Эталонная цепочка не задана. Для подсчета ошибок запустите `python main.py --expected <номер>`.")
        report.append("")

    report.extend([
        "Оценка достоверности:",
        "",
        "```text",
        f"{confidence * 100:.1f}%",
        "```",
        "",
        "## Сегменты",
        "",
        "| № | start, s | end, s | symbol | confidence |",
        "|--:|---------:|-------:|:------:|-----------:|",
    ])
    for row in rows:
        report.append(f"| {row['index']} | {row['start']:.3f} | {row['end']:.3f} | `{row['symbol']}` | {row['confidence'] * 100:.1f}% |")

    text = "\n".join(report) + "\n"
    (REPORTS_DIR / "report.txt").write_text(text, encoding="utf-8")
    (BASE_DIR / "report_lab10.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default="", help="Expected phone sequence, for example +89261551495")
    args = parser.parse_args()

    ensure_dirs()
    sample_rate, phone, channels, templates = prepare_samples()
    rows, recognized, confidence, segments, rms, starts, threshold = recognize(phone, sample_rate, templates)
    errors = levenshtein(args.expected, recognized) if args.expected else None

    draw_spectrogram(phone, sample_rate, SPECTROGRAMS_DIR / "phone_spectrogram.png")
    draw_segmentation(phone, sample_rate, segments, rms, starts, threshold, SEGMENTATION_DIR / "phone_segments.png")
    save_recognition_csv(rows)
    save_report(sample_rate, phone, channels, rows, recognized, args.expected, errors, confidence)

    print(f"recognized={recognized}")
    if args.expected:
        print(f"expected={args.expected}")
        print(f"errors={errors}")
    print(f"segments={len(rows)}")
    print(f"confidence={confidence * 100:.1f}%")


if __name__ == "__main__":
    main()

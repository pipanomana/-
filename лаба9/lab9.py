from pathlib import Path
import csv
import math
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
INPUT_WAV = BASE_DIR / "фортепиано.wav"

N_FFT = 2048
HOP = 512
NOVERLAP = N_FFT - HOP
NOISE_WINDOW_SECONDS = 0.5
NOISE_QUANTILE = 0.10
SPECTRAL_SUBTRACTION_ALPHA = 1.25
SPECTRAL_FLOOR = 0.08
DELTA_T = 0.1
DELTA_F = 50
TOP_ENERGY_COUNT = 12


def ensure_dirs():
    SRC_DIR.mkdir(parents=True, exist_ok=True)


def read_wav_mono(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

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

    max_abs = float(np.max(np.abs(data)))
    if max_abs > 1.0:
        data = data / max_abs

    return sample_rate, data.astype(np.float64), channels, sample_width


def write_wav_mono(path, sample_rate, signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def normalize_audio(signal):
    max_abs = float(np.max(np.abs(signal)))
    if max_abs < 1e-12:
        return signal.copy()
    return signal / max_abs * 0.95


def stft(signal, n_fft=N_FFT, hop=HOP):
    window = np.hanning(n_fft)
    if len(signal) < n_fft:
        padded = np.pad(signal, (0, n_fft - len(signal)))
    else:
        frames = 1 + int(math.ceil((len(signal) - n_fft) / hop))
        target_len = frames * hop + n_fft
        padded = np.pad(signal, (0, target_len - len(signal)))

    frames = []
    for start in range(0, len(padded) - n_fft + 1, hop):
        chunk = padded[start:start + n_fft] * window
        frames.append(np.fft.rfft(chunk))
    return np.array(frames).T, window, len(signal)


def istft(spec, window, original_len, hop=HOP):
    n_fft = (spec.shape[0] - 1) * 2
    length = (spec.shape[1] - 1) * hop + n_fft
    output = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)

    for frame_index in range(spec.shape[1]):
        start = frame_index * hop
        frame = np.fft.irfft(spec[:, frame_index], n=n_fft)
        output[start:start + n_fft] += frame * window
        weight[start:start + n_fft] += window ** 2

    valid = weight > 1e-10
    output[valid] /= weight[valid]
    return output[:original_len]


def spectral_subtract(signal):
    spec, window, original_len = stft(signal)
    magnitude = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))

    frame_energy = np.mean(magnitude ** 2, axis=0)
    threshold = np.quantile(frame_energy, NOISE_QUANTILE)
    noise_frames = frame_energy <= threshold
    if not np.any(noise_frames):
        noise_frames[: max(1, int(NOISE_WINDOW_SECONDS * 44100 / HOP))] = True

    noise_profile = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
    reduced = magnitude - SPECTRAL_SUBTRACTION_ALPHA * noise_profile
    floor = SPECTRAL_FLOOR * magnitude
    reduced = np.maximum(reduced, floor)
    denoised_spec = reduced * phase
    denoised = istft(denoised_spec, window, original_len)
    return normalize_audio(denoised), spec, denoised_spec, noise_frames


def power_db(spec):
    power = np.abs(spec) ** 2
    return 10.0 * np.log10(power + 1e-12)


def estimate_noise_rms(signal, sample_rate):
    win = int(NOISE_WINDOW_SECONDS * sample_rate)
    if len(signal) <= win:
        return float(np.sqrt(np.mean(signal ** 2)))

    hop = max(1, win // 4)
    rms_values = []
    for start in range(0, len(signal) - win + 1, hop):
        chunk = signal[start:start + win]
        rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
    return min(rms_values)


def snr_from_noise_rms(signal, noise_rms):
    p_signal = float(np.mean(signal ** 2))
    p_noise = float(noise_rms ** 2)
    return 10.0 * math.log10((p_signal + 1e-12) / (p_noise + 1e-12))


def draw_waveforms(sample_rate, original, denoised, path):
    width, height = 1200, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin_l, margin_r = 60, 20
    margin_t, margin_b = 34, 28
    plot_w = width - margin_l - margin_r
    half_h = (height - margin_t - margin_b) // 2

    def draw_one(signal, top, color, title):
        center = top + half_h // 2
        draw.rectangle((margin_l, top, margin_l + plot_w, top + half_h - 8), outline=(210, 210, 210))
        draw.line((margin_l, center, margin_l + plot_w, center), fill=(180, 180, 180))
        draw.text((margin_l, top - 16), title, fill="black", font=font)
        step = max(1, len(signal) // plot_w)
        reduced = signal[:plot_w * step].reshape(-1, step)
        max_vals = reduced.max(axis=1)
        min_vals = reduced.min(axis=1)
        scale = (half_h - 20) / 2.0
        for x, (mn, mx) in enumerate(zip(min_vals, max_vals)):
            xx = margin_l + x
            y0 = center - int(mx * scale)
            y1 = center - int(mn * scale)
            draw.line((xx, y0, xx, y1), fill=color)

    draw_one(original, margin_t, (45, 95, 170), "Noisy input waveform")
    draw_one(denoised, margin_t + half_h, (190, 70, 45), "Denoised output waveform")
    duration = len(original) / sample_rate
    draw.text((width - 190, height - 20), f"duration {duration:.3f} s", fill="black", font=font)
    image.save(path)


def draw_spectrogram(spec, sample_rate, path, title):
    db = power_db(spec)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    max_freq = min(10000, sample_rate / 2)
    freq_mask = freqs <= max_freq
    db = db[freq_mask, :]
    freqs = freqs[freq_mask]

    vmin = np.percentile(db, 5)
    vmax = np.percentile(db, 99)
    norm = np.clip((db - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)

    out_w = 900
    out_h = 520
    plot_w = 820
    plot_h = 440
    image = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin_l, margin_t = 60, 36

    spec_img = Image.fromarray((norm[::-1, :] * 255).astype(np.uint8), "L")
    spec_img = spec_img.resize((plot_w, plot_h), Image.Resampling.BILINEAR)
    color = Image.merge("RGB", (
        spec_img,
        Image.fromarray(np.zeros((plot_h, plot_w), dtype=np.uint8) + 40, "L"),
        Image.fromarray(255 - np.array(spec_img), "L"),
    ))
    image.paste(color, (margin_l, margin_t))
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline="black")
    draw.text((margin_l, 12), title, fill="black", font=font)
    draw.text((margin_l, margin_t + plot_h + 12), "time", fill="black", font=font)
    draw.text((8, margin_t + 4), f"{int(max_freq)} Hz", fill="black", font=font)
    draw.text((24, margin_t + plot_h - 10), "0 Hz", fill="black", font=font)
    image.save(path)


def high_energy_moments(spec, sample_rate, top_n=TOP_ENERGY_COUNT):
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    frame_times = np.arange(spec.shape[1]) * HOP / sample_rate
    duration = frame_times[-1] if frame_times.size else 0.0
    max_freq = min(5000, sample_rate / 2)

    rows = []
    t_edges = np.arange(0.0, duration + DELTA_T, DELTA_T)
    f_edges = np.arange(0.0, max_freq + DELTA_F, DELTA_F)
    for ti in range(len(t_edges) - 1):
        t0, t1 = t_edges[ti], t_edges[ti + 1]
        t_mask = (frame_times >= t0) & (frame_times < t1)
        if not np.any(t_mask):
            continue
        for fi in range(len(f_edges) - 1):
            f0, f1 = f_edges[fi], f_edges[fi + 1]
            f_mask = (freqs >= f0) & (freqs < f1)
            if not np.any(f_mask):
                continue
            energy = float(np.mean(power[np.ix_(f_mask, t_mask)]))
            rows.append({"time": t0, "f0": f0, "f1": f1, "energy": energy})

    rows.sort(key=lambda row: row["energy"], reverse=True)
    return rows[:top_n], rows


def save_high_energy_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["rank", "time_s", "freq_from_hz", "freq_to_hz", "energy"])
        for rank, row in enumerate(rows, start=1):
            writer.writerow([rank, f"{row['time']:.3f}", f"{row['f0']:.0f}", f"{row['f1']:.0f}", f"{row['energy']:.8e}"])


def write_report(metrics, top_rows):
    lines = [
        "# Лабораторная работа №9",
        "## Анализ шума",
        "",
        "### Вариант 8",
        "",
        "### Исходные данные",
        "- Формат дорожки: WAV, исходная запись приведена к моно.",
        "- Тип дорожки: запись музыкального инструмента (`фортепиано.wav`).",
        f"- Частота дискретизации: `{metrics['sample_rate']}` Гц.",
        f"- Длительность: `{metrics['duration']:.3f}` с.",
        "- Спектрограмма: STFT с окном Ханна.",
        f"- Параметры STFT: `nperseg={N_FFT}`, `noverlap={NOVERLAP}`.",
        f"- Шумоподавление: спектральное вычитание, оценка шума по {NOISE_QUANTILE:.0%} наименее энергичных STFT-кадров.",
        f"- Поиск максимумов энергии: `Δt={DELTA_T:.1f} с`, `Δf={DELTA_F} Гц`.",
        "",
        "### Формулы",
        "",
        "Кратковременное преобразование Фурье:",
        "",
        "```text",
        "X(m, k) = sum_n x[n] * w[n - mR] * exp(-j 2*pi*k*n/N)",
        "```",
        "",
        "Спектрограмма мощности:",
        "",
        "```text",
        "P(m, k) = |X(m, k)|^2",
        "P_dB(m, k) = 10 * log10(P(m, k))",
        "```",
        "",
        "Оценка отношения сигнал/шум:",
        "",
        "```text",
        "SNR = 10 * log10(P_signal / P_noise)",
        "```",
        "",
        "Спектральное вычитание шума:",
        "",
        "```text",
        "|Y(f,t)| = max(|X(f,t)| - alpha * |W(f)|, floor * |X(f,t)|)",
        "phase(Y) = phase(X)",
        "```",
        "",
        "### 1. Подготовка сигнала",
        "",
        "- Исходная запись: `фортепиано.wav`.",
        "- Моно-дорожка для анализа: `src/noisy_input.wav`.",
        "- Восстановленная дорожка: `src/denoised_output.wav`.",
        "- Временные формы сигналов: `src/waveforms_compare.png`.",
        "",
        "![wave](src/waveforms_compare.png)",
        "",
        "### 2. Спектрограммы до и после подавления шума",
        "",
        "| До шумоподавления | После шумоподавления |",
        "|:-----------------:|:--------------------:|",
        "| ![sp_noisy](src/spectrogram_noisy.png) | ![sp_denoised](src/spectrogram_denoised.png) |",
        "",
        "### 3. Оценка уровня шума и качества подавления",
        "",
        "| Показатель | Значение |",
        "|:----------|---------:|",
        f"| Оценка RMS шума до (минимальное окно {NOISE_WINDOW_SECONDS:.1f} c) | `{metrics['noise_rms']:.6f}` |",
        f"| Оценка RMS шума после (минимальное окно {NOISE_WINDOW_SECONDS:.1f} c) | `{metrics['denoised_noise_rms']:.6f}` |",
        f"| SNR до шумоподавления | `{metrics['snr_before']:.3f}` dB |",
        f"| SNR после шумоподавления | `{metrics['snr_after']:.3f}` dB |",
        f"| Прирост SNR | `{metrics['snr_gain']:.3f}` dB |",
        "",
        "### 4. Моменты времени с наибольшей энергией",
        "",
        f"Поиск выполнен по окрестности `Δt={DELTA_T:.1f} с` и полосам `Δf={DELTA_F} Гц`.",
        "",
        "| № | Время, с | Полоса, Гц | Энергия |",
        "|--:|---------:|-----------:|--------:|",
    ]

    for rank, row in enumerate(top_rows, start=1):
        lines.append(f"| {rank} | {row['time']:.3f} | {row['f0']:.0f}..{row['f1']:.0f} | {row['energy']:.8e} |")

    lines.extend([
        "",
        "- Полная таблица: `src/high_energy_moments.csv`.",
        "",
        "### Вывод",
        "Для варианта 8 выполнен анализ шумовой аудиодорожки: построены спектрограммы STFT с окном Ханна, оценен уровень шума, применено шумоподавление методом спектрального вычитания и выполнено сравнение качества по SNR. Также найдены временные интервалы и частотные полосы с максимальной энергией при заданных шагах.",
        "",
    ])
    (BASE_DIR / "report_lab9.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs()
    sample_rate, noisy, channels, sample_width = read_wav_mono(INPUT_WAV)
    noisy = normalize_audio(noisy)
    denoised, noisy_spec, denoised_spec, noise_frames = spectral_subtract(noisy)

    write_wav_mono(SRC_DIR / "noisy_input.wav", sample_rate, noisy)
    write_wav_mono(SRC_DIR / "denoised_output.wav", sample_rate, denoised)

    draw_waveforms(sample_rate, noisy, denoised, SRC_DIR / "waveforms_compare.png")
    draw_spectrogram(noisy_spec, sample_rate, SRC_DIR / "spectrogram_noisy.png", "Noisy input STFT spectrogram")
    draw_spectrogram(denoised_spec, sample_rate, SRC_DIR / "spectrogram_denoised.png", "Denoised output STFT spectrogram")

    top_rows, all_rows = high_energy_moments(noisy_spec, sample_rate)
    save_high_energy_csv(all_rows, SRC_DIR / "high_energy_moments.csv")

    noise_rms = estimate_noise_rms(noisy, sample_rate)
    denoised_noise_rms = estimate_noise_rms(denoised, sample_rate)
    snr_before = snr_from_noise_rms(noisy, noise_rms)
    snr_after = snr_from_noise_rms(denoised, denoised_noise_rms)
    metrics = {
        "sample_rate": sample_rate,
        "duration": len(noisy) / sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "noise_rms": noise_rms,
        "denoised_noise_rms": denoised_noise_rms,
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_gain": snr_after - snr_before,
    }
    write_report(metrics, top_rows)


if __name__ == "__main__":
    main()

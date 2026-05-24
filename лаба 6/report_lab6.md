# Лабораторная работа №6
## Сегментация текста

### Вариант 8: алфавит Османья

### Исходные данные
- Фраза: `𐒓𐒛𐒒 𐒏𐒚 𐒃𐒗𐒋𐒐𐒖𐒔𐒖𐒕`
- Транслитерация: `waan ku jeclahay`
- Шрифт: `/System/Library/Fonts/Supplemental/NotoSansOsmanya-Regular.ttf`, размер `92`
- Межбуквенный интервал при подготовке строки: `8` px
- Размер монохромного изображения: `805x68`
- Количество найденных символов: `13`

### Формулы профилей

```text
H(y) = sum_x I_b(x, y)
V(x) = sum_y I_b(x, y)
```

Где `I_b(x,y)=1` для черного пикселя и `0` для белого.

### 1. Подготовка строки

#### 1.1 Монохромное изображение фразы
![input](src/input/phrase_mono.bmp)

### 2. Профили изображения

| Горизонтальный профиль | Вертикальный профиль |
|:----------------------:|:--------------------:|
| ![h](src/profiles/horizontal_profile.png) | ![v](src/profiles/vertical_profile.png) |

### 3. Сегментация символов по вертикальному профилю с прореживанием

#### 3.1 Обрамляющие прямоугольники
![boxes](src/segments/segmentation_boxes.png)

#### 3.2 Вырезанные сегменты

- Сегмент 1: `[segment_01]` -> ![s1](src/segments/segment_01.bmp)
- Сегмент 2: `[segment_02]` -> ![s2](src/segments/segment_02.bmp)
- Сегмент 3: `[segment_03]` -> ![s3](src/segments/segment_03.bmp)
- Сегмент 4: `[segment_04]` -> ![s4](src/segments/segment_04.bmp)
- Сегмент 5: `[segment_05]` -> ![s5](src/segments/segment_05.bmp)
- Сегмент 6: `[segment_06]` -> ![s6](src/segments/segment_06.bmp)
- Сегмент 7: `[segment_07]` -> ![s7](src/segments/segment_07.bmp)
- Сегмент 8: `[segment_08]` -> ![s8](src/segments/segment_08.bmp)
- Сегмент 9: `[segment_09]` -> ![s9](src/segments/segment_09.bmp)
- Сегмент 10: `[segment_10]` -> ![s10](src/segments/segment_10.bmp)
- Сегмент 11: `[segment_11]` -> ![s11](src/segments/segment_11.bmp)
- Сегмент 12: `[segment_12]` -> ![s12](src/segments/segment_12.bmp)
- Сегмент 13: `[segment_13]` -> ![s13](src/segments/segment_13.bmp)

#### 3.3 Массив координат прямоугольников

| idx | x0 | y0 | x1 | y1 | w | h |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 72 | 67 | 73 | 67 |
| 2 | 89 | 0 | 127 | 66 | 39 | 67 |
| 3 | 144 | 0 | 186 | 67 | 43 | 68 |
| 4 | 235 | 1 | 270 | 66 | 36 | 66 |
| 5 | 283 | 0 | 324 | 66 | 42 | 67 |
| 6 | 377 | 1 | 391 | 66 | 15 | 66 |
| 7 | 411 | 1 | 441 | 67 | 31 | 67 |
| 8 | 453 | 1 | 495 | 67 | 43 | 67 |
| 9 | 512 | 0 | 570 | 67 | 59 | 68 |
| 10 | 585 | 0 | 625 | 67 | 41 | 68 |
| 11 | 644 | 1 | 688 | 67 | 45 | 67 |
| 12 | 702 | 0 | 742 | 67 | 41 | 68 |
| 13 | 761 | 0 | 804 | 67 | 44 | 68 |

CSV с координатами (`;`-разделитель): `results/segments_boxes.csv`.

### 4. Профили символов выбранного алфавита

- Эталоны символов: `src/alphabet/templates/`.
- Профили X/Y: `src/alphabet/profiles/`.
- Построены для всех 30 букв алфавита Османья варианта 8.

Пример (первые 6 символов):

| Символ | Unicode | Эталон | Профиль X | Профиль Y |
|:------:|:-------:|:------:|:---------:|:---------:|
| 𐒀 | `U+10480` | ![t](src/alphabet/templates/sym_01_U10480.bmp) | ![px](src/alphabet/profiles/sym_01_U10480_profile_x.png) | ![py](src/alphabet/profiles/sym_01_U10480_profile_y.png) |
| 𐒁 | `U+10481` | ![t](src/alphabet/templates/sym_02_U10481.bmp) | ![px](src/alphabet/profiles/sym_02_U10481_profile_x.png) | ![py](src/alphabet/profiles/sym_02_U10481_profile_y.png) |
| 𐒂 | `U+10482` | ![t](src/alphabet/templates/sym_03_U10482.bmp) | ![px](src/alphabet/profiles/sym_03_U10482_profile_x.png) | ![py](src/alphabet/profiles/sym_03_U10482_profile_y.png) |
| 𐒃 | `U+10483` | ![t](src/alphabet/templates/sym_04_U10483.bmp) | ![px](src/alphabet/profiles/sym_04_U10483_profile_x.png) | ![py](src/alphabet/profiles/sym_04_U10483_profile_y.png) |
| 𐒄 | `U+10484` | ![t](src/alphabet/templates/sym_05_U10484.bmp) | ![px](src/alphabet/profiles/sym_05_U10484_profile_x.png) | ![py](src/alphabet/profiles/sym_05_U10484_profile_y.png) |
| 𐒅 | `U+10485` | ![t](src/alphabet/templates/sym_06_U10485.bmp) | ![px](src/alphabet/profiles/sym_06_U10485_profile_x.png) | ![py](src/alphabet/profiles/sym_06_U10485_profile_y.png) |

### Вывод
Реализованы расчет горизонтального и вертикального профилей, сегментация символов строки на основе вертикального профиля с прореживанием, сохранение массива координат прямоугольников и построение профилей всех букв алфавита Османья.

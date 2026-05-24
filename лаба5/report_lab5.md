# Лабораторная работа №5
## Выделение признаков символов

### Вариант 8: алфавит Османья

### Параметры генерации
- Шрифт: `/System/Library/Fonts/Supplemental/NotoSansOsmanya-Regular.ttf`
- Размер шрифта: `96`
- Количество символов: `30`
- Принцип: `1 символ = 1 файл`, белые поля обрезаны.

### Формулы признаков

```text
I_b(x, y) = 1 для черного пикселя и 0 для белого
M_q = sum I_b(x, y), где (x, y) принадлежит четверти q
w_q = M_q / S_q
M = sum I_b(x, y)
xc = (1/M) * sum x * I_b(x, y)
yc = (1/M) * sum y * I_b(x, y)
xc_norm = xc / (W - 1)
yc_norm = yc / (H - 1)
Ix = sum (y - yc)^2 * I_b(x, y)
Iy = sum (x - xc)^2 * I_b(x, y)
Ix_norm = Ix / (M * H^2)
Iy_norm = Iy / (M * W^2)
profile_X(x) = sum_y I_b(x, y)
profile_Y(y) = sum_x I_b(x, y)
```

### Результаты генерации символов и профилей

| № | Символ | Unicode | Название | Эталон | Профиль X | Профиль Y |
|:--:|:------:|:-------:|:---------|:------:|:---------:|:---------:|
| 1 | 𐒀 | `U+10480` | OSMANYA LETTER ALEF | ![t](src/templates/sym_01_U10480.bmp) | ![px](src/profiles/sym_01_U10480_profile_x.png) | ![py](src/profiles/sym_01_U10480_profile_y.png) |
| 2 | 𐒁 | `U+10481` | OSMANYA LETTER BA | ![t](src/templates/sym_02_U10481.bmp) | ![px](src/profiles/sym_02_U10481_profile_x.png) | ![py](src/profiles/sym_02_U10481_profile_y.png) |
| 3 | 𐒂 | `U+10482` | OSMANYA LETTER TA | ![t](src/templates/sym_03_U10482.bmp) | ![px](src/profiles/sym_03_U10482_profile_x.png) | ![py](src/profiles/sym_03_U10482_profile_y.png) |
| 4 | 𐒃 | `U+10483` | OSMANYA LETTER JA | ![t](src/templates/sym_04_U10483.bmp) | ![px](src/profiles/sym_04_U10483_profile_x.png) | ![py](src/profiles/sym_04_U10483_profile_y.png) |
| 5 | 𐒄 | `U+10484` | OSMANYA LETTER XA | ![t](src/templates/sym_05_U10484.bmp) | ![px](src/profiles/sym_05_U10484_profile_x.png) | ![py](src/profiles/sym_05_U10484_profile_y.png) |
| 6 | 𐒅 | `U+10485` | OSMANYA LETTER KHA | ![t](src/templates/sym_06_U10485.bmp) | ![px](src/profiles/sym_06_U10485_profile_x.png) | ![py](src/profiles/sym_06_U10485_profile_y.png) |
| 7 | 𐒆 | `U+10486` | OSMANYA LETTER DEEL | ![t](src/templates/sym_07_U10486.bmp) | ![px](src/profiles/sym_07_U10486_profile_x.png) | ![py](src/profiles/sym_07_U10486_profile_y.png) |
| 8 | 𐒇 | `U+10487` | OSMANYA LETTER RA | ![t](src/templates/sym_08_U10487.bmp) | ![px](src/profiles/sym_08_U10487_profile_x.png) | ![py](src/profiles/sym_08_U10487_profile_y.png) |
| 9 | 𐒈 | `U+10488` | OSMANYA LETTER SA | ![t](src/templates/sym_09_U10488.bmp) | ![px](src/profiles/sym_09_U10488_profile_x.png) | ![py](src/profiles/sym_09_U10488_profile_y.png) |
| 10 | 𐒉 | `U+10489` | OSMANYA LETTER SHIIN | ![t](src/templates/sym_10_U10489.bmp) | ![px](src/profiles/sym_10_U10489_profile_x.png) | ![py](src/profiles/sym_10_U10489_profile_y.png) |
| 11 | 𐒊 | `U+1048A` | OSMANYA LETTER DHA | ![t](src/templates/sym_11_U1048A.bmp) | ![px](src/profiles/sym_11_U1048A_profile_x.png) | ![py](src/profiles/sym_11_U1048A_profile_y.png) |
| 12 | 𐒋 | `U+1048B` | OSMANYA LETTER CAYN | ![t](src/templates/sym_12_U1048B.bmp) | ![px](src/profiles/sym_12_U1048B_profile_x.png) | ![py](src/profiles/sym_12_U1048B_profile_y.png) |
| 13 | 𐒌 | `U+1048C` | OSMANYA LETTER GA | ![t](src/templates/sym_13_U1048C.bmp) | ![px](src/profiles/sym_13_U1048C_profile_x.png) | ![py](src/profiles/sym_13_U1048C_profile_y.png) |
| 14 | 𐒍 | `U+1048D` | OSMANYA LETTER FA | ![t](src/templates/sym_14_U1048D.bmp) | ![px](src/profiles/sym_14_U1048D_profile_x.png) | ![py](src/profiles/sym_14_U1048D_profile_y.png) |
| 15 | 𐒎 | `U+1048E` | OSMANYA LETTER QAAF | ![t](src/templates/sym_15_U1048E.bmp) | ![px](src/profiles/sym_15_U1048E_profile_x.png) | ![py](src/profiles/sym_15_U1048E_profile_y.png) |
| 16 | 𐒏 | `U+1048F` | OSMANYA LETTER KAAF | ![t](src/templates/sym_16_U1048F.bmp) | ![px](src/profiles/sym_16_U1048F_profile_x.png) | ![py](src/profiles/sym_16_U1048F_profile_y.png) |
| 17 | 𐒐 | `U+10490` | OSMANYA LETTER LAAN | ![t](src/templates/sym_17_U10490.bmp) | ![px](src/profiles/sym_17_U10490_profile_x.png) | ![py](src/profiles/sym_17_U10490_profile_y.png) |
| 18 | 𐒑 | `U+10491` | OSMANYA LETTER MIIN | ![t](src/templates/sym_18_U10491.bmp) | ![px](src/profiles/sym_18_U10491_profile_x.png) | ![py](src/profiles/sym_18_U10491_profile_y.png) |
| 19 | 𐒒 | `U+10492` | OSMANYA LETTER NUUN | ![t](src/templates/sym_19_U10492.bmp) | ![px](src/profiles/sym_19_U10492_profile_x.png) | ![py](src/profiles/sym_19_U10492_profile_y.png) |
| 20 | 𐒓 | `U+10493` | OSMANYA LETTER WAW | ![t](src/templates/sym_20_U10493.bmp) | ![px](src/profiles/sym_20_U10493_profile_x.png) | ![py](src/profiles/sym_20_U10493_profile_y.png) |
| 21 | 𐒔 | `U+10494` | OSMANYA LETTER HA | ![t](src/templates/sym_21_U10494.bmp) | ![px](src/profiles/sym_21_U10494_profile_x.png) | ![py](src/profiles/sym_21_U10494_profile_y.png) |
| 22 | 𐒕 | `U+10495` | OSMANYA LETTER YA | ![t](src/templates/sym_22_U10495.bmp) | ![px](src/profiles/sym_22_U10495_profile_x.png) | ![py](src/profiles/sym_22_U10495_profile_y.png) |
| 23 | 𐒖 | `U+10496` | OSMANYA LETTER A | ![t](src/templates/sym_23_U10496.bmp) | ![px](src/profiles/sym_23_U10496_profile_x.png) | ![py](src/profiles/sym_23_U10496_profile_y.png) |
| 24 | 𐒗 | `U+10497` | OSMANYA LETTER E | ![t](src/templates/sym_24_U10497.bmp) | ![px](src/profiles/sym_24_U10497_profile_x.png) | ![py](src/profiles/sym_24_U10497_profile_y.png) |
| 25 | 𐒘 | `U+10498` | OSMANYA LETTER I | ![t](src/templates/sym_25_U10498.bmp) | ![px](src/profiles/sym_25_U10498_profile_x.png) | ![py](src/profiles/sym_25_U10498_profile_y.png) |
| 26 | 𐒙 | `U+10499` | OSMANYA LETTER O | ![t](src/templates/sym_26_U10499.bmp) | ![px](src/profiles/sym_26_U10499_profile_x.png) | ![py](src/profiles/sym_26_U10499_profile_y.png) |
| 27 | 𐒚 | `U+1049A` | OSMANYA LETTER U | ![t](src/templates/sym_27_U1049A.bmp) | ![px](src/profiles/sym_27_U1049A_profile_x.png) | ![py](src/profiles/sym_27_U1049A_profile_y.png) |
| 28 | 𐒛 | `U+1049B` | OSMANYA LETTER AA | ![t](src/templates/sym_28_U1049B.bmp) | ![px](src/profiles/sym_28_U1049B_profile_x.png) | ![py](src/profiles/sym_28_U1049B_profile_y.png) |
| 29 | 𐒜 | `U+1049C` | OSMANYA LETTER EE | ![t](src/templates/sym_29_U1049C.bmp) | ![px](src/profiles/sym_29_U1049C_profile_x.png) | ![py](src/profiles/sym_29_U1049C_profile_y.png) |
| 30 | 𐒝 | `U+1049D` | OSMANYA LETTER OO | ![t](src/templates/sym_30_U1049D.bmp) | ![px](src/profiles/sym_30_U1049D_profile_x.png) | ![py](src/profiles/sym_30_U1049D_profile_y.png) |

### Табличные признаки

- Скалярные признаки сохранены в CSV (`;`-разделитель): `results/features.csv`.
- Профили сохранены в PNG с целочисленными подписями осей.

### Вывод
Для алфавита Османья по варианту 8 сгенерированы эталонные изображения букв, рассчитаны массы четвертей, удельные веса, координаты центра тяжести, нормированные координаты, осевые моменты инерции, нормированные моменты и профили X/Y.

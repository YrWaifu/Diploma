import random
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

def build_mode_segments(total_seconds: float, min_s: int = 7, max_s: int = 10, start_value: int = 0):
    """
    Возвращает список сегментов (start_t, end_t, value),
    где value = 0/1, а длительности сегментов в секундах 7..10
    (последний сегмент может быть обрезан, чтобы точно попасть в total_seconds).
    """
    segments = []
    t = 0.0
    value = start_value

    while t < total_seconds - 1e-12:
        dur = random.randint(min_s, max_s)
        end_t = min(t + dur, total_seconds)
        segments.append((t, end_t, value))
        t = end_t
        value = 1 - value

    return segments

def mode_at_time(segments, t):
    # segments упорядочены по времени
    # ищем сегмент, куда попадает t
    for start_t, end_t, v in segments:
        if start_t <= t < end_t or abs(t - end_t) < 1e-12 and end_t == segments[-1][1]:
            return v
    # на всякий случай
    return segments[-1][2]

def generate_xlsx(
    filename: str = "generated_table.xlsx",
    rows: int = 4000,
    total_seconds: float = 40.0,
    series_count: int = 7,
    seed: int | None = 42
):
    if seed is not None:
        random.seed(seed)

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    # Заголовки
    headers = ["Time"] + [f"Series_{i:02d}" for i in range(1, series_count + 1)] + ["Режим"]
    ws.append(headers)

    # Сегменты режима 0/1 по 7-10 секунд
    segments = build_mode_segments(total_seconds, 7, 10, start_value=0)

    # Время: 0..40 на 4000 строк (включая 40)
    dt = total_seconds / (rows - 1)

    # Генерация данных
    for r in range(rows):
        t = r * dt

        # Пример значений как на скрине: что-то около [-2.5..2.5]
        series_values = [round(random.uniform(-2.5, 2.5), 6) for _ in range(series_count)]

        mode = mode_at_time(segments, t)

        ws.append([round(t, 6), *series_values, mode])

    # Чуть-чуть оформления, чтобы не плакать при просмотре
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    # Ширина столбцов
    ws.column_dimensions["A"].width = 10
    for i in range(2, 2 + series_count):
        ws.column_dimensions[get_column_letter(i)].width = 12
    ws.column_dimensions[get_column_letter(2 + series_count)].width = 10

    wb.save(filename)
    return filename, segments

if __name__ == "__main__":
    file_name, mode_segments = generate_xlsx(
        filename="generated_table.xlsx",
        rows=4000,
        total_seconds=40.0,
        series_count=7,
        seed=42
    )

    print("Saved:", file_name)
    print("Mode segments (start, end, value):")
    for s in mode_segments:
        print(s)

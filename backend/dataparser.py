import csv
import os

CSV_PATH = "dataset.csv"

BUFFER_SIZE = 200
EXPECTED_VALUES = 126 * BUFFER_SIZE


def ensure_csv_header():
    if os.path.exists(CSV_PATH):
        return

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(42 * BUFFER_SIZE):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)


def append_frame(values):
    """
    values: list of 126 floats
    """
    if len(values) != EXPECTED_VALUES:
        print(f"⚠️ Expected 126 values, got {len(values)}")
        return

    ensure_csv_header()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(values)

    print("✅ Frame written to dataset.csv")

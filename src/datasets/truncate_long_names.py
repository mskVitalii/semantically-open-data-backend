"""
Скрипт для обрезки длинных имён файлов и папок, чтобы пути
помещались в Windows MAX_PATH (260 символов) с запасом.

Запуск:
    python truncate_long_names.py          # dry-run (только показать что будет переименовано)
    python truncate_long_names.py --apply  # применить переименования
"""

import os
import hashlib
import argparse
from pathlib import Path

# --- Настройки ---
# Максимальная длина одного компонента пути (имя папки или файла).
# При структуре city(8) / folder(MAX) / file(MAX) даёт:
#   8 + 1 + 80 + 1 + 80 = 170 символов относительного пути,
#   что оставляет 90 символов на Windows-префикс (C:\Users\...\src\datasets\).
DEFAULT_MAX_COMPONENT_LENGTH = 80

# Длина хеш-суффикса для уникальности (напр. "_a1b2c3")
HASH_LENGTH = 6


def short_hash(name: str) -> str:
    """Короткий хеш от оригинального имени для уникальности."""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def truncate_filename(name: str, max_len: int) -> str:
    """Обрезает имя файла, сохраняя расширение и добавляя хеш."""
    if len(name) <= max_len:
        return name

    stem, ext = os.path.splitext(name)
    # Для двойных расширений типа ".csv.csv"
    if ext and stem.endswith(ext):
        stem = stem[: -len(ext)]
        ext = ext + ext

    h = "_" + short_hash(name)
    available = max_len - len(ext) - len(h)
    if available < 10:
        available = 10

    truncated_stem = stem[:available].rstrip("_").rstrip("-").rstrip(".")
    return truncated_stem + h + ext


def truncate_dirname(name: str, max_len: int) -> str:
    """Обрезает имя директории, добавляя хеш."""
    if len(name) <= max_len:
        return name

    h = "_" + short_hash(name)
    available = max_len - len(h)
    if available < 10:
        available = 10

    truncated = name[:available].rstrip("_").rstrip("-").rstrip(".")
    return truncated + h


def collect_renames(base_dir: str, max_len: int) -> list[tuple[str, str]]:
    """
    Обходит дерево снизу вверх и собирает список переименований.
    Возвращает список (old_path, new_path) в порядке, безопасном для применения.
    """
    renames = []

    # Сначала файлы (в любом порядке — они не влияют друг на друга)
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for f in files:
            new_name = truncate_filename(f, max_len)
            if new_name != f:
                old_path = os.path.join(root, f)
                new_path = os.path.join(root, new_name)
                renames.append((old_path, new_path))

    # Затем директории снизу вверх (topdown=False гарантирует правильный порядок)
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for d in dirs:
            new_name = truncate_dirname(d, max_len)
            if new_name != d:
                old_path = os.path.join(root, d)
                new_path = os.path.join(root, new_name)
                renames.append((old_path, new_path))

    return renames


def main():
    parser = argparse.ArgumentParser(
        description="Обрезка длинных имён файлов/папок для совместимости с Windows MAX_PATH"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить переименования (без этого флага — только dry-run)",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_COMPONENT_LENGTH,
        help=f"Макс. длина компонента пути (по умолчанию {DEFAULT_MAX_COMPONENT_LENGTH})",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    renames = collect_renames(str(base_dir), args.max_len)

    if not renames:
        print("Все имена файлов и папок уже в пределах допустимой длины.")
        return

    print(f"{'DRY RUN — ' if not args.apply else ''}Найдено переименований: {len(renames)}\n")

    dirs_count = 0
    files_count = 0

    for old_path, new_path in renames:
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        is_dir = os.path.isdir(old_path)
        kind = "DIR " if is_dir else "FILE"

        if is_dir:
            dirs_count += 1
        else:
            files_count += 1

        print(f"  [{kind}] {len(old_name):3d} -> {len(new_name):3d} chars")
        print(f"    OLD: {old_name}")
        print(f"    NEW: {new_name}")
        print()

    print(f"Итого: {dirs_count} папок, {files_count} файлов")

    if args.apply:
        conflicts = []
        for old_path, new_path in renames:
            if os.path.exists(new_path):
                conflicts.append((old_path, new_path))

        if conflicts:
            print(f"\nКОНФЛИКТЫ ({len(conflicts)})! Следующие целевые пути уже существуют:")
            for old_p, new_p in conflicts:
                print(f"  {old_p} -> {new_p}")
            print("Переименование отменено. Разрешите конфликты вручную.")
            return

        applied = 0
        for old_path, new_path in renames:
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                applied += 1
            else:
                print(f"  SKIP (уже перемещён): {old_path}")

        print(f"\nУспешно переименовано: {applied}")
    else:
        print("Это dry-run. Добавьте --apply для применения.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Сохранение и загрузка проекта (workspace).
Формат: JSON с расширением .odproj. В файле только состояние: пути к файлам,
какие ряды добавлены на графики, геометрия стиков и вид по X. Данные при открытии
подгружаются заново из исходных файлов.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


PROJECT_VERSION = 1
PROJECT_EXT = ".odproj"


def _path_to_str(p: Path) -> str:
    """Единообразное строковое представление пути для JSON."""
    return str(p.resolve())


def save_project(state: dict[str, Any], path: Path) -> None:
    """
    Сохраняет состояние проекта в JSON-файл.

    state: словарь с ключами version, sources, plots, view (см. структуру ниже).
    path: путь к файлу (обычно с расширением .odproj).
    """
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXT:
        path = path.with_suffix(PROJECT_EXT)

    out = {
        "version": state.get("version", PROJECT_VERSION),
        "sources": state.get("sources", []),
        "plots": state.get("plots", []),
        "view": state.get("view", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_project(path: Path) -> dict[str, Any]:
    """
    Загружает состояние проекта из JSON-файла.

    Возвращает словарь: version, sources, plots, view.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    version = data.get("version", 1)
    if version > PROJECT_VERSION:
        raise ValueError(f"Версия проекта {version} новее поддерживаемой {PROJECT_VERSION}")

    return {
        "version": version,
        "sources": data.get("sources", []),
        "plots": data.get("plots", []),
        "view": data.get("view", {}),
    }


def is_project_file(path: Path) -> bool:
    """Проверяет, что путь указывает на файл проекта."""
    return path.suffix.lower() == PROJECT_EXT

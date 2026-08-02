"""
Local + Live USDA Tiered Nutrition Lookup for Detected Food Classes.

Reads local CSV tables (INDB/Anuvaad & local USDA CSV) and falls back to
the live USDA FoodData Central REST API for rich macronutrient & micronutrient data
(vitamins, minerals, dietary fiber, serving size).
"""

import csv
import logging
from pathlib import Path
from difflib import get_close_matches
from typing import Optional, Dict, Any, List

from config import NUTRITION_DB_PATHS, ENABLE_USDA_LIVE_API, USDA_API_KEY
from usda_client import USDAFoodDataClient

logger = logging.getLogger(__name__)

# Map actual CSV column names -> canonical keys used in output.
COLUMN_MAP = {
    "calories_kcal": ["calories_kcal", "energy_kcal", "calories"],
    "protein_g": ["protein_g", "protein"],
    "carbs_g": ["carbs_g", "carbohydrate_g", "carbs"],
    "fat_g": ["fat_g", "total_fat_g", "fat"],
    "fiber_g": ["fiber_g", "dietary_fiber_g", "fiber"],
    "sugars_g": ["sugars_g", "total_sugars_g", "sugars"],
    "sodium_mg": ["sodium_mg", "sodium"],
    "potassium_mg": ["potassium_mg", "potassium"],
    "calcium_mg": ["calcium_mg", "calcium"],
    "iron_mg": ["iron_mg", "iron"],
    "vitamin_a_iu": ["vitamin_a_iu", "vitamin_a"],
    "vitamin_c_mg": ["vitamin_c_mg", "vitamin_c"],
    "vitamin_d_iu": ["vitamin_d_iu", "vitamin_d"],
}

FUZZY_CUTOFF = 0.75


def _normalize(name: str) -> str:
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def _extract_value(row: dict, candidates: list) -> Optional[float]:
    for col in candidates:
        if col in row and row[col] not in (None, ""):
            try:
                return round(float(row[col]), 2)
            except (TypeError, ValueError):
                continue
    return None


class NutritionLookup:
    """
    Tiered nutrition resolution:
      1. Local CSV Exact Match
      2. Local CSV Fuzzy Match
      3. Live USDA FoodData Central REST API Search
    """

    def __init__(self, db_paths: Optional[List[Path]] = None):
        self.db_paths = [Path(p) for p in (db_paths or NUTRITION_DB_PATHS)]
        self.tables = []
        for path in self.db_paths:
            table = self._load_table(path)
            self.tables.append((path.stem, table))
            if not path.is_file():
                logger.warning("[nutrition] Local DB '%s' not found.", path)
            else:
                logger.info("[nutrition] Loaded %d rows from '%s'", len(table), path)

        self.usda_client = USDAFoodDataClient(api_key=USDA_API_KEY) if ENABLE_USDA_LIVE_API else None

    def _load_table(self, path: Path) -> dict:
        table = {}
        if not path.is_file():
            return table
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            name_col = "name"
            if reader.fieldnames:
                if "name" in reader.fieldnames:
                    name_col = "name"
                else:
                    name_col = reader.fieldnames[0]
            for row in reader:
                raw_name = row.get(name_col, "")
                if not raw_name:
                    continue
                table[_normalize(raw_name)] = row
        return table

    def _row_to_profile(self, row: dict, source: str, matched_name: str, match_type: str) -> dict:
        profile = {k: _extract_value(row, cands) for k, cands in COLUMN_MAP.items()}
        profile["source"] = source
        profile["matched_name"] = matched_name
        profile["match_type"] = match_type
        profile["serving_size"] = _extract_value(row, ["serving_size", "serving_size_g"]) or 100.0
        profile["serving_size_unit"] = row.get("serving_size_unit", "g")
        return profile

    def lookup(self, class_name: str) -> Optional[dict]:
        norm = _normalize(class_name)

        # Tier 1: Local CSV exact match
        for source, table in self.tables:
            if norm in table:
                profile = self._row_to_profile(table[norm], source, norm, "exact")
                return self._enrich_with_usda_if_needed(norm, profile)

        # Tier 2: Local CSV fuzzy match
        for source, table in self.tables:
            if not table:
                continue
            close = get_close_matches(norm, table.keys(), n=1, cutoff=FUZZY_CUTOFF)
            if close:
                logger.info("[nutrition] Local fuzzy match '%s' -> '%s' in '%s'", norm, close[0], source)
                profile = self._row_to_profile(table[close[0]], source, close[0], "fuzzy")
                return self._enrich_with_usda_if_needed(norm, profile)

        # Tier 3: Live USDA FoodData Central REST API
        if self.usda_client:
            logger.info("[nutrition] Querying live USDA REST API for '%s'...", norm)
            usda_profile = self.usda_client.search_food(norm)
            if usda_profile:
                return usda_profile

        logger.info("[nutrition] NO MATCH for '%s' in local DBs or USDA REST API", norm)
        return None

    def _enrich_with_usda_if_needed(self, norm: str, profile: dict) -> dict:
        """
        If local match lacks key micronutrients/vitamins and USDA API is enabled,
        attempt live query to enrich missing fields.
        """
        missing_vitamins = (profile.get("vitamin_a_iu") is None and profile.get("vitamin_c_mg") is None)
        if self.usda_client and missing_vitamins:
            usda_profile = self.usda_client.search_food(norm)
            if usda_profile:
                for k, v in usda_profile.items():
                    if profile.get(k) is None and v is not None:
                        profile[k] = v
                profile["enriched_via"] = "usda_live_api"
        return profile

    def lookup_many(self, class_names: list) -> dict:
        return {name: self.lookup(name) for name in class_names}
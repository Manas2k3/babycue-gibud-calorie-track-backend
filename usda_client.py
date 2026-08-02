"""
USDA FoodData Central REST API Integration Client.

Queries the live USDA FoodData Central API (https://api.nal.usda.gov/fdc/v1/foods/search)
to retrieve rich nutritional information including calories, macronutrients,
micronutrients (vitamins, minerals), dietary fiber, sugars, and serving size details.
"""

import logging
import os
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)

# Standard USDA Nutrient IDs mapped to canonical metric names
NUTRIENT_ID_MAP = {
    1008: "calories_kcal",     # Energy (kcal)
    1003: "protein_g",         # Protein (g)
    1005: "carbs_g",           # Carbohydrate, by difference (g)
    1004: "fat_g",             # Total lipid (fat) (g)
    1079: "fiber_g",           # Fiber, total dietary (g)
    2000: "sugars_g",          # Total Sugars (g)
    1093: "sodium_mg",         # Sodium, Na (mg)
    1092: "potassium_mg",      # Potassium, K (mg)
    1087: "calcium_mg",        # Calcium, Ca (mg)
    1089: "iron_mg",           # Iron, Fe (mg)
    1106: "vitamin_a_iu",      # Vitamin A, IU
    1162: "vitamin_c_mg",      # Vitamin C, total ascorbic acid (mg)
    1114: "vitamin_d_iu",      # Vitamin D (D2 + D3) (IU)
}

# String name fallbacks if nutrientId is missing or altered
NUTRIENT_NAME_KEYWORDS = {
    "energy": "calories_kcal",
    "protein": "protein_g",
    "carbohydrate": "carbs_g",
    "total lipid": "fat_g",
    "fiber": "fiber_g",
    "sugars": "sugars_g",
    "sodium": "sodium_mg",
    "potassium": "potassium_mg",
    "calcium": "calcium_mg",
    "iron": "iron_mg",
    "vitamin a": "vitamin_a_iu",
    "vitamin c": "vitamin_c_mg",
    "vitamin d": "vitamin_d_iu",
}


class USDAFoodDataClient:
    """
    Client for USDA FoodData Central REST API with local request caching.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("USDA_API_KEY", "DEMO_KEY")
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self._cache: Dict[str, Dict[str, Any]] = {}

    def search_food(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Searches USDA FoodData Central for a food item and returns a normalized nutrient profile.
        """
        query_clean = query.strip().lower()
        if query_clean in self._cache:
            logger.info("[USDA API] Returning cached result for '%s'", query_clean)
            return self._cache[query_clean]

        url = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query_clean,
            "pageSize": 1,
            "dataType": ["Survey (FNDDS)", "Foundation", "Branded"],
        }

        try:
            response = requests.get(url, params=params, timeout=5.0)
            if response.status_code != 200:
                logger.warning(
                    "[USDA API] Query for '%s' returned HTTP %d: %s",
                    query_clean, response.status_code, response.text[:200]
                )
                return None

            data = response.json()
            foods = data.get("foods", [])
            if not foods:
                logger.info("[USDA API] No food items found for query '%s'", query_clean)
                return None

            food = foods[0]
            profile = self._parse_food_item(food, query_clean)
            self._cache[query_clean] = profile
            return profile

        except Exception as e:
            logger.error("[USDA API] Exception during request for '%s': %s", query_clean, str(e))
            return None

    def _parse_food_item(self, food: Dict[str, Any], query: str) -> Dict[str, Any]:
        nutrients_dict: Dict[str, Optional[float]] = {
            "calories_kcal": None,
            "protein_g": None,
            "carbs_g": None,
            "fat_g": None,
            "fiber_g": None,
            "sugars_g": None,
            "sodium_mg": None,
            "potassium_mg": None,
            "calcium_mg": None,
            "iron_mg": None,
            "vitamin_a_iu": None,
            "vitamin_c_mg": None,
            "vitamin_d_iu": None,
        }

        for nutrient in food.get("foodNutrients", []):
            n_id = nutrient.get("nutrientId")
            n_name = nutrient.get("nutrientName", "").lower()
            val = nutrient.get("value")

            if val is None:
                continue

            target_key = NUTRIENT_ID_MAP.get(n_id)
            if not target_key:
                for kw, mapped_key in NUTRIENT_NAME_KEYWORDS.items():
                    if kw in n_name:
                        target_key = mapped_key
                        break

            if target_key and nutrients_dict.get(target_key) is None:
                nutrients_dict[target_key] = round(float(val), 2)

        serving_size = food.get("servingSize")
        serving_unit = food.get("servingSizeUnit", "g")

        return {
            "source": "usda_live_api",
            "matched_name": food.get("description", query),
            "match_type": "api_search",
            "fdc_id": food.get("fdcId"),
            "serving_size": serving_size or 100.0,
            "serving_size_unit": serving_unit or "g",
            **nutrients_dict,
        }

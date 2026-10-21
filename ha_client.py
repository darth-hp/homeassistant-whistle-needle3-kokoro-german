#!/usr/bin/env python3
import os
import json
import time
import logging
import urllib.request
from rapidfuzz import process, fuzz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
_LOGGER = logging.getLogger("ha_client")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")
RESPONSES_FILE = os.path.join(SCRIPT_DIR, "ha_client_responses.json")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "ha_client_config.json")

def load_env(env_path: str):
    """Load key-value pairs from a .env file into os.environ."""
    if not os.path.isfile(env_path):
        _LOGGER.warning(f"No .env file found at '{env_path}'.")
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                key, val = clean_line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip("\"'")

def load_json_file(file_path: str, default: dict) -> dict:
    """Safely load JSON configuration with fallback default."""
    if not os.path.isfile(file_path):
        _LOGGER.warning(f"File '{file_path}' not found. Using defaults.")
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        _LOGGER.error(f"Error loading '{file_path}': {err}. Using defaults.")
        return default

load_env(ENV_FILE)
HA_URL = os.environ.get("HA_URL", "").rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

RESPONSES = load_json_file(RESPONSES_FILE, {})
CLIENT_CONFIG = load_json_file(CONFIG_FILE, {
    "matching": {
        "strict_threshold": 82,
        "scorer": "token_set_ratio",
        "generic_nouns": ["rollo", "rolladen", "licht", "lampe", "led", "blind", "shutter", "light", "climate", "heizung"],
        "all_trigger_words": ["alle", "all", "jeder", "every"],
        "group_entity_patterns": ["alle_", "all_"],
        "group_name_patterns": ["alle ", "all "]
    }
})

def get_scorer_function(name: str):
    """Map string config to RapidFuzz scorer function."""
    scorers = {
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
        "ratio": fuzz.ratio,
        "partial_ratio": fuzz.partial_ratio,
        "WRatio": fuzz.WRatio
    }
    return scorers.get(name, fuzz.token_set_ratio)

def get_template(key: str, lang: str = "de") -> str:
    """Get language-specific response template with fallback to English."""
    lang_dict = RESPONSES.get(lang, RESPONSES.get("en", {}))
    return lang_dict.get(key, "")

def fetch_all_entities_from_ha(retries: int = 5, delay: int = 2) -> list[dict]:
    """Fetch all entity states and friendly names from Home Assistant with connection retries."""
    if not HA_URL or not HA_TOKEN:
        _LOGGER.error("HA_URL or HA_TOKEN is missing.")
        return []

    url = f"{HA_URL}/api/states"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    device_list = []
                    for item in data:
                        entity_id = item.get("entity_id", "")
                        domain = entity_id.split(".")[0]
                        if domain in ["light", "cover", "climate", "switch"]:
                            friendly_name = item.get("attributes", {}).get("friendly_name", "")
                            device_list.append({
                                "entity_id": entity_id,
                                "domain": domain,
                                "name": friendly_name,
                                "search_key": f"{friendly_name} {entity_id.split('.')[1]}".lower()
                            })
                    _LOGGER.info(f"Loaded {len(device_list)} controllable devices from HA.")
                    return device_list
        except Exception as err:
            _LOGGER.warning(f"Attempt {attempt}/{retries} to fetch entities failed: {err}")
            if attempt < retries:
                time.sleep(delay)

    _LOGGER.error("All attempts to fetch entities from Home Assistant failed.")
    return []

ALL_DEVICES = fetch_all_entities_from_ha()

def find_best_entity(spoken_device: str, target_domain: str = None) -> dict:
    """Find matching entity using configured confidence threshold and scorer."""
    global ALL_DEVICES
    if not ALL_DEVICES:
        ALL_DEVICES = fetch_all_entities_from_ha()

    candidates = ALL_DEVICES
    if target_domain:
        candidates = [d for d in ALL_DEVICES if d["domain"] == target_domain]

    if not candidates:
        return None

    cleaned_spoken = spoken_device.strip().lower()
    matching_cfg = CLIENT_CONFIG.get("matching", {})

    # Guard: reject generic solitary nouns without identifier
    generic_nouns = matching_cfg.get("generic_nouns", [])
    if cleaned_spoken in generic_nouns:
        _LOGGER.warning(f"Rejected generic device name '{spoken_device}' without specific target.")
        return None

    # Guard: filter out group entities unless explicitly requested
    all_triggers = matching_cfg.get("all_trigger_words", ["alle", "all"])
    if not any(word in cleaned_spoken for word in all_triggers):
        group_entities = matching_cfg.get("group_entity_patterns", ["alle_", "all_"])
        group_names = matching_cfg.get("group_name_patterns", ["alle ", "all "])
        candidates = [
            c for c in candidates
            if not any(tok in c["entity_id"].lower() for tok in group_entities)
            and not any(tok in c["name"].lower() for tok in group_names)
        ]

    if not candidates:
        return None

    choices = [c["search_key"] for c in candidates]
    scorer_func = get_scorer_function(matching_cfg.get("scorer", "token_set_ratio"))
    threshold = matching_cfg.get("strict_threshold", 82)

    match = process.extractOne(cleaned_spoken, choices, scorer=scorer_func)

    if match and match[1] >= threshold:
        matched_candidate = candidates[match[2]]
        _LOGGER.info(
            f"Strict match accepted: '{spoken_device}' -> '{matched_candidate['name']}' "
            f"({matched_candidate['entity_id']}) with score {match[1]} (threshold: {threshold})"
        )
        return matched_candidate

    best_name = candidates[match[2]]["name"] if match else "None"
    best_score = match[1] if match else 0
    _LOGGER.warning(
        f"Confidence too low for '{spoken_device}'. Best candidate was '{best_name}' "
        f"with score {best_score} < {threshold}. Action aborted."
    )
    return None

def call_ha_service(domain: str, service: str, payload: dict) -> bool:
    """Execute a REST POST call against the Home Assistant service API."""
    if not HA_URL or not HA_TOKEN:
        _LOGGER.error("HA_URL or HA_TOKEN missing for service call.")
        return False

    url = f"{HA_URL}/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception as err:
        _LOGGER.error(f"Error calling service {domain}.{service}: {err}")
        return False

def dispatch_device_action(device_name: str, action_type: str, params: dict, lang: str = "de") -> str:
    """Dispatch parsed action directly to matched device, resolving percentage for light and cover."""
    spoken_lower = device_name.lower()

    if any(k in spoken_lower for k in ["rollo", "rolladen", "jalousie", "shutter", "blind", "curtain"]):
        domain_filter = "cover"
    elif any(k in spoken_lower for k in ["licht", "lampe", "led", "spot", "leuchte", "light", "lamp"]):
        domain_filter = "light"
    elif any(k in spoken_lower for k in ["klima", "heizung", "thermostat", "temperatur", "climate", "heat", "ac"]):
        domain_filter = "climate"
    else:
        domain_filter = "light" if action_type in ["light", "switch"] else action_type

    device = find_best_entity(device_name, domain_filter)
    if not device:
        tmpl = get_template("device_not_found", lang)
        return tmpl.format(device=device_name) if tmpl else f"Device '{device_name}' not found."

    entity_id = device["entity_id"]
    domain = device["domain"]

    # Unified percentage extraction
    percentage = params.get("percentage")
    if percentage is None:
        percentage = params.get("brightness") if params.get("brightness") is not None else params.get("position")

    # 1. LIGHT / SWITCH
    if domain in ["light", "switch"]:
        state = params.get("state", "on")
        service = "turn_on" if state == "on" else "turn_off"
        payload = {"entity_id": entity_id}

        if state == "on" and percentage is not None and domain == "light":
            payload["brightness_pct"] = int(percentage)

        _LOGGER.info(f"Setting {entity_id} via {domain}.{service} with payload: {payload}")
        call_ha_service(domain, service, payload)

        if percentage is not None and state == "on":
            tmpl = get_template("light_brightness", lang)
            return tmpl.format(name=device["name"], brightness=percentage)

        state_str = get_template("light_on" if state == "on" else "light_off", lang)
        tmpl = get_template("light_state", lang)
        return tmpl.format(name=device["name"], state=state_str)

    # 2. COVER
    elif domain == "cover":
        action = params.get("action", "open")
        payload = {"entity_id": entity_id}

        if percentage is not None:
            service = "set_cover_position"
            payload["position"] = int(percentage)
            call_ha_service("cover", service, payload)
            tmpl = get_template("cover_position", lang)
            return tmpl.format(name=device["name"], position=percentage)
        else:
            service = "open_cover" if action == "open" else ("close_cover" if action == "close" else "stop_cover")
            call_ha_service("cover", service, payload)
            if action == "open":
                tmpl = get_template("cover_opened", lang)
            elif action == "close":
                tmpl = get_template("cover_closed", lang)
            else:
                tmpl = get_template("cover_stopped", lang)
            return tmpl.format(name=device["name"])

    # 3. CLIMATE
    elif domain == "climate":
        temp = params.get("temperature")
        mode = params.get("hvac_mode")
        if temp is not None:
            call_ha_service("climate", "set_temperature", {"entity_id": entity_id, "temperature": float(temp)})
            tmpl = get_template("climate_temp", lang)
            return tmpl.format(name=device["name"], temp=temp)
        if mode:
            call_ha_service("climate", "set_hvac_mode", {"entity_id": entity_id, "hvac_mode": mode})
            tmpl = get_template("climate_mode", lang)
            return tmpl.format(name=device["name"], mode=mode)

    return get_template("action_completed", lang)

from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS
import langdetect
import json
import urllib.parse
import requests
import cachetools.func
import re

app = Flask(__name__)
CORS(app)

# --- API Keys ---
WEATHER_API_KEY = "ddf4e7231d180c2f839841ed477d9314"
GEMINI_API_KEY = "AIzaSyBknxTViPKyADxmeZpdnRV4J4PyrgFWeFM"

# --- Gemini API Setup ---
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="tunedModels/moment-creator-vvvnjduxt6dl",
    generation_config={
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)
chat_session = model.start_chat(history=[])

# --- Load local data ---
with open("tamilnadu_data.json", "r", encoding="utf-8") as f:
    local_data = json.load(f)

# --- In-memory session ---
previous_question = ""

# --- Tourism-related keywords ---
TOURISM_KEYWORDS = [
    "kodaikanal", "ooty", "yercaud", "salem", "madurai", "rameswaram", "kanyakumari", "coimbatore",
    "thanjavur", "chennai", "tiruvannamalai", "vellore", "theni", "pollachi", "temple", "fort", "falls",
    "beach", "hill", "museum", "tamil nadu", "hiri", "heritage", "culture", "tourism", "travel", "shop",
    "market", "spot", "attraction", "monument", "palace", "lake", "park"
]
ONLINE_KEYWORDS = ["hotel", "stay", "restaurant", "shop", "travel", "bus", "train", "taxi", "cab", "flight"]

# --- Helper Functions ---
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "en"

def is_tamilnadu_tourism_query(question):
    return any(k in question.lower() for k in TOURISM_KEYWORDS)

def needs_online_search(question):
    return any(k in question.lower() for k in ONLINE_KEYWORDS)

@cachetools.func.ttl_cache(maxsize=100, ttl=86400)  # Cache for 24 hours
def get_unsplash_image(location_name):
    """Fetch an image for a given location from Unsplash or return a placeholder."""
    if not any(keyword in location_name.lower() for keyword in TOURISM_KEYWORDS):
        return None  # Skip image fetch for non-tourism-related names
    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": f"{location_name} Tamil Nadu",
        "client_id": "S7v7DH6VEMQwDJDpQDwpINrlILxme2zsi4jia94dAzg",
        "per_page": 1,
        "orientation": "landscape"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
        print(f"No images found for {location_name}")
    except Exception as e:
        print(f"Image fetch error for {location_name}: {e}")
    return None

@cachetools.func.ttl_cache(maxsize=100, ttl=3600)  # Cache for 1 hour
def get_weather_data(location_name):
    """Fetch weather data for a given location from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": f"{location_name},IN",
        "appid": WEATHER_API_KEY,
        "units": "metric"  # Celsius for temperature
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"].capitalize(),
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    except Exception as e:
        print(f"Weather fetch error for {location_name}: {e}")
        return None

@cachetools.func.ttl_cache(maxsize=100, ttl=86400)  # Cache for 24 hours
def enrich_description(location_name):
    """Fetch additional details for a location using Gemini AI."""
    prompt = f"""You are a Tamil Nadu tourism expert. Provide a detailed description of {location_name} in Tamil Nadu, including historical or cultural significance, visiting tips, or hidden gems. Keep the response concise and factual, under 200 words."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error enriching description for {location_name}: {e}")
        return None

def search_local_data(question):
    """Search local data for places and include enriched descriptions, images, and weather."""
    results = []
    question_lower = question.lower()
    for place, info in local_data.items():
        if place.lower() in question_lower or any(word in question_lower for word in place.lower().split()):
            map_link = info.get("map") or f"https://www.google.com/maps/search/{urllib.parse.quote_plus(place)}"
            image_url = info.get("image") or get_unsplash_image(place)
            weather_data = get_weather_data(place)
            description = info.get("description", "No description available.")
            # Enrich description with internet data
            enriched_desc = enrich_description(place)
            if enriched_desc:
                description = enriched_desc
            result = {
                "name": place.title(),
                "description": description,
                "map_link": map_link,
                "image_url": image_url
            }
            if weather_data:
                result["weather"] = weather_data
            results.append(result)
    return results

def extract_locations_from_text(text, question):
    """Extract potential location names from the AI response, relevant to the question."""
    locations = set()
    question_lower = question.lower()
    # Prioritize locations mentioned in the question
    for place in local_data.keys():
        if place.lower() in question_lower or any(word in question_lower for word in place.lower().split()):
            locations.add(place.title())
    # Check AI response for additional tourism-related locations
    for place in local_data.keys():
        if re.search(r'\b' + re.escape(place) + r'\b', text, re.IGNORECASE):
            locations.add(place.title())
    # Heuristic for additional tourism-related locations in the response
    words = text.split()
    for word in words:
        if (word[0].isupper() and len(word) > 3 and word.lower() not in ONLINE_KEYWORDS and
                any(keyword in word.lower() for keyword in TOURISM_KEYWORDS) and
                word.lower() in question_lower):  # Ensure relevance to question
            locations.add(word)
    return list(locations)

def generate_gemini_prompt(question, previous_context=None):
    if previous_context:
        return f"""You are a Tamil Nadu tourism expert assisting travelers with detailed, engaging, and factual information.

The user asked previously: "{previous_context}"
Now they are asking: "{question}"

Please respond as a continuation, giving deeper or contextually relevant travel advice related to Tamil Nadu."""
    else:
        return f"""You are a Tamil Nadu tourism expert assisting travelers with detailed, engaging, and factual information.

Please answer the following query in a conversational tone and include any relevant tourist attractions, historical or cultural significance, travel tips, or hidden gems.

Query: "{question}"

Respond as if you are guiding a tourist in Tamil Nadu."""

def is_followup_question(current_q, previous_q):
    if not previous_q:
        return False
    current_set = set(current_q.lower().split())
    previous_set = set(previous_q.lower().split())
    common_words = current_set.intersection(previous_set)
    return len(common_words) < len(current_set) / 2

# --- Main Route ---
@app.route("/ask", methods=["POST"])
def ask():
    global previous_question

    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not is_tamilnadu_tourism_query(question):
        return jsonify({
            "status": "rejected",
            "message": "Only Tamil Nadu tourism queries are accepted."
        }), 200

    try:
        lang = detect_language(question)
        followup = is_followup_question(question, previous_question)
        prompt = generate_gemini_prompt(question, previous_question if followup else None)
        previous_question = question

        # Store the user question in session history
        chat_session.history.append({
            "role": "user",
            "parts": [prompt]
        })

        locations = search_local_data(question)
        answer = ""
        
        if needs_online_search(question):
            response = chat_session.send_message(prompt)
            answer = response.text.strip()
            # Store the AI response in session history
            chat_session.history.append({
                "role": "model",
                "parts": [answer]
            })
        else:
            if locations:
                answer = f"Here's what I found about {question} from local Tamil Nadu tourism data:\n\n"
                for loc in locations:
                    answer += f"**{loc['name']}**\n"
                    answer += f"{loc['description']}\n"
                    answer += f"[View on Google Maps]({loc['map_link']})\n"
                    if loc.get("weather"):
                        answer += f"Weather: {loc['weather']['description']}, {loc['weather']['temperature']}°C, "
                        answer += f"Humidity: {loc['weather']['humidity']}%, Wind: {loc['weather']['wind_speed']} m/s\n\n"
            else:
                response = chat_session.send_message(prompt)
                answer = response.text.strip()
                # Store the AI response in session history
                chat_session.history.append({
                    "role": "model",
                    "parts": [answer]
                })

        # Analyze AI response for additional locations, ensuring relevance to question
        extracted_locations = extract_locations_from_text(answer, question)
        for place in extracted_locations:
            if not any(loc["name"] == place for loc in locations):
                map_link = f"https://www.google.com/maps/search/{urllib.parse.quote_plus(place)}"
                image_url = get_unsplash_image(place)
                weather_data = get_weather_data(place)
                description = local_data.get(place.lower(), {}).get("description", "No description available.")
                # Enrich description for non-local data locations
                if description == "No description available.":
                    enriched_desc = enrich_description(place)
                    if enriched_desc:
                        description = enriched_desc
                result = {
                    "name": place,
                    "description": description,
                    "map_link": map_link
                }
                if image_url:  # Only include image_url if valid
                    result["image_url"] = image_url
                if weather_data:
                    result["weather"] = weather_data
                locations.append(result)

        # Ensure images and weather are included for valid locations only
        for loc in locations:
            if "image_url" not in loc or not loc["image_url"]:
                loc["image_url"] = get_unsplash_image(loc["name"]) or "No image available"
            if "weather" not in loc:
                weather_data = get_weather_data(loc["name"])
                if weather_data:
                    loc["weather"] = weather_data

        return jsonify({
            "status": "success",
            "message": "Response generated",
            "language": lang,
            "data": {
                "question": question,
                "answer": answer,
                "locations": locations
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run Server ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  
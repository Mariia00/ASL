# gemini_api.py
from google import genai

# Set your actual API key here or load it securely from env/config
API_KEY = "AIzaSyBFY9b0EDNl3J9G64-GP9lj3CT2ezhONZg"

client = genai.Client(api_key=API_KEY)

def get_gemini_response(text: str) -> str:
    """
    Uses Gemini API to generate a response from the given input text.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text,
        )
        return response.text
    except Exception as e:
        return f"Error from Gemini API: {str(e)}"

# testitng emini flash 2.5
import sys
import os
import google.genai as genai
MODEL_NAME = 'gemini-2.5-flash'    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# testingt sample query to check if api key and model are working fine
 
CLIENT = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
def test_gemini_generation():
    print("--- Testing Gemini Generation Validity ---")
    
    test_question = "What is the primary basis of the recommendations in this guide?"
    print(f"\nTest Question: {test_question}")
    
    try:
        response = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=[test_question]
        )
        
        print("\n--- RESULTS ---")
        print(f"Model Response: {response.text}")
        
    except Exception as e:
        print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    test_gemini_generation()
import os
import sys
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello and briefly describe what you can do."

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    print(response.text)

if __name__ == "__main__":
    main()

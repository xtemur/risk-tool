import os
import requests
from dotenv import load_dotenv

def authenticate():
    """
    Authenticate with the API and return authorization token.

    Returns:
        str: Authorization token if successful, None if failed
    """
    load_dotenv()

    api_url = os.getenv('API_URL')
    api_user = os.getenv('API_USER')
    api_pass = os.getenv('API_PASS')

    if not all([api_url, api_user, api_pass]):
        raise ValueError("Missing required environment variables: API_URL, API_USER, API_PASS")

    payload = {
        'action': 'login',
        'user': api_user,
        'password': api_pass
    }

    try:
        response = requests.post(
            api_url,
            data=payload,
            timeout=30,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        response.raise_for_status()

        # Handle plain text response (not JSON)
        response_text = response.text.strip()

        # Check if response looks like a token (format: hash:timestamp)
        if ':' in response_text and len(response_text.split(':')) == 2:
            token_part, timestamp = response_text.split(':')
            if len(token_part) == 32 and timestamp.isdigit():  # 32-char hash + numeric timestamp
                return response_text

        # If not a token format, try JSON parsing
        try:
            result = response.json()
            if 'token' in result:
                return result['token']
            else:
                raise ValueError(f"Authentication failed: {result}")
        except:
            raise ValueError(f"Unexpected response format: {response_text}")

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to API: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid response from API: {e}")

if __name__ == "__main__":
    try:
        token = authenticate()
        print(f"Authentication successful. Token: {token}")
    except Exception as e:
        print(f"Authentication failed: {e}")

import aiohttp
import asyncio
import json

class api_hook:
    def __init__(self, type='fetch', output_path="fetched.json"):
        self.type = type
        self.output_path = output_path

    async def _fetching_api(self, url):
        """Asynchronously fetch data from the provided URL with error handling."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()  # Raises exception for non-200 responses.
                    return await response.text()
        except aiohttp.ClientError as e:
            print(f"HTTP error occurred: {e}")
        except asyncio.TimeoutError:
            print("Request timed out.")
        except Exception as e:
            print(f"An unexpected error occurred during fetch: {e}")
    def _to_json(self, text):
        """Convert the fetched text to JSON and save it to a file."""
        try:
            data = json.loads(text)
            with open(self.output_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved to {self.output_path}")
        except json.JSONDecodeError:
            print("Failed to decode the response as JSON.")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    def run(self, url):
        loop = asyncio.get_event_loop()
        fetched_data = loop.run_until_complete(self._fetching_api(url))
        if fetched_data is None:
            print("Failed to fetch data.")
            return

        if self.type == 'fetch':
            self._to_json(fetched_data)
        else:
            print(fetched_data)

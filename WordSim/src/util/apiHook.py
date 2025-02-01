import aiohttp
import asyncio
import json

class api_hook:
    def __init__(self, type='fetch', output_path="fetched.json"):
        self.type = type
        self.output_path = output_path
    async def _fetching_api(self, url):
        """Asynchronously fetch data from the provided URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()

    def _to_json(self, text):
        """Convert the fetched text to JSON and save it to a file."""
        try:
            data = json.loads(text)
            with open(self.output_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved to {self.output_path}")
        except json.JSONDecodeError:
            print("Failed to decode the response as JSON.")
    def run(self, url):
        loop = asyncio.get_event_loop()
        fetched_data = loop.run_until_complete(self._fetching_api(url))
        if self.type == 'fetch':
            self._to_json(fetched_data)
        else:
            print(fetched_data)
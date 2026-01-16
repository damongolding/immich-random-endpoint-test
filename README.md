# Test for Immich's `search/random` api endpoint

A test script to see if Immich's `search/random` api endpoint holds any bias towards newer assets. The script only tests whole library assets, not filtered assets e.g. albums, people, tags etc.

## Requirements
- Python (tested with 3.12.1)
- [uv](https://docs.astral.sh/uv/)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/damongolding/immich-random-endpoint-test
   ```

2. Install dependencies:
   ```bash
   cd immich-random-endpoint-test
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   # fish
   # source venv/bin/activate.fish
   ```

4. Create .env file:
   ```bash
   cp .env.example .env
   ```

5. Add your API key to the .env file:
   ```bash
   IMMICH_URL=https://your_immich_instance_url
   IMMICH_API_KEY=your_api_key_here
   ```

6. Run the script:
   ```bash
   python main.py
   ```

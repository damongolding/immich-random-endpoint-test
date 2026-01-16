

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/immich-app/immich-random-test.git
   ```

2. Install dependencies:
   ```bash
   cd immich-random-test
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Create .env file:
   ```bash
   cp .env.example .env
   ```

5. Add your API key to the .env file:
   ```bash
   API_ENDPOINT=https://your_immich_instance_url/api/search/random
   API_KEY=your_api_key_here
   ```

6. Run the script:
   ```bash
   python main.py
   ```

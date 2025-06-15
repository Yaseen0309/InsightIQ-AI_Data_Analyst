# modules/env_loader.py

from dotenv import load_dotenv
import os

def load_environment():
    load_dotenv()
    os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_API_KEY', '')

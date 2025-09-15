import os, sys
sys.path.append("src")
from inference import VillageBot

def test_model_loads():
    model_dir = os.environ.get("MODEL_DIR", "./models/villageconnect-dialo")
    bot = VillageBot(model_dir)
    resp = bot.chat("Hello")
    assert isinstance(resp, str)
    assert len(resp) > 0

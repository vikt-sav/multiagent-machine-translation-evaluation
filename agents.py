import json
import time
import os
import requests
from dotenv import load_dotenv
import sys
from pathlib import Path

# Определяем путь к папке проекта
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / '.env'

# Явно загружаем .env файл
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"✅ Загружен .env файл: {ENV_PATH}")
else:
    print(f"❌ .env файл не найден по пути: {ENV_PATH}")
    # Пробуем альтернативные пути
    alt_paths = [
        BASE_DIR / 'config.env',
        BASE_DIR / 'settings.env',
        Path.cwd() / '.env'
    ]
    
    for alt_path in alt_paths:
        if alt_path.exists():
            load_dotenv(alt_path)
            print(f"✅ Загружен альтернативный файл: {alt_path}")
            break

class YandexGPTClient:
    def __init__(self):
        # Пробуем получить ключи разными способами
        self.api_key = os.getenv("YANDEX_API_KEY") or os.getenv("API_KEY") or os.getenv("YANDEX_KEY")
        self.folder_id = os.getenv("YANDEX_FOLDER_ID") or os.getenv("FOLDER_ID") or os.getenv("YANDEX_FOLDER")
        self.model = os.getenv("MODEL_NAME", "yandexgpt")
        
        # Если все еще не найдены - запрашиваем вручную
        if not self.api_key:
            self.api_key = input("Введите YANDEX_API_KEY: ").strip()
        
        if not self.folder_id:
            self.folder_id = input("Введите YANDEX_FOLDER_ID: ").strip()
        
        if not self.api_key or not self.folder_id:
            raise ValueError("Не удалось получить API ключи")
        
        print(f"✅ YandexGPT клиент инициализирован")
        print(f"   Модель: {self.model}")
        print(f"   Folder ID: {self.folder_id[:5]}...")
    
    def generate(self, system_prompt, user_prompt, temperature=0.1):
        """Отправка запроса к YandexGPT через REST API"""
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "modelUri": f"gpt://{self.folder_id}/{self.model}",
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": "2000"
            },
            "messages": [
                {
                    "role": "system",
                    "text": system_prompt
                },
                {
                    "role": "user",
                    "text": user_prompt
                }
            ]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                if "result" in result and "alternatives" in result["result"]:
                    alternatives = result["result"]["alternatives"]
                    if alternatives and len(alternatives) > 0:
                        return alternatives[0]["message"]["text"]
                return str(result)
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                return json.dumps({"error": error_msg})
                
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})


class TranslationEvaluator:
    def __init__(self):
        self.client = YandexGPTClient()
        # Импортируем промпты
        from config import AGENT_PROMPTS
        self.agent_prompts = AGENT_PROMPTS
        
    def evaluate_with_agent(self, agent_name, original, translation, **kwargs):
        """Оценка одним агентом"""
        if agent_name not in self.agent_prompts:
            return {"error": f"Агент {agent_name} не найден"}
        
        prompt_config = self.agent_prompts[agent_name]
        system_prompt = prompt_config["system"]
        
        # Форматируем user prompt
        try:
            user_prompt = prompt_config["user_template"].format(
                original=original,
                translation=translation,
                **kwargs
            )
        except KeyError:
            # Если в шаблоне есть дополнительные переменные
            user_prompt = prompt_config["user_template"].format(
                original=original,
                translation=translation
            )
        
        # Отправляем запрос
        response_text = self.client.generate(system_prompt, user_prompt)
        
        # Парсим JSON из ответа
        try:
            import re
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, response_text)
            
            if matches:
                json_str = max(matches, key=len)
                return json.loads(json_str)
            else:
                return {"raw_response": response_text, "error": "No JSON found"}
                
        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parse error: {e}",
                "raw_response": response_text[:500]
            }
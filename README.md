# Test Models - Генерация изображений с помощью AI

Этот проект содержит примеры использования различных AI моделей для генерации изображений через Hugging Face.

## 🚀 Возможности

- Генерация изображений через Hugging Face Inference API
- Локальная генерация с помощью diffusers
- Поддержка различных моделей (Stable Diffusion, NSFW модели)
- Автоматическое сохранение результатов

## 📁 Структура проекта

```
test-models/
├── README.md                 # Этот файл
├── requirements.txt          # Зависимости Python
├── .env.example             # Пример файла с переменными окружения
├── .gitignore               # Игнорируемые файлы
├── setup.sh                 # Скрипт для настройки окружения
├── test.py                  # Основной скрипт (Inference API)
├── test-hf-api.py          # Генерация через Hugging Face API
├── test-nsfw.py            # Локальная NSFW модель
├── test-nsfw-api.py        # NSFW модель через API
├── generated_images/        # Папка для сгенерированных изображений
└── models/                  # Папка для локальных моделей
```

## 🛠 Установка и настройка

### 1. Клонирование репозитория
```bash
git clone <your-repo-url>
cd test-models
```

### 2. Создание виртуального окружения
```bash
python3 -m venv env
source env/bin/activate  # Linux/Mac
# или
env\Scripts\activate     # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения
```bash
cp .env.example .env
# Отредактируйте .env файл и добавьте ваш HF_TOKEN
```

## 🎯 Использование

### Генерация через Hugging Face API (рекомендуется)
```bash
python test-hf-api.py
```

### Локальная генерация (требует больше ресурсов)
```bash
python test-nsfw.py
```

### Основной скрипт
```bash
python test.py
```

## 📋 Доступные модели

### Через Inference API:
- `stabilityai/stable-diffusion-xl-base-1.0`
- `runwayml/stable-diffusion-v1-5`
- `CompVis/stable-diffusion-v1-4`

### Локально (требует GPU):
- `UnfilteredAI/NSFW-gen-v2`
- `stabilityai/stable-diffusion-2-1`

## 🔧 Требования

- Python 3.8+
- Hugging Face токен с правами на Inference API
- Для локальных моделей: GPU с CUDA (рекомендуется)

## 📝 Переменные окружения

Создайте файл `.env` со следующим содержимым:
```
HF_TOKEN=your_huggingface_token_here
```

## 🚨 Важные замечания

- NSFW модели могут быть недоступны через Inference API
- Локальные модели требуют значительных ресурсов (GPU, RAM)
- Убедитесь, что у вашего токена есть права на Inference API

## 📄 Лицензия

MIT License

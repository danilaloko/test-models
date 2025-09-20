# Telegram Bot для генерации изображений

Telegram бот для тестирования различных AI моделей генерации изображений с модульной архитектурой.

## 🎯 Возможности

- **Модульная архитектура**: Каждый провайдер моделей реализован отдельно
- **Множественные провайдеры**: Поддержка различных сервисов и моделей
- **Telegram интерфейс**: Удобное взаимодействие через Telegram
- **Тестовый режим**: Mock провайдер для тестирования без реальных моделей

## 🏗️ Архитектура

### Провайдеры

1. **Mock Provider** - Тестовый провайдер для проверки логики
2. **Hugging Face API** - Быстрая генерация через Inference API
3. **NSFW Local** - Локальные NSFW модели (требует GPU)
4. **FLUX** - FLUX модели с поддержкой LoRA адаптеров

### Структура проекта

```
├── telegram_bot.py          # Основной файл бота
├── config.py               # Конфигурация
├── test_bot.py             # Скрипт для тестирования
├── providers/              # Директория провайдеров
│   ├── __init__.py
│   ├── base_provider.py    # Базовый класс провайдера
│   ├── provider_manager.py # Менеджер провайдеров
│   ├── mock_provider.py    # Тестовый провайдер
│   ├── hf_api_provider.py  # Hugging Face API
│   ├── nsfw_provider.py    # NSFW модели
│   └── flux_provider.py    # FLUX модели
├── generated_images/       # Сгенерированные изображения
└── requirements.txt        # Зависимости
```

## 🚀 Установка и настройка

### 1. Клонирование и установка зависимостей

```bash
git clone <repository>
cd test-models
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

Скопируйте `env.example` в `.env` и заполните токены:

```bash
cp env.example .env
```

Отредактируйте `.env`:

```env
# Telegram Bot Token (получите у @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Hugging Face Token (получите на https://huggingface.co/settings/tokens)
HF_TOKEN=your_huggingface_token_here
```

### 3. Создание Telegram бота

1. Найдите @BotFather в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям для создания бота
4. Скопируйте полученный токен в `.env` файл

## 🎮 Использование

### Запуск бота

```bash
python telegram_bot.py
```

### Команды бота

- `/start` - Начать работу с ботом
- `/help` - Показать справку
- `/models` - Выбрать модель для генерации
- `/status` - Показать статус всех провайдеров

### Процесс генерации

1. Выберите модель командой `/models`
2. Отправьте текстовое описание изображения
3. Дождитесь генерации
4. Получите готовое изображение

## 🧪 Тестирование

### Тестирование без реальных моделей

```bash
python test_bot.py
```

Этот скрипт протестирует:
- Инициализацию всех провайдеров
- Генерацию изображений через доступные провайдеры
- Mock провайдер (работает без токенов)

### Тестирование только Mock провайдера

Если у вас нет токенов, скрипт автоматически протестирует только Mock провайдер.

## 🔧 Разработка

### Добавление нового провайдера

1. Создайте новый файл в `providers/`
2. Наследуйтесь от `BaseProvider`
3. Реализуйте обязательные методы:
   - `initialize()` - инициализация
   - `generate_image()` - генерация изображения
   - `is_available()` - проверка доступности

4. Зарегистрируйте провайдер в `telegram_bot.py`:

```python
self.provider_manager.register_provider('new_provider', NewProvider())
```

### Пример провайдера

```python
from providers.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self):
        super().__init__("My Provider", "Описание провайдера")
    
    async def initialize(self) -> bool:
        # Инициализация
        return True
    
    async def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Генерация изображения
        return {
            "success": True,
            "image_path": "path/to/image.png",
            "metadata": {}
        }
    
    def is_available(self) -> bool:
        # Проверка доступности
        return True
```

## 📋 Требования

- Python 3.8+
- CUDA (для локальных моделей)
- Достаточно RAM для загрузки моделей
- Токены для API сервисов

## 🐛 Устранение неполадок

### Проблемы с токенами

- Убедитесь, что токены правильно указаны в `.env`
- Проверьте права токенов (для HF нужен Inference API)

### Проблемы с GPU

- Установите CUDA драйверы
- Проверьте доступность GPU: `nvidia-smi`
- Для CPU режима модели будут работать медленно

### Проблемы с памятью

- Используйте меньшие модели
- Уменьшите размер изображений
- Закройте другие приложения

## 📝 Лицензия

MIT License
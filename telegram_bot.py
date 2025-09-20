#!/usr/bin/env python3
"""
Telegram бот для тестирования различных моделей генерации изображений
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from providers.base_provider import BaseProvider
from providers.hf_api_provider import HFApiProvider
from providers.nsfw_provider import NSFWProvider
from providers.flux_provider import FluxProvider
from providers.mock_provider import MockProvider
from providers.provider_manager import ProviderManager

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

class ImageGenBot:
    """Основной класс Telegram бота для генерации изображений"""
    
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN не найден в переменных окружения")
        
        # Инициализируем менеджер провайдеров
        self.provider_manager = ProviderManager()
        
        # Регистрируем провайдеры
        self.provider_manager.register_provider('mock', MockProvider())
        self.provider_manager.register_provider('hf_api', HFApiProvider())
        self.provider_manager.register_provider('nsfw', NSFWProvider())
        self.provider_manager.register_provider('flux', FluxProvider())
        
        # Создаем приложение
        self.application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики
        self._register_handlers()
        
        # Инициализируем только mock провайдер при запуске
        self._initialize_mock_provider()
    
    def _register_handlers(self):
        """Регистрирует обработчики команд и сообщений"""
        # Команды
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("models", self.models_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
        # Обработчик выбора модели
        self.application.add_handler(CallbackQueryHandler(self.model_selection_callback, pattern="^model_"))
        
        # Обработчик текстовых сообщений (промпты)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_prompt))
    
    async def _initialize_mock_provider(self):
        """Инициализирует только mock провайдер при запуске"""
        logger.info("Инициализация mock провайдера...")
        success = await self.provider_manager.initialize_provider('mock')
        if success:
            logger.info("✅ Mock провайдер инициализирован")
        else:
            logger.warning("❌ Mock провайдер не инициализирован")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_text = """
🎨 Добро пожаловать в бот для генерации изображений!

Этот бот позволяет тестировать различные AI модели для создания изображений по текстовым описаниям.

Доступные команды:
/help - Показать справку
/models - Выбрать модель
/status - Показать статус провайдеров

Просто отправьте текстовое описание изображения, и я сгенерирую его для вас!
        """
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📖 Справка по использованию бота:

🎯 Основные команды:
/start - Начать работу с ботом
/help - Показать эту справку
/models - Выбрать модель для генерации
/status - Показать статус всех провайдеров

🎨 Как использовать:
1. Выберите модель командой /models
2. Отправьте текстовое описание желаемого изображения
3. Дождитесь генерации (может занять время)
4. Получите готовое изображение!

💡 Советы для лучших результатов:
- Будьте конкретными в описании
- Указывайте стиль, цвета, детали
- Используйте ключевые слова: "high quality", "detailed", "8k"

⚠️ Ограничения:
- Некоторые модели могут быть недоступны
- Генерация может занять время
- Размер изображений может варьироваться
        """
        await update.message.reply_text(help_text)
    
    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /models - показывает доступные модели"""
        keyboard = []
        
        for provider_id, provider in self.provider_manager.providers.items():
            is_initialized = self.provider_manager.initialization_status.get(provider_id, False)
            is_initializing = self.provider_manager.initialization_in_progress.get(provider_id, False)
            
            if is_initializing:
                status = "🔄"
                button_text = f"{status} {provider.get_name()} (загружается...)"
            elif is_initialized and provider.is_available():
                status = "✅"
                button_text = f"{status} {provider.get_name()}"
            elif is_initialized:
                status = "❌"
                button_text = f"{status} {provider.get_name()} (недоступен)"
            else:
                status = "⏸️"
                button_text = f"{status} {provider.get_name()} (не загружен)"
            
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"model_{provider_id}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🎨 Выберите модель для генерации изображений:\n\n"
            "🔄 - загружается\n"
            "✅ - доступен\n"
            "❌ - недоступен\n"
            "⏸️ - не загружен (будет загружен при выборе)",
            reply_markup=reply_markup
        )
    
    async def model_selection_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик выбора модели"""
        query = update.callback_query
        await query.answer()
        
        provider_id = query.data.replace("model_", "")
        provider = self.provider_manager.get_provider(provider_id)
        
        if not provider:
            await query.edit_message_text("❌ Модель не найдена")
            return
        
        # Проверяем, инициализирован ли провайдер
        is_initialized = self.provider_manager.initialization_status.get(provider_id, False)
        is_initializing = self.provider_manager.initialization_in_progress.get(provider_id, False)
        
        if is_initializing:
            await query.edit_message_text(
                f"⏳ Модель {provider.get_name()} уже загружается...\n"
                f"Пожалуйста, подождите и попробуйте снова через несколько секунд."
            )
            return
        
        if not is_initialized:
            # Показываем сообщение о начале загрузки
            loading_message = await query.edit_message_text(
                f"🔄 Загружаю модель {provider.get_name()}...\n"
                f"⏳ Это может занять некоторое время..."
            )
            
            # Инициализируем провайдер
            success = await self.provider_manager.initialize_provider_lazy(provider_id)
            
            if not success:
                await loading_message.edit_text(
                    f"❌ Не удалось загрузить модель {provider.get_name()}\n"
                    f"Попробуйте выбрать другую модель."
                )
                return
            
            # Проверяем доступность после инициализации
            if not provider.is_available():
                await loading_message.edit_text(
                    f"❌ Модель {provider.get_name()} недоступна после загрузки"
                )
                return
        
        # Сохраняем выбранную модель в контексте пользователя
        context.user_data['selected_provider'] = provider_id
        
        await query.edit_message_text(
            f"✅ Выбрана модель: {provider.get_name()}\n\n"
            f"📝 Отправьте текстовое описание изображения для генерации!"
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status - показывает статус провайдеров"""
        status_text = "📊 Статус провайдеров моделей:\n\n"
        
        provider_status = self.provider_manager.get_provider_status()
        
        for provider_id, status_info in provider_status.items():
            provider = self.provider_manager.get_provider(provider_id)
            status_text += f"🔹 {provider.get_name()}\n"
            status_text += f"   Статус: {status_info['status']}\n"
            status_text += f"   Описание: {status_info['description']}\n"
            
            # Добавляем дополнительную информацию если доступна
            if hasattr(provider, 'get_current_model'):
                status_text += f"   Модель: {provider.get_current_model()}\n"
            
            if hasattr(provider, 'get_device'):
                status_text += f"   Устройство: {provider.get_device()}\n"
            
            status_text += "\n"
        
        await update.message.reply_text(status_text)
    
    async def handle_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений (промптов)"""
        user_id = update.effective_user.id
        prompt = update.message.text
        
        # Проверяем, выбрана ли модель
        selected_provider_id = context.user_data.get('selected_provider')
        if not selected_provider_id:
            await update.message.reply_text(
                "⚠️ Сначала выберите модель командой /models"
            )
            return
        
        provider = self.provider_manager.get_provider(selected_provider_id)
        if not provider:
            await update.message.reply_text(
                "❌ Выбранная модель не найдена. Выберите другую модель командой /models"
            )
            return
        
        # Проверяем, инициализирован ли провайдер
        is_initialized = self.provider_manager.initialization_status.get(selected_provider_id, False)
        if not is_initialized:
            await update.message.reply_text(
                "❌ Модель не инициализирована. Выберите модель командой /models"
            )
            return
        
        # Отправляем сообщение о начале генерации
        status_message = await update.message.reply_text(
            f"🎨 Генерирую изображение с помощью {provider.get_name()}...\n"
            f"⏳ Это может занять некоторое время..."
        )
        
        try:
            # Генерируем изображение
            result = await provider.generate_image(prompt)
            
            if result and result.get('success'):
                # Отправляем изображение
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=result['image_path'],
                    caption=f"🎨 Изображение сгенерировано с помощью {provider.get_name()}\n"
                           f"📝 Промпт: {prompt}"
                )
                
                # Удаляем сообщение о статусе
                await status_message.delete()
                
            else:
                error_msg = result.get('error', 'Неизвестная ошибка') if result else 'Ошибка генерации'
                await status_message.edit_text(
                    f"❌ Ошибка генерации: {error_msg}\n"
                    f"Попробуйте другую модель или измените промпт."
                )
                
        except Exception as e:
            logger.error(f"Ошибка при генерации изображения: {e}")
            await status_message.edit_text(
                f"❌ Произошла ошибка при генерации: {str(e)}\n"
                f"Попробуйте позже или выберите другую модель."
            )
    
    def run(self):
        """Запускает бота"""
        logger.info("Запуск Telegram бота...")
        self.application.run_polling()

def main():
    """Главная функция"""
    try:
        bot = ImageGenBot()
        bot.run()
    except Exception as e:
        logger.error(f"Ошибка запуска бота: {e}")
        print(f"❌ Ошибка запуска бота: {e}")

if __name__ == "__main__":
    main()

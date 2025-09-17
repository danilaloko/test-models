import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys

def load_model():
    """Загружает модель и токенизатор с правильными параметрами"""
    try:
        print("Загружаем модель NSFW-3B...")
        model = AutoModelForCausalLM.from_pretrained(
            "UnfilteredAI/NSFW-3B", 
            trust_remote_code=True, 
            torch_dtype=torch.float16,  # Изменено с bfloat16 на float16
            device_map="auto"  # Автоматическое размещение на GPU
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "UnfilteredAI/NSFW-3B", 
            trust_remote_code=True
        )
        
        # Устанавливаем pad_token если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Модель успешно загружена!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return None, None

def create_prompt(system_message, user_message):
    """Создает промпт в правильном формате"""
    return f"""<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{user_message}
<|im_end|>
<|im_start|>assistant
"""

def generate_response(model, tokenizer, prompt, max_length=512):
    """Генерирует ответ модели"""
    try:
        # Токенизируем входной текст
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Перемещаем на GPU если доступно
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Создаем стример для вывода
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Генерируем ответ
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
                use_cache=True
            )
        
        # Декодируем ответ
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def main():
    """Основная функция для непрерывного общения с моделью"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("Не удалось загрузить модель. Завершение работы.")
        return
    
    print("\n" + "="*50)
    print("🤖 Чат с NSFW-3B запущен!")
    print("Введите 'quit', 'exit' или 'выход' для завершения")
    print("Введите 'clear' для очистки истории")
    print("="*50 + "\n")
    
    # Системное сообщение
    system_message = "You are a helpful AI assistant."
    
    while True:
        try:
            # Получаем ввод пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            # Проверяем команды выхода
            if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                print("👋 До свидания!")
                break
            
            # Проверяем команду очистки
            if user_input.lower() in ['clear', 'очистить']:
                print("🧹 История очищена!")
                continue
            
            # Пропускаем пустые сообщения
            if not user_input:
                continue
            
            # Создаем промпт
            prompt = create_prompt(system_message, user_input)
            
            print("\n🤖 Ассистент: ", end="", flush=True)
            
            # Генерируем ответ
            response = generate_response(model, tokenizer, prompt)
            
            print()  # Новая строка после ответа
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            continue

if __name__ == "__main__":
    main()

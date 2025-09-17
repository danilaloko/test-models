import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import sys

def load_alternative_model():
    """Загружает альтернативную модель в случае проблем с NSFW-3B"""
    try:
        # Список альтернативных моделей для тестирования
        models_to_try = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small", 
            "gpt2",
            "distilgpt2"
        ]
        
        for model_name in models_to_try:
            try:
                print(f"Попытка загрузки модели: {model_name}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Устанавливаем pad_token если его нет
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Перемещаем на GPU если доступно
                if torch.cuda.is_available():
                    model = model.to("cuda")
                
                print(f"✅ Модель {model_name} успешно загружена!")
                return model, tokenizer, model_name
                
            except Exception as e:
                print(f"❌ Не удалось загрузить {model_name}: {e}")
                continue
        
        print("Не удалось загрузить ни одну из альтернативных моделей")
        return None, None, None
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return None, None, None

def generate_response_alternative(model, tokenizer, user_input, model_name):
    """Генерирует ответ для альтернативных моделей"""
    try:
        # Для DialoGPT используем специальный формат
        if "DialoGPT" in model_name:
            # Кодируем входное сообщение
            inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = inputs.to(model.device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            return response.strip()
        
        else:
            # Для обычных GPT моделей
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return "Извините, произошла ошибка при генерации ответа."

def main():
    """Основная функция для работы с альтернативными моделями"""
    model, tokenizer, model_name = load_alternative_model()
    
    if model is None:
        print("Не удалось загрузить ни одну модель. Завершение работы.")
        return
    
    print(f"\n{'='*60}")
    print(f"🤖 Чат с {model_name} запущен!")
    print("Введите 'quit', 'exit' или 'выход' для завершения")
    print(f"{'='*60}\n")
    
    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'выход', 'q']:
                print("👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Ассистент: ", end="", flush=True)
            response = generate_response_alternative(model, tokenizer, user_input, model_name)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Произошла ошибка: {e}")
            continue

if __name__ == "__main__":
    main()

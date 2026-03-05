"""
Быстрый тест системы
"""
import pandas as pd
import os
from main import load_data, prepare_long_data, evaluate_single_translation
from agents import TranslationEvaluator

def quick_test():
    """Быстрая проверка работы системы"""
    print("🚀 БЫСТРЫЙ ТЕСТ СИСТЕМЫ")
    print("="*50)
    
    # Создаем тестовые данные
    test_data = {
        'domain': ['medical', 'legal', 'publicistic'],
        'segment_id': [1, 2, 3],
        'source': [
            'The patient was administered 500mg of aspirin intravenously.',
            'The defendant shall be liable for all damages incurred.',
            'The rapid development of AI technologies raises important ethical questions.'
        ],
        'google': [
            'Пациенту внутривенно вводили 500 мг аспирина.',
            'Ответчик несет ответственность за все причиненные убытки.',
            'Быстрое развитие технологий ИИ поднимает важные этические вопросы.'
        ],
        'yandex': [
            'Пациенту было введено 500 мг аспирина внутривенно.',
            'Ответчик должен нести ответственность за все причиненные убытки.',
            'Стремительное развитие технологий ИИ ставит важные этические вопросы.'
        ],
        'deepl': [
            'Пациенту внутривенно ввели 500 мг аспирина.',
            'Ответчик несет ответственность за все причиненные убытки.',
            'Быстрое развитие технологий искусственного интеллекта ставит важные этические вопросы.'
        ]
    }
    
    # Создаем папку data если её нет
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Сохраняем тестовые данные
    test_file = os.path.join(data_dir, "test_translations.csv")
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_file, index=False, encoding='utf-8')
    
    print(f"✅ Создан тестовый файл: {test_file}")
    
    # Загружаем данные
    print("\n📂 Загрузка тестовых данных...")
    df_raw = load_data(test_file)
    df_long = prepare_long_data(df_raw)
    
    # Тестируем оценку
    print("\n🤖 Тестирование оценки переводов...")
    evaluator = TranslationEvaluator()
    
    # Берем по одному примеру из каждой тематики
    for theme in ['medical', 'legal', 'publicistic']:
        theme_rows = df_long[df_long['theme'] == theme]
        if not theme_rows.empty:
            row = theme_rows.iloc[0]
            print(f"\n--- {theme.upper()} ---")
            result = evaluate_single_translation(evaluator, row)
            
            # Показываем результат
            mod_eval = result.get('moderator_evaluation', {})
            if isinstance(mod_eval, dict) and 'final_score' in mod_eval:
                print(f"Финальная оценка: {mod_eval['final_score']}")
    
    print("\n✅ Тест завершен!")

if __name__ == "__main__":
    quick_test()
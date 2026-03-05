import pandas as pd
import json
import time
import os
from datetime import datetime
from agents import TranslationEvaluator
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename="translations.csv"):
    possible_paths = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, "data", "translations.csv"),
        os.path.join(BASE_DIR, filename),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"📁 Найден файл: {path}")
            return path
    
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "translations.csv")

def load_data(filepath=None):
    if filepath is None:
        filepath = get_data_path()
    
    print(f"📂 Загрузка файла: {filepath}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp1251')
    
    # Проверяем структуру
    expected_columns = ['domain', 'segment_id', 'source', 'google', 'yandex', 'deepl']
    df_columns_lower = [col.lower() for col in df.columns]
    
    missing = [col for col in expected_columns if col not in df_columns_lower]
    if missing:
        print(f"❌ Отсутствуют колонки: {missing}")
        print(f"   Имеются: {list(df.columns)}")
        raise ValueError("Неверная структура CSV")
    
    # Переименовываем
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'source':
            rename_map[col] = 'original'
        elif col_lower == 'domain':
            rename_map[col] = 'theme'
    
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    
    print(f"✅ Загружено {len(df)} сегментов")
    print("\n📊 Распределение по доменам:")
    print(df['theme'].value_counts())
    
    return df

def prepare_long_data(df):
    """Преобразование в long-формат"""
    records = []
    
    for idx, row in df.iterrows():
        for engine in ['google', 'yandex', 'deepl']:
            if pd.notna(row[engine]) and str(row[engine]).strip():
                records.append({
                    'segment_id': row['segment_id'],
                    'theme': row['theme'],
                    'original': row['original'],
                    'translation_engine': engine,
                    'translation': str(row[engine])
                })
    
    result_df = pd.DataFrame(records)
    
    print(f"\n📊 Создано {len(result_df)} записей для оценки:")
    print(f"  • Сегментов: {result_df['segment_id'].nunique()}")
    print(f"  • Тематик: {result_df['theme'].nunique()}")
    print(f"  • Движков: {result_df['translation_engine'].nunique()}")
    
    return result_df

def get_full_agent_set(theme):
    """ПОЛНЫЙ набор агентов для любой тематики"""
    
    # Базовые универсальные агенты (для всех)
    base_agents = ["linguist", "semantic", "terminologist", "pragmatist", "cultural_adaptor"]
    
    # Тематический агент
    theme_lower = str(theme).lower()
    if any(word in theme_lower for word in ['мед', 'med', 'health', 'clinical']):
        theme_agent = "medical_expert"
    elif any(word in theme_lower for word in ['юр', 'law', 'legal', 'court']):
        theme_agent = "legal_expert"
    else:
        theme_agent = "stylist"
    
    # Все агенты + модератор
    all_agents = base_agents + [theme_agent]
    
    return all_agents

def evaluate_single_translation(evaluator, row):
    """Оценка с полным набором агентов"""
    
    # Получаем полный список агентов для этой тематики
    agents_to_run = get_full_agent_set(row['theme'])
    
    print(f"\n  🤖 Агенты ({len(agents_to_run)}): {', '.join(agents_to_run)}")
    
    # Запуск всех экспертов
    expert_results = {}
    for agent_name in agents_to_run:
        try:
            result = evaluator.evaluate_with_agent(
                agent_name, 
                row['original'], 
                row['translation']
            )
            expert_results[agent_name] = result
            print(f"    ✓ {agent_name}")
            time.sleep(0.3)  # Небольшая пауза
        except Exception as e:
            print(f"    ✗ {agent_name}: {str(e)[:50]}")
            expert_results[agent_name] = {"error": str(e)}
    
    # Запуск модератора
    try:
        moderator_result = evaluator.evaluate_with_agent(
            "moderator",
            row['original'],
            row['translation'],
            theme=row['theme'],
            experts_evaluations=json.dumps(expert_results, ensure_ascii=False, indent=2)
        )
        print(f"    ✓ moderator")
    except Exception as e:
        print(f"    ✗ moderator: {str(e)[:50]}")
        moderator_result = {"error": str(e)}
    
    # Формируем результат
    final_result = {
        "segment_id": int(row['segment_id']) if pd.notna(row['segment_id']) else 0,
        "theme": str(row['theme']),
        "translation_engine": row['translation_engine'],
        "original_preview": row['original'][:150] + "..." if len(str(row['original'])) > 150 else str(row['original']),
        "translation_preview": row['translation'][:150] + "..." if len(str(row['translation'])) > 150 else str(row['translation']),
        "expert_evaluations": expert_results,
        "moderator_evaluation": moderator_result,
        "processing_time": datetime.now().isoformat(),
        "agents_used": agents_to_run
    }
    
    return final_result

def run_full_evaluation(dataframe, start_from=0, max_rows=None, sample_size=None):
    """Запуск полной оценки со всеми агентами"""
    evaluator = TranslationEvaluator()
    results = []
    
    results_dir = os.path.join(BASE_DIR, "results_full")
    os.makedirs(results_dir, exist_ok=True)
    
    df_to_process = dataframe.iloc[start_from:].copy()
    
    if max_rows:
        df_to_process = df_to_process.head(max_rows)
    
    if sample_size and sample_size < len(df_to_process):
        df_to_process = df_to_process.sample(sample_size, random_state=42)
    
    total_rows = len(df_to_process)
    print(f"\n📊 Будет обработано: {total_rows} переводов")
    print(f"🤖 Агентов на перевод: 6 (linguist, semantic, terminologist, pragmatist, cultural_adaptor + тематический)")
    print(f"💰 Примерная стоимость: {total_rows * 6 * 0.4:.1f} руб.")
    
    print("\n📈 Распределение в выборке:")
    pivot = df_to_process.groupby(['theme', 'translation_engine']).size().unstack(fill_value=0)
    print(pivot)
    
    confirm = input("\nПродолжить? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Прервано пользователем")
        return []
    
    progress_bar = tqdm(df_to_process.iterrows(), total=total_rows, desc="Полная оценка")
    
    for idx, row in progress_bar:
        try:
            progress_bar.set_postfix({
                'id': row['segment_id'],
                'theme': str(row['theme'])[:10],
                'engine': row['translation_engine']
            })
            
            result = evaluate_single_translation(evaluator, row)
            results.append(result)
            
            # Промежуточное сохранение каждые 3 записи
            if len(results) % 3 == 0:
                save_intermediate_results(results, results_dir, f"batch_{len(results)}")
            
            time.sleep(1)  # Пауза между запросами
            
        except Exception as e:
            tqdm.write(f"❌ Ошибка в строке {idx}: {e}")
            continue
    
    progress_bar.close()
    return results

def save_intermediate_results(results, results_dir, suffix):
    """Сохранение промежуточных результатов"""
    timestamp = datetime.now().strftime("%H%M%S")
    filename = os.path.join(results_dir, f"partial_eval_{suffix}_{timestamp}.json")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n  💾 Сохранено {len(results)} результатов")

def save_final_results(results, results_dir="results_full"):
    """Сохранение финальных результатов"""
    if not results:
        print("Нет результатов для сохранения")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Полные результаты
    full_filename = os.path.join(results_dir, f"full_evaluations_{timestamp}.json")
    with open(full_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 2. Сводная таблица
    summary_data = []
    for res in results:
        summary = {
            'segment_id': res['segment_id'],
            'theme': res['theme'],
            'engine': res['translation_engine'],
            'agents_used': len(res['agents_used'])
        }
        
        # Добавляем оценки от каждого агента (если есть)
        for agent, eval_data in res['expert_evaluations'].items():
            if isinstance(eval_data, dict):
                # Ищем overall score
                for key in eval_data:
                    if 'overall' in key.lower() or 'score' in key.lower():
                        if isinstance(eval_data[key], (int, float)):
                            summary[f'{agent}_score'] = eval_data[key]
                            break
        
        # Оценка модератора
        mod_eval = res.get('moderator_evaluation', {})
        if isinstance(mod_eval, dict):
            if 'final_score' in mod_eval:
                summary['final_score'] = mod_eval['final_score']
            summary['expert_agreement'] = mod_eval.get('expert_agreement', 'N/A')
            summary['confidence'] = mod_eval.get('confidence_level', 'N/A')
        
        summary_data.append(summary)
    
    df_summary = pd.DataFrame(summary_data)
    csv_filename = os.path.join(results_dir, f"evaluations_summary_{timestamp}.csv")
    df_summary.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("✅ ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН")
    print(f"{'='*60}")
    print(f"📁 Полные данные:    {full_filename}")
    print(f"📊 Сводная таблица:  {csv_filename}")
    
    # Базовая статистика
    if 'final_score' in df_summary.columns:
        print(f"\n📈 Базовая статистика:")
        print(df_summary.groupby('theme')['final_score'].agg(['mean', 'std', 'count']).round(2))
    
    return df_summary

def main():
    print("="*70)
    print("🚀 ПОЛНЫЙ МЕЖАГЕНТНЫЙ АНАЛИЗ КАЧЕСТВА ПЕРЕВОДА")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🤖 Состав: 6 агентов (linguist, semantic, terminologist, pragmatist, cultural_adaptor + тематический)")
    print("="*70)
    
    try:
        # Загрузка данных
        print("\n📂 1. ЗАГРУЗКА ДАННЫХ...")
        df_raw = load_data()
        df_long = prepare_long_data(df_raw)
        
        # Выбор режима
        print("\n🎯 2. ВЫБЕРИТЕ РЕЖИМ:")
        print("   1. Полный анализ (все переводы, 6 агентов)")
        print("   2. Тестовый запуск (3 перевода)")
        print("   3. Выборочный анализ (N случайных)")
        
        choice = input("\nВаш выбор (1-3): ").strip()
        
        results = []
        if choice == "1":
            results = run_full_evaluation(df_long)
        elif choice == "2":
            results = run_full_evaluation(df_long, sample_size=3)
        elif choice == "3":
            sample_size = int(input("Размер выборки: "))
            results = run_full_evaluation(df_long, sample_size=sample_size)
        else:
            print("Неверный выбор")
            return
        
        # Сохранение
        if results:
            print("\n💾 3. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")
            save_final_results(results)
            print("\n✅ Полный анализ успешно завершен!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n📌 Поместите CSV файл в папку 'data' с именем 'translations.csv'")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
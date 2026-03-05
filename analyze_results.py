"""
Полный анализ результатов межагентной оценки переводов
Для ВАК-статьи
Исправленная версия с обработкой ошибок и разной длины данных
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
from scipy import stats
from scipy.stats import pearsonr, f_oneway, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Создаем папку для визуализаций
os.makedirs('visualizations', exist_ok=True)

class TranslationAnalysis:
    def __init__(self, summary_file=None):
        """Инициализация с загрузкой последнего файла результатов"""
        print("="*80)
        print("🔬 ИНИЦИАЛИЗАЦИЯ АНАЛИЗА")
        print("="*80)
        
        # Загрузка сводной таблицы
        if summary_file is None:
            files = glob.glob("results_full/evaluations_summary_*.csv")
            if not files:
                raise FileNotFoundError("❌ Нет файлов с результатами в папке results_full/")
            self.summary_file = max(files, key=os.path.getctime)
        else:
            self.summary_file = summary_file
        
        print(f"📊 Загружаем сводную таблицу: {self.summary_file}")
        self.df = pd.read_csv(self.summary_file)
        
        # Загрузка полных данных
        full_files = glob.glob("results_full/full_evaluations_*.json")
        if not full_files:
            raise FileNotFoundError("❌ Нет полных данных в папке results_full/")
        self.full_file = max(full_files, key=os.path.getctime)
        
        print(f"📁 Загружаем полные данные: {self.full_file}")
        with open(self.full_file, 'r', encoding='utf-8') as f:
            self.full_data = json.load(f)
        
        print(f"✅ Загружено: {len(self.df)} оценок, {len(self.full_data)} полных записей")
        
        # Анализ состава агентов
        self.analyze_agent_composition()
    
    def analyze_agent_composition(self):
        """Анализ того, какие агенты присутствуют в данных"""
        print("\n📊 СОСТАВ АГЕНТОВ В ДАННЫХ:")
        
        agent_presence = {}
        for record in self.full_data:
            for agent in record.get('expert_evaluations', {}).keys():
                agent_presence[agent] = agent_presence.get(agent, 0) + 1
        
        for agent, count in sorted(agent_presence.items()):
            percentage = (count / len(self.full_data)) * 100
            print(f"   {agent}: {count} записей ({percentage:.1f}%)")
        
        self.agent_presence = agent_presence
    
    def basic_statistics(self):
        """Базовая статистика"""
        print("\n" + "="*70)
        print("1. БАЗОВАЯ СТАТИСТИКА")
        print("="*70)
        
        # Проверка наличия пропусков
        missing = self.df['final_score'].isna().sum()
        if missing > 0:
            print(f"⚠️ Обнаружено {missing} пропущенных значений")
            df_clean = self.df.dropna(subset=['final_score'])
        else:
            df_clean = self.df
        
        # Общая статистика
        print(f"\n📊 Общая статистика:")
        print(f"   Всего оценок: {len(df_clean)}")
        print(f"   Средняя оценка: {df_clean['final_score'].mean():.2f}")
        print(f"   Медиана: {df_clean['final_score'].median():.2f}")
        print(f"   Стандартное отклонение: {df_clean['final_score'].std():.2f}")
        print(f"   Минимум: {df_clean['final_score'].min():.2f}")
        print(f"   Максимум: {df_clean['final_score'].max():.2f}")
        
        # По тематикам
        print(f"\n📊 По тематикам:")
        theme_stats = df_clean.groupby('theme')['final_score'].agg(['mean', 'std', 'count', 'min', 'max'])
        print(theme_stats.round(2))
        
        # По движкам
        print(f"\n📊 По движкам перевода:")
        engine_stats = df_clean.groupby('engine')['final_score'].agg(['mean', 'std', 'count'])
        print(engine_stats.round(2))
        
        return theme_stats, engine_stats
    
    def engine_by_theme_analysis(self):
        """Анализ движков по тематикам"""
        print("\n" + "="*70)
        print("2. СРАВНЕНИЕ ДВИЖКОВ ПО ТЕМАТИКАМ")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score'])
        
        # Сводная таблица
        pivot_mean = df_clean.pivot_table(
            values='final_score',
            index='theme',
            columns='engine',
            aggfunc='mean'
        ).round(2)
        
        pivot_std = df_clean.pivot_table(
            values='final_score',
            index='theme',
            columns='engine',
            aggfunc='std'
        ).round(2)
        
        print("\n📊 Средние оценки (mean):")
        print(pivot_mean)
        
        print("\n📊 Стандартные отклонения (std):")
        print(pivot_std)
        
        # Определяем лучший движок для каждой тематики
        print("\n🏆 Лучший движок по тематикам:")
        for theme in df_clean['theme'].unique():
            theme_data = df_clean[df_clean['theme'] == theme]
            if not theme_data.empty:
                means = theme_data.groupby('engine')['final_score'].mean()
                best_engine = means.idxmax()
                best_score = means.max()
                print(f"   {theme}: {best_engine} ({best_score:.2f})")
        
        return pivot_mean, pivot_std
    
    def statistical_tests(self):
        """Статистические тесты"""
        print("\n" + "="*70)
        print("3. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score'])
        
        # ANOVA по тематикам
        themes = df_clean['theme'].unique()
        theme_groups = [df_clean[df_clean['theme'] == t]['final_score'] for t in themes]
        
        f_stat, p_value = f_oneway(*theme_groups)
        print(f"\n📊 ANOVA по тематикам:")
        print(f"   F-статистика: {f_stat:.4f}")
        print(f"   p-значение: {p_value:.4f}")
        print(f"   {'✅ Статистически значимо (p < 0.05)' if p_value < 0.05 else '❌ Не значимо (p >= 0.05)'}")
        
        # t-test между движками
        print(f"\n📊 Сравнение движков (t-test):")
        engines = df_clean['engine'].unique()
        results = []
        for i, e1 in enumerate(engines):
            for e2 in engines[i+1:]:
                scores1 = df_clean[df_clean['engine'] == e1]['final_score']
                scores2 = df_clean[df_clean['engine'] == e2]['final_score']
                t_stat, p_val = stats.ttest_ind(scores1, scores2)
                sig = "✅" if p_val < 0.05 else "❌"
                print(f"   {e1} vs {e2}: p={p_val:.4f} {sig}")
                results.append({'engine1': e1, 'engine2': e2, 'p_value': p_val})
        
        # Сохраняем результаты тестов
        pd.DataFrame(results).to_csv('results_full/statistical_tests.csv', index=False)
        print(f"\n💾 Результаты сохранены: results_full/statistical_tests.csv")
    
    def agent_correlation_analysis(self):
        """Анализ корреляции между агентами (исправленная версия)"""
        print("\n" + "="*70)
        print("4. КОРРЕЛЯЦИЯ МЕЖДУ АГЕНТАМИ")
        print("="*70)
        
        # Собираем оценки всех агентов
        agent_scores = {}
        agent_counts = {}
        
        for record in self.full_data:
            for agent, eval_data in record.get('expert_evaluations', {}).items():
                if isinstance(eval_data, dict):
                    # Ищем overall score или любую числовую оценку
                    score_found = None
                    
                    # Сначала ищем по стандартным ключам
                    for key in ['overall_score', 'final_score', 'score', 'overall']:
                        if key in eval_data and isinstance(eval_data[key], (int, float)):
                            score_found = eval_data[key]
                            break
                    
                    # Если не нашли, ищем любое число в значениях
                    if score_found is None:
                        for key, value in eval_data.items():
                            if isinstance(value, (int, float)) and 'score' in key.lower():
                                score_found = value
                                break
                    
                    # Если все еще не нашли, берем первое попавшееся число
                    if score_found is None:
                        for value in eval_data.values():
                            if isinstance(value, (int, float)):
                                score_found = value
                                break
                    
                    if score_found is not None:
                        if agent not in agent_scores:
                            agent_scores[agent] = []
                            agent_counts[agent] = 0
                        agent_scores[agent].append(score_found)
                        agent_counts[agent] += 1
        
        # Показываем статистику по агентам
        print("\n📊 Количество оценок по агентам:")
        for agent, count in sorted(agent_counts.items()):
            print(f"   {agent}: {count} оценок")
        
        # Фильтруем агентов с достаточным количеством данных
        min_samples = 20
        valid_agents = {agent: scores for agent, scores in agent_scores.items() 
                       if len(scores) >= min_samples}
        
        print(f"\n✅ Агентов с >= {min_samples} оценками: {len(valid_agents)}")
        
        if len(valid_agents) < 2:
            print("❌ Недостаточно данных для корреляционного анализа")
            return None
        
        # Создаем DataFrame с выравниванием по длине
        # Берем минимальную длину среди всех агентов
        min_length = min(len(scores) for scores in valid_agents.values())
        
        # Обрезаем все массивы до одинаковой длины
        aligned_scores = {}
        for agent, scores in valid_agents.items():
            aligned_scores[agent] = scores[:min_length]
        
        agent_df = pd.DataFrame(aligned_scores)
        
        print(f"\n📊 Корреляционная матрица агентов (на {min_length} общих оценках):")
        
        # Вычисляем корреляции Пирсона и Спирмена
        pearson_corr = agent_df.corr(method='pearson').round(3)
        spearman_corr = agent_df.corr(method='spearman').round(3)
        
        print("\n📈 Корреляция Пирсона (линейная):")
        print(pearson_corr)
        
        print("\n📈 Корреляция Спирмена (ранговая):")
        print(spearman_corr)
        
        # Сохраняем матрицы
        pearson_corr.to_csv('results_full/agent_correlation_pearson.csv')
        spearman_corr.to_csv('results_full/agent_correlation_spearman.csv')
        print(f"\n💾 Матрицы сохранены в results_full/")
        
        # Находим самые сильные корреляции
        print("\n🔗 Сильные корреляции Пирсона (>0.7 или <-0.7):")
        strong_corr = []
        for i, a1 in enumerate(agent_df.columns):
            for a2 in agent_df.columns[i+1:]:
                corr = agent_df[a1].corr(agent_df[a2])
                if abs(corr) > 0.7:
                    strong_corr.append((a1, a2, corr))
                    print(f"   {a1} & {a2}: {corr:.3f}")
        
        if not strong_corr:
            print("   Сильных корреляций не обнаружено")
        
        # Создаем тепловую карту корреляций
        try:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Пирсон
            sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1, ax=axes[0],
                       cbar_kws={'label': 'Коэффициент корреляции'})
            axes[0].set_title('Корреляция Пирсона (линейная)', fontsize=14, pad=20)
            
            # Спирмен
            sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1, ax=axes[1],
                       cbar_kws={'label': 'Коэффициент корреляции'})
            axes[1].set_title('Корреляция Спирмена (ранговая)', fontsize=14, pad=20)
            
            plt.suptitle('Корреляция между оценками агентов', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig('visualizations/agent_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✅ Тепловая карта сохранена: visualizations/agent_correlation_heatmap.png")
        except Exception as e:
            print(f"⚠️ Не удалось создать тепловую карту: {e}")
        
        return agent_df
    
    def consistency_analysis(self):
        """Анализ согласованности оценок"""
        print("\n" + "="*70)
        print("5. СОГЛАСОВАННОСТЬ ЭКСПЕРТОВ")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score', 'expert_agreement'])
        
        agreement_counts = df_clean['expert_agreement'].value_counts()
        print(f"\n📊 Распределение согласованности:")
        for agreement, count in agreement_counts.items():
            pct = count / len(df_clean) * 100
            print(f"   {agreement}: {count} ({pct:.1f}%)")
        
        # Анализ согласованности по тематикам
        print(f"\n📊 Согласованность по тематикам:")
        for theme in df_clean['theme'].unique():
            theme_data = df_clean[df_clean['theme'] == theme]
            theme_agree = theme_data['expert_agreement'].value_counts()
            print(f"\n   {theme}:")
            for agreement, count in theme_agree.items():
                pct = count / len(theme_data) * 100
                print(f"      {agreement}: {count} ({pct:.1f}%)")
        
        return agreement_counts
    
    def best_worst_examples(self, n=5):
        """Лучшие и худшие примеры переводов"""
        print("\n" + "="*70)
        print(f"6. ТОП-{n} ЛУЧШИХ И ХУДШИХ ПЕРЕВОДОВ")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score'])
        
        # Лучшие
        best = df_clean.nlargest(n, 'final_score')[['segment_id', 'theme', 'engine', 'final_score']]
        print(f"\n🏆 Топ-{n} лучших переводов:")
        print(best.to_string(index=False))
        
        # Худшие
        worst = df_clean.nsmallest(n, 'final_score')[['segment_id', 'theme', 'engine', 'final_score']]
        print(f"\n📉 Топ-{n} худших переводов:")
        print(worst.to_string(index=False))
        
        # Сохраняем в файл
        best.to_csv('results_full/best_translations.csv', index=False)
        worst.to_csv('results_full/worst_translations.csv', index=False)
        print(f"\n💾 Списки сохранены в results_full/")
        
        return best, worst
    
    def create_visualizations(self):
        """Создание графиков для статьи"""
        print("\n" + "="*70)
        print("7. СОЗДАНИЕ ГРАФИКОВ")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score'])
        
        # График 1: Boxplot по тематикам и движкам
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=df_clean, x='theme', y='final_score', hue='engine')
        plt.title('Распределение оценок по тематикам и движкам перевода', fontsize=16, pad=20)
        plt.xlabel('Тематика', fontsize=14)
        plt.ylabel('Финальная оценка', fontsize=14)
        plt.legend(title='Движок')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/boxplot_theme_engine.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ boxplot_theme_engine.png")
        
        # График 2: Heatmap средних оценок
        plt.figure(figsize=(10, 8))
        pivot_mean = df_clean.pivot_table(values='final_score', index='theme', columns='engine', aggfunc='mean')
        sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='RdYlGn', center=8.0,
                   cbar_kws={'label': 'Средняя оценка'})
        plt.title('Средние оценки: тематика × движок', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('visualizations/heatmap_mean_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ heatmap_mean_scores.png")
        
        # График 3: Распределение оценок по движкам
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, engine in enumerate(df_clean['engine'].unique()):
            engine_data = df_clean[df_clean['engine'] == engine]['final_score']
            axes[i].hist(engine_data, bins=15, alpha=0.7, edgecolor='black', color=f'C{i}')
            axes[i].set_title(f'{engine}', fontsize=14)
            axes[i].set_xlabel('Оценка')
            axes[i].set_ylabel('Частота')
            axes[i].axvline(engine_data.mean(), color='red', linestyle='--', 
                          label=f'Среднее: {engine_data.mean():.2f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        plt.suptitle('Распределение оценок по движкам перевода', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('visualizations/histogram_engines.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ histogram_engines.png")
        
        # График 4: Сравнение движков по тематикам (столбцы с ошибками)
        plt.figure(figsize=(14, 8))
        means = df_clean.groupby(['theme', 'engine'])['final_score'].mean().unstack()
        stds = df_clean.groupby(['theme', 'engine'])['final_score'].std().unstack()
        
        x = np.arange(len(means.index))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, engine in enumerate(means.columns):
            plt.bar(x + i*width, means[engine], width, label=engine, 
                   color=colors[i], alpha=0.8, yerr=stds[engine], capsize=5)
        
        plt.xlabel('Тематика', fontsize=14)
        plt.ylabel('Средняя оценка', fontsize=14)
        plt.title('Сравнение движков перевода по тематикам', fontsize=16, pad=20)
        plt.xticks(x + width, means.index)
        plt.legend(title='Движок')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/bar_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ bar_comparison.png")
        
        # График 5: Violin plot (альтернатива boxplot)
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=df_clean, x='theme', y='final_score', hue='engine', split=False)
        plt.title('Распределение оценок (violin plot)', fontsize=16, pad=20)
        plt.xlabel('Тематика', fontsize=14)
        plt.ylabel('Финальная оценка', fontsize=14)
        plt.legend(title='Движок')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/violin_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ violin_plot.png")
        
        print(f"\n✅ Все графики сохранены в папке visualizations/")
    
    def generate_report(self):
        """Генерация полного отчета"""
        print("\n" + "="*70)
        print("📋 ГЕНЕРАЦИЯ ПОЛНОГО ОТЧЕТА")
        print("="*70)
        
        df_clean = self.df.dropna(subset=['final_score'])
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ОТЧЕТ ПО МЕЖАГЕНТНОЙ ОЦЕНКЕ КАЧЕСТВА ПЕРЕВОДА")
        report_lines.append(f"Дата: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        # 1. Общая статистика
        report_lines.append("\n1. ОБЩАЯ СТАТИСТИКА")
        report_lines.append("-"*50)
        report_lines.append(f"Всего оценок: {len(df_clean)}")
        report_lines.append(f"Пропущено значений: {self.df['final_score'].isna().sum()}")
        report_lines.append(f"Средняя оценка: {df_clean['final_score'].mean():.2f} ± {df_clean['final_score'].std():.2f}")
        report_lines.append(f"Медиана: {df_clean['final_score'].median():.2f}")
        report_lines.append(f"Минимум: {df_clean['final_score'].min():.2f}")
        report_lines.append(f"Максимум: {df_clean['final_score'].max():.2f}")
        
        # 2. По тематикам
        report_lines.append("\n2. ПО ТЕМАТИКАМ")
        report_lines.append("-"*50)
        theme_stats = df_clean.groupby('theme')['final_score'].agg(['mean', 'std', 'count'])
        for theme, row in theme_stats.iterrows():
            report_lines.append(f"{theme}: {row['mean']:.2f} ± {row['std']:.2f} (n={int(row['count'])})")
        
        # 3. По движкам
        report_lines.append("\n3. ПО ДВИЖКАМ ПЕРЕВОДА")
        report_lines.append("-"*50)
        engine_stats = df_clean.groupby('engine')['final_score'].agg(['mean', 'std', 'count'])
        for engine, row in engine_stats.iterrows():
            report_lines.append(f"{engine}: {row['mean']:.2f} ± {row['std']:.2f} (n={int(row['count'])})")
        
        # 4. Тематика × движок
        report_lines.append("\n4. ТЕМАТИКА × ДВИЖОК")
        report_lines.append("-"*50)
        pivot = df_clean.pivot_table(values='final_score', index='theme', columns='engine', aggfunc='mean').round(2)
        report_lines.append("\nСредние оценки:")
        report_lines.append(pivot.to_string())
        
        # 5. Лучшие и худшие
        report_lines.append("\n5. ЛУЧШИЕ ПЕРЕВОДЫ")
        report_lines.append("-"*50)
        best = df_clean.nlargest(5, 'final_score')[['segment_id', 'theme', 'engine', 'final_score']]
        report_lines.append(best.to_string(index=False))
        
        report_lines.append("\n6. ХУДШИЕ ПЕРЕВОДЫ")
        report_lines.append("-"*50)
        worst = df_clean.nsmallest(5, 'final_score')[['segment_id', 'theme', 'engine', 'final_score']]
        report_lines.append(worst.to_string(index=False))
        
        # Сохраняем отчет
        report_file = f"results_full/analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✅ Отчет сохранен: {report_file}")
        
        # Также сохраняем в CSV для удобства
        summary_stats = pd.DataFrame({
            'Параметр': ['Всего оценок', 'Средняя оценка', 'Стд отклонение', 'Медиана', 'Минимум', 'Максимум'],
            'Значение': [len(df_clean), f"{df_clean['final_score'].mean():.2f}", 
                       f"{df_clean['final_score'].std():.2f}", f"{df_clean['final_score'].median():.2f}",
                       f"{df_clean['final_score'].min():.2f}", f"{df_clean['final_score'].max():.2f}"]
        })
        summary_stats.to_csv('results_full/basic_statistics.csv', index=False)
        
        return report_lines
    
    def run_all_analyses(self):
        """Запуск всех анализов"""
        print("\n" + "="*80)
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА")
        print("="*80)
        
        self.basic_statistics()
        self.engine_by_theme_analysis()
        self.statistical_tests()
        self.agent_correlation_analysis()
        self.consistency_analysis()
        self.best_worst_examples()
        self.create_visualizations()
        self.generate_report()
        
        print("\n" + "="*80)
        print("✅ ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН")
        print("="*80)
        print("\n📁 Результаты сохранены в папках:")
        print("   - results_full/ (таблицы и отчеты)")
        print("   - visualizations/ (графики)")

def main():
    """Основная функция"""
    try:
        # Инициализация анализа
        analysis = TranslationAnalysis()
        
        # Запуск всех анализов
        analysis.run_all_analyses()
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nУбедитесь, что:")
        print("1. Запущен полный анализ (python main.py с режимом 1)")
        print("2. В папке results_full/ есть файлы с результатами")
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
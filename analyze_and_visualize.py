import re
import argparse
import os

# 1. Парсинг stats-файла

def extract_double_bridge_percent(stat_file):
    print(f"\nПроценты двойных мостов из файла: {stat_file}\n")
    with open(stat_file, encoding='utf-8') as f:
        for line in f:
            m = re.search(r'(Hs_\S+\.has).*?(\d+\.\d+)%', line)
            if m:
                print(f"{m.group(1)}: {m.group(2)}% двойных мостов")

# 2. Генерация и визуализация задачи

def generate_and_visualize(w, h, islands, double_percent, deg7, deg8, alpha, img_out=None, cell_unit=50):
    from core.generator import generate_solvable_puzzle
    from hashiwokakero.hashi.visualiser import draw_grid
    from hashiwokakero.hashi.export import output_image
    print(f"\nГенерируем задачу: {w}x{h}, островов={islands}, двойных мостов={double_percent}%, deg7={deg7}, deg8={deg8}, alpha={alpha}")
    grid = generate_solvable_puzzle(w, h, islands, target_double_bridge_percentage=double_percent, target_degree_7=deg7, target_degree_8=deg8, connectivity_factor_alpha=alpha)
    if img_out:
        ok = output_image(grid, img_out, cell_unit=cell_unit)
        if ok:
            print(f"Картинка сохранена в файл: {img_out}")
        else:
            print(f"Ошибка при сохранении картинки!")
    draw_grid(grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ и визуализация Hashiwokakero.")
    parser.add_argument('--stats', type=str, help='Путь к stats-файлу для парсинга процентов двойных мостов')
    parser.add_argument('--generate', action='store_true', help='Сгенерировать и визуализировать задачу')
    parser.add_argument('--w', type=int, default=16)
    parser.add_argument('--h', type=int, default=16)
    parser.add_argument('--islands', type=int, default=80)
    parser.add_argument('--double_percent', type=int, default=75)
    parser.add_argument('--deg7', type=int, default=2)
    parser.add_argument('--deg8', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=15)
    parser.add_argument('--img_out', type=str, default='output.png', help='Путь для сохранения картинки (PNG/JPG)')
    parser.add_argument('--cell_unit', type=int, default=50, help='Размер клетки в пикселях для картинки')
    args = parser.parse_args()

    if args.stats:
        if os.path.exists(args.stats):
            extract_double_bridge_percent(args.stats)
        else:
            print(f"Файл {args.stats} не найден!")

    if args.generate:
        generate_and_visualize(args.w, args.h, args.islands, args.double_percent, args.deg7, args.deg8, args.alpha, img_out=args.img_out, cell_unit=args.cell_unit) 
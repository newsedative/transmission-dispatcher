'''
считает метрики плана по выводу test_run.py

через stdin принимает вывод test_run.py (строки-приглашения «введите ...» пропускаются):
в первой строке T — число тактов (допускается и «T R»)
далее T наборов по R строк: <номер робота> <вершина, в которую он переходит> <задача, -1 если нет>

вывод:
1. время последней сданной задачи — такт, на котором последний груз в последний раз
   сдвинулся (после этого он лежит на финише)
2. используемость роботов — на каждом такте доля роботов, сменивших вершину,
   усреднённая по всем T тактам

стартовые вершины в выводе test_run.py не печатаются, поэтому берутся как в prepare_data:
робот №i стоит в вершине i - 1
'''

import sys


def read_plan(lines):
    nums = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            nums.append([int(x) for x in parts])
        except ValueError:
            if line.startswith('Решение не найдено'):
                raise SystemExit(line.strip())
            continue  # приглашения ввода и прочий текст

    if not nums:
        raise SystemExit('пустой вывод test_run.py')
    T = nums[0][0]
    rows = nums[1:]
    if T == 0:
        return 0, 0, []
    if len(rows) % T != 0:
        raise SystemExit(f'{len(rows)} строк плана не делятся на T = {T}')
    R = len(rows) // T
    if len(nums[0]) > 1 and nums[0][1] != R:
        raise SystemExit(f'в заголовке R = {nums[0][1]}, а в плане {R} роботов')
    ticks = [rows[t * R:(t + 1) * R] for t in range(T)]
    return T, R, ticks


def compute_metrics(T, R, ticks):
    pos = {r_id: r_id - 1 for r_id, _v, _k in ticks[0]} if ticks else {}
    last_delivery = 0
    used_share_sum = 0.0

    for t, rows in enumerate(ticks, start=1):
        used = 0
        for r_id, v_next, task in rows:
            moved = v_next != pos[r_id]
            if moved:
                used += 1
                if task != -1:
                    last_delivery = max(last_delivery, t)
            pos[r_id] = v_next
        used_share_sum += used / R

    utilization = used_share_sum / T if T else 0.0
    return last_delivery, utilization


def main():
    T, R, ticks = read_plan(sys.stdin.read().splitlines())
    last_delivery, utilization = compute_metrics(T, R, ticks)
    print(f'время последней сданной задачи: {last_delivery}')
    print(f'используемость роботов: {utilization:.4f}')


if __name__ == '__main__':
    main()

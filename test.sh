#!/bin/bash

list_a=(12 30 64 128)
list_b=(12 30 64 128)
list_n=(5 12 20 64)

length=${#list_a[@]}
PYTHON="python"  # или python

for ((i=1; i<=length; i++)); do
    a=${list_a[$i]}
    b=${list_b[$i]}
    n=${list_n[$i]}
    
    output_file="output_${a}_${b}_${n}.txt"
    temp_plan="plan_${a}_${b}_${n}.tmp"
    
    echo "Запуск для n=$a, m=$b, tasks=$n -> $output_file"
    
    # 1. Запуск test_run.py: stdout (план) пишем во временный файл, stderr (/usr/bin/time) сохраняем в переменную
    time_stats=$( { printf "%s %s\n%s\n" "$a" "$b" "$n" | /usr/bin/time -l $PYTHON test_run.py 1> "$temp_plan"; } 2>&1 )
    
    # 2. Расчет метрик через metrics.py
    metrics_result=$($PYTHON metrics.py < "$temp_plan" 2>&1)
    
    # 3. Формирование итогового файла по контракту
    {
        echo "Условия генерации:"
        echo "$a $b $n"
        echo ""
        echo "Метрики:"
        echo "$metrics_result"
        echo ""
        echo "Ресурсы и время:"
        echo "$time_stats"
        echo ""
        echo "Решение:"
        cat "$temp_plan"
    } > "$output_file"
    
    # Удаляем временный файл плана
    rm -f "$temp_plan"
done

echo "Все тесты завершены!"
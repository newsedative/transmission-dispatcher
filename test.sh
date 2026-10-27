#!/bin/bash

list_a=(12 30 64 128)
list_b=(12 30 64 128)
list_n=(5 12 20 64)

length=${#list_a[@]}

# Один цикл по индексам (от 0 до length-1)
for ((i=1; i<=length; i++)); do
    # Берем элементы с одинаковым индексом i
    a=${list_a[$i]}
    b=${list_b[$i]}
    n=${list_n[$i]}
    
    # Формируем имя файла на основе текущих переменных
    output_file="output_${a}_${b}_${n}.txt"
    
    echo "Запуск для a=$a, b=$b, n=$n -> Лог: $output_file"
    
    # Передаем переменные в Python через stdin и сохраняем ВСЁ в файл
    ( echo "$a $b\n$n" | /usr/bin/time -l python test_run.py ) > "$output_file" 2>&1
    echo "\ntask for $a $b $n" >> "$output_file"
    
done

echo "Все тесты завершены!"
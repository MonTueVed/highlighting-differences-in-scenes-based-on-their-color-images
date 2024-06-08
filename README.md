# highlighting-differences-in-scenes-based-on-their-color-images

## Описание
Программе на вход подается два изображения одной и той же сцены, на другом изменены условия съемки (затенение) и, возможно, появился новый объект. На выход подается проекция, являющаяся по сути, усреднением анализируемого изображения по зонно-цветовым особеностям эталонного изображения. Еще на выход подается разность, где проявляются новые объекты, в отличие от теней. Также зная аналитическую часть алгоритма работы, можно видеть в каком месте появился пиксель какого цвета и качественно понимать результат работы алгоритмов сжатия и обработки изображения при делании снимка на камеру.

## Запуск
1) Перейдите в папку проекта
2) Поместите в папку изображения в формате png. Их размер (n*n пикселей) должен совпадать.

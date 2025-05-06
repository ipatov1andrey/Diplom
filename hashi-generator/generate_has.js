const { generate } = require('bridges-generator');
const fs = require('fs');
const path = require('path');

function generatePuzzleInHasFormat(rows, columns, numberOfIslands, doubleBridges = 0.2, outputDir = './output') {
    console.log(`Генерация головоломки ${rows}x${columns}...`);
    
    try {
        // Генерируем головоломку
        const result = generate(rows, columns, numberOfIslands, doubleBridges);
        console.log("Головоломка сгенерирована");
        
        // Создаем имя файла в формате Hs_16_100_75_15_012.has
        const timestamp = new Date().getTime();
        const filename = `Hs_${rows}_${columns}_${numberOfIslands}_${Math.floor(doubleBridges * 100)}_${timestamp % 1000}.has`;
        
        // Создаем директорию, если её нет
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }
        
        // Формируем содержимое файла
        let content = `${rows} ${columns} ${numberOfIslands}\n`;
        
        // Преобразуем матрицу в строки
        for (let i = 0; i < rows; i++) {
            let row = '';
            for (let j = 0; j < columns; j++) {
                // Получаем значение из пазла или 0, если значение отсутствует
                const value = (result.puzzle[i] && result.puzzle[i][j]) || 0;
                row += value.toString().padStart(2, ' ') + ' ';
            }
            content += row.trim() + '\n';
        }
        
        // Записываем файл
        const filePath = path.join(outputDir, filename);
        fs.writeFileSync(filePath, content);
        
        console.log(`Головоломка сохранена в файл: ${filePath}`);
        console.log('Решение:');
        console.log(result.solution);
        
        return {
            filename: filename,
            path: filePath,
            puzzle: result.puzzle,
            solution: result.solution
        };
    } catch (error) {
        console.error("Ошибка при генерации:", error);
        throw error;
    }
}

// Генерация простой тестовой головоломки
console.log("Запуск генератора тестовой головоломки...");
generatePuzzleInHasFormat(20, 20, 50, 0.4); // Небольшая головоломка для теста
console.log("Генерация завершена"); 
const { generate } = require('bridges-generator');

// Функция для генерации головоломки
function generatePuzzle(rows, columns, numberOfIslands, doubleBridges = 0.2) {
    const result = generate(rows, columns, numberOfIslands, doubleBridges);
    
    console.log('Сгенерированная головоломка:');
    console.log(result.puzzle);
    
    console.log('\nРешение:');
    console.log(result.solution);
    
    return result;
}

// Пример использования
const puzzle = generatePuzzle(10, 10, 5, 0.2);
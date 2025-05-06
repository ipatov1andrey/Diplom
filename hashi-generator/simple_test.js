console.log("Тест запущен");
const { generate } = require('bridges-generator');
console.log("Библиотека загружена");

try {
    console.log("Попытка генерации головоломки 5x5...");
    const result = generate(5, 5, 3, 0.2);
    console.log("Головоломка сгенерирована:");
    console.log("Пазл:", result.puzzle);
    console.log("Решение:", result.solution);
} catch (error) {
    console.error("Ошибка при генерации:", error);
    console.error("Стек ошибки:", error.stack);
} 
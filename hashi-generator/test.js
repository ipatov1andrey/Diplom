console.log("Тест запущен");
const { generate } = require('bridges-generator');
console.log("Библиотека загружена");

try {
    const result = generate(5, 5, 3, 0.2);
    console.log("Головоломка сгенерирована:");
    console.log(result);
} catch (error) {
    console.error("Ошибка при генерации:", error);
} 
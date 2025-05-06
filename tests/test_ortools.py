from ortools.sat.python import cp_model

model = cp_model.CpModel()
x = model.NewBoolVar('x')
y = model.NewBoolVar('y')

model.Add(x + y >= 1)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Установка OR-Tools прошла успешно!")
    print(f"x = {solver.Value(x)}")
    print(f"y = {solver.Value(y)}")
else:
    print("Произошла ошибка во время решения. Проверьте установку OR-Tools.")
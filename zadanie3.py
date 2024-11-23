import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
x_sym, y_sym = sp.symbols('x y')

# Define function
f = 2*x_sym**4 - 4*x_sym**2 + y_sym**4 - y_sym**2 - x_sym**2*y_sym**2

# Calculate gradient
grad_f = [sp.diff(f, var) for var in (x_sym, y_sym)]

# Calculate Hessian
hessian_f = [[sp.diff(g, var) for var in (x_sym, y_sym)] for g in grad_f]

# Convert to numpy functions
f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
grad_f_func = sp.lambdify((x_sym, y_sym), grad_f, 'numpy')
hessian_f_func = sp.lambdify((x_sym, y_sym), hessian_f, 'numpy')

def newton_raphson(x0, y0, tol=0.001):
    x = np.array([x0, y0], dtype=float)
    d = np.inf
    i = 0
    path = []

    while d > tol:
        # Calculate gradient and Hessian
        # f_val = f_func(x[0], x[1])
        path.append((x[0], x[1]))
        g = np.array(grad_f_func(x[0], x[1]), dtype=float)
        H = np.array(hessian_f_func(x[0], x[1]), dtype=float)

        # Calculate delta
        d = np.sum(g**2)

        # Check if Hessian is singular
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print(f"Hessian je singulárny v iterácii {i}.")
            break

        # Calculate new x
        delta = H_inv @ g
        x_new = x - delta

        # Update x
        x = x_new
        i += 1

    return x[0], x[1], path

def levenberg_marquardt(x0, y0, tol=0.001, alfa_initial=8, c_initial=4):
    xn, yn = x0, y0
    path = [(xn, yn)]
    alfa = alfa_initial
    c = c_initial
    d = np.inf

    while d > tol:
        f = f_func(xn, yn)
        grad = np.array(grad_f_func(xn, yn), dtype=float).flatten()
        hessian = np.array(hessian_f_func(xn, yn), dtype=float)

        # Calculate next point
        xt = np.array([xn, yn]) - np.linalg.inv(hessian + alfa * np.eye(2)) @ grad
        ft = f_func(xt[0], xt[1])
        gt = np.array(grad_f_func(xt[0], xt[1]), dtype=float).flatten()

        # Calculate delta
        d = np.sum(gt**2)

        # Update alfa
        if ft < f:
            alfa /= c
            xn, yn = xt
            path.append((xn, yn))
        else:
            alfa *= c

    return xn, yn, path

def plot_function_and_path(f_func, path, algorithm_name):
    # Create meshgrid
    X = np.linspace(-2, 2, 400)
    Y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(X, Y)
    Z = f_func(X, Y)

    plt.figure(figsize=(8,6))
    contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # Write path
    path = np.array(path)
    plt.plot(path[:,0], path[:,1], 'r-o', label='Cesta optimalizácie', markersize=4, zorder=1)

    # Plot start and end points
    plt.scatter(path[0,0], path[0,1], color='green', marker='o', label='Začiatočný bod', zorder=2)
    plt.scatter(path[-1,0], path[-1,1], color='blue', marker='o', label='Optimum', zorder=2)

    plt.title(f'Optimalizácia funkcie pomocou {algorithm_name} algoritmu')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

# Main
if __name__ == "__main__":
    print("Vyberte optimalizačný algoritmus:")
    print("1. Newton-Raphson")
    print("2. Levenberg-Marquardt")
    choice = input("Zadajte 1 alebo 2: ").strip()

    while choice not in ['1', '2']:
        print("Neplatná voľba. Skúste to znova.")
        choice = input("Zadajte 1 (Newton-Raphson) alebo 2 (Levenberg-Marquardt): ").strip()

    # Input
    try:
        x0 = float(input("Zadajte počiatočnú hodnotu x: ").strip())
        y0 = float(input("Zadajte počiatočnú hodnotu y: ").strip())
    except ValueError:
        print("Neplatný vstup. Používajú sa predvolené hodnoty x0 = 1.0, y0 = 1.0.")
        x0, y0 = 1.0, 1.0

    try:
        d = float(input("Zadajte toleranciu: ").strip())
    except ValueError:
        print("Neplatný vstup. Používa sa predvolená hodnota tolerancie 0.001.")
        d = 0.001

    if choice == '1':
        algorithm = 'Newton-Raphson'
        xmin, ymin, path = newton_raphson(x0, y0, d)
    else:
        algorithm = 'Levenberg-Marquardt'
        xmin, ymin, path = levenberg_marquardt(x0, y0, d)

    print(f"Optimum nachádza na: x = {xmin}, y = {ymin}")
    print(f"Hodnota funkcie v optimu: f(x,y) = {f_func(xmin, ymin)}")

    # Plot
    plot_function_and_path(f_func, path, algorithm)

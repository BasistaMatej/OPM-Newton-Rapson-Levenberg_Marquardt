# Optimization of a Function of Two Variables using Newton-Raphson and Levenberg-Marquardt Algorithms

This repository contains a Python implementation of optimization techniques for a given two-variable function. The algorithms used are **Newton-Raphson (NR)** and **Levenberg-Marquardt (L-M)**. The code leverages symbolic computation (SymPy) to compute gradients and Hessians, and provides visualization of optimization paths.

---

## Table of Contents

1. [Objective](#objective)
2. [Function Description](#function-description)
3. [Algorithms](#algorithms)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Example Output](#example-output)
7. [Customization](#customization)

---

## Objective

The goal of this project is to optimize the following function:

\[
f(x, y) = 2x^4 - 4x^2 + y^4 - y^2 - x^2y^2
\]

The optimization is performed using:
- Newton-Raphson Algorithm
- Levenberg-Marquardt Algorithm

Both methods aim to find the minimum point of the function, and the optimization path is visualized on a contour plot.

---

## Function Description

The given function is a **nonlinear two-variable function** with several critical points. The objective is to determine the global minimum starting from a user-defined initial point.

Function definition:
\[
f(x, y) = 2x^4 - 4x^2 + y^4 - y^2 - x^2y^2
\]

---

## Algorithms

### 1. Newton-Raphson (NR)
An iterative method that uses the gradient and Hessian of the function to update the optimization variable:
\[
x_{new} = x - H^{-1} \cdot g
\]
Where:
- \( g \) is the gradient vector
- \( H \) is the Hessian matrix

The method stops when the gradient norm (\( ||g||^2 \)) is below a predefined tolerance.

### 2. Levenberg-Marquardt (L-M)
A modification of NR that adds a damping factor to improve robustness. The updated rule is:
\[
x_{new} = x - (H + \lambda I)^{-1} \cdot g
\]
Where \( \lambda \) is a damping factor adjusted dynamically based on the improvement of the function.

---

## Requirements

This project requires the following Python libraries:
- `sympy`
- `numpy`
- `matplotlib`

Install these libraries using pip:
```bash
pip install sympy numpy matplotlib
```

## Usage

1. Clone this repository:
```bash
git clone https://github.com/BasistaMatej/OPM-Newton-Rapson-Levenberg_Marquardt
```
2. Navigate to the project folder:
```bash
cd OPM-Newton-Rapson-Levenberg_Marquardt
```
3. Run the Python script:
```bash
python zadanie3.py
```
4. Select the desired algorithm:
5. Input the initial values for x and y when prompted.
6. Observe the optimization results and the visualization of the optimization path.

## Example Output

**Console Output:**
```
Vyberte optimalizačný algoritmus:
1. Newton-Raphson
2. Levenberg-Marquardt
Zadajte 1 alebo 2: 1
Zadajte počiatočnú hodnotu x: 1.0
Zadajte počiatočnú hodnotu y: 1.0
Konvergencia dosiahnutá po 5 iteráciách.
Optimum nachádza na: x = 0.0, y = 0.0
Hodnota funkcie v optimu: f(x,y) = 0.0
```
**Visualization**

The script produces a contour plot of the function with the optimization path:

- Red line with circles: Optimization path
- Green marker: Starting point
- Blue marker: Optimum point

**Customization**
- Function: Modify the function f in the script to use a different objective.
- Algorithm Parameters:
  - Update tol (tolerance) for convergence criteria.
  - For L-M, modify lambda_initial and lambda_factor to adjust the damping behavior.


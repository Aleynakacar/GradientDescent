#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd

# path : path for getting dataset
path = input("please enter your path for dataset : ")
temp = pd.read_excel(path)
data = list()
for i in temp.values:
    data.append(list(i))
    
    
# GradDescentThree function
def GradDescentThree(data, alpha=1.18, err=10e-5, iteration=1000):
    # alpha : learning rate for Gradient Descent
    # err : epsilon

    # Separating the variables
    X1, X2, X3, Y = [], [], [], []
    for i in data:
        X1.append(i[0])
        X2.append(i[1])
        X3.append(i[1] * i[1])
        Y.append(i[2])

    # Feature Scaling
    X1_range = (max(X1) - min(X1))
    X2_range = (max(X2) - min(X2))
    X3_range = (max(X3) - min(X3))

    i = 0
    while i < len(data):
        X1[i] = (X1[i]) / X1_range
        X2[i] = (X2[i]) / X2_range
        X3[i] = (X3[i]) / X3_range
        i += 1

    # hypothesis for linear regression 
    def h_list_func(q0, q1, q2, q3):
        h_list = []
        for x1, x2, x3 in zip(X1, X2, X3):
            h = q0 + q1 * x1 + q2 * x2 + q3 * x3
            h_list.append(h)
        return h_list

    # Cost function 
    def minimize(q0, q1, q2, q3):
        total = 0
        for y, x1, x2, x3 in zip(Y, X1, X2, X3):
            difference = (q0 + q1 * x1 + q2 * x2 + q3 * x3) - y
            total += difference ** 2
        return total / (2 * len(data))

    # initializing the coefficients of hypothesis
    q0, q1, q2, q3 = 0, 0, 0, 0
    counter = 0

    for iter_ in range(1, iteration + 1):
        counter += 1
        # k : alpha * partial derivative of J(theta) 
        hf = h_list_func(q0, q1, q2, q3)

        sum1, sum2, sum3, sum4 = 0, 0, 0, 0

        for i, j in zip(hf, Y):
            sum1 += (i - j)
        k0 = alpha / 1000 * sum1
        q0 = q0 - k0

        for i, j, x1 in zip(hf, Y, X1):
            sum2 += (i - j) * x1
        k1 = alpha / 1000 * sum2
        q1 = q1 - k1

        for i, j, x2 in zip(hf, Y, X2):
            sum3 += (i - j) * x2
        k2 = alpha / 1000 * sum3
        q2 = q2 - k2

        for i, j, x3 in zip(hf, Y, X3):
            sum4 += (i - j) * x3
        k3 = alpha / 1000 * sum4
        q3 = q3 - k3

        # If all abs k values are smaller than the epsilon(err), the loop will stop

        if abs(k0) < err and abs(k1) < err and abs(k2) < err and abs(k3) < err:
            break

    # running the cost function with coefficients (q0, q1, q2, q3)
    cost_func = minimize(q0, q1, q2, q3)
    
    
    print(
        f"İteration     :  {counter}",
        f"q0            :  {q0}",
        f"q1            :  {q1}",
        f"q2            :  {q2}",
        f"q3            :  {q3}",
        f"Cost Function :  {q0} + ({q1}*x1) + ({q2}*x2) + ({q3}*x3)",
        f"MSE           :  {cost_func}",
        sep="\n"
    )

    return (q0, q1, q2, q3), counter, cost_func


print("\n\n\nModel 3 :\nq0 + q1*x1 + q2*x2 + q3*(x2**2)")
print("\ntoo small alpha for Model3: 0.001")
model3_1 = GradDescentThree(data, alpha=0.001)
print("\nsuitable alpha for Model3 : 1.18")
model3_2 = GradDescentThree(data, alpha=1.18, iteration=5000)
print("\ntoo large alpha for Model3: 1.4")
model3_3 = GradDescentThree(data, alpha=1.4)


# In[ ]:





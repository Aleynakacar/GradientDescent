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


# GradDescentTwo function
def GradDescentTwo(data, alpha=1.24, err=10e-5, iteration=1000):
    # alpha : learning rate for Gradient Descent
    # err : epsilon

    # Separating the variables
    X1, X2, Y = [], [], []
    for i in data:
        X1.append(i[0])
        X2.append(i[1])
        Y.append(i[2])

    # Feature Scaling
    X1_range = (max(X1) - min(X1))
    X2_range = (max(X2) - min(X2))

    i = 0
    while i < len(data):
        X1[i] = (X1[i]) / X1_range
        X2[i] = (X2[i]) / X2_range
        i += 1

    # hypothesis for linear regression 
    def h_list_func(q0, q1, q2):
        h_list = []
        for x1, x2 in zip(X1, X2):
            h = q0 + q1 * x1 + q2 * x2
            h_list.append(h)
        return h_list

    # Cost function 
    def minimize(q0, q1, q2):
        total = 0
        for y, x1, x2 in zip(Y, X1, X2):
            difference = (q0 + q1 * x1 + q2 * x2) - y
            total += difference ** 2
        return total / (2 * len(data))

    # initializing the coefficients of hypothesis
    q0, q1, q2 = 0, 0, 0
    counter = 0

    for iter_ in range(1, iteration + 1):
        counter += 1

        # k : alpha * partial derivative of J(theta) 
        hf = h_list_func(q0, q1, q2)

        sum1, sum2, sum3 = 0, 0, 0

        for i, j in zip(hf, Y):
            sum1 += (i - j)
        k0 = alpha / len(data) * sum1
        q0 = q0 - k0

        for i, j, x1 in zip(hf, Y, X1):
            sum2 += (i - j) * x1
        k1 = alpha / len(data) * sum2
        q1 = q1 - k1

        for i, j, x2 in zip(hf, Y, X2):
            sum3 += (i - j) * x2
        k2 = alpha / len(data) * sum3
        q2 = q2 - k2

        # If all abs k values are smaller than the epsilon(err), the loop will stop

        if abs(k0) < err and abs(k1) < err and abs(k2) < err:
            break

    # running the cost function with coefficients (q0, q1, q2)
    cost_func = minimize(q0, q1, q2)

    print(
        f"İteration     :  {counter}",
        f"q0            :  {q0}",
        f"q1            :  {q1}",
        f"q2            :  {q2}",
        f"Cost Function :  {q0} + ({q1}*x1) + ({q2}*x2)",
        f"MSE           :  {cost_func}",
        sep="\n"
    )

    return (q0, q1, q2), counter, cost_func

print("\n\nModel 2 :\nq0 + q1*x1 + q2*x2")
print("\ntoo small alpha for Model2: 0.001")
model2_1 = GradDescentTwo(data, alpha=0.001)
print("\nsuitable alpha for Model2 : 1.24")
model2_2 = GradDescentTwo(data, alpha=1.24)
print("\ntoo large alpha for Model2: 1.5")
model2_3 = GradDescentTwo(data, alpha=1.5)


# In[ ]:





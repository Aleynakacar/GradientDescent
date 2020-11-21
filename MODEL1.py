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


# GradDescentOne function
def GradDescentOne(data, alpha=1.48, err=10e-5, iteration=1000):
    # alpha : learning rate for Gradient Descent
    # err : epsilon

    # Separating the variables
    X1, Y = [], []
    for i in data:
        X1.append(i[0])
        Y.append(i[2])

    # Feature Scaling
    X1_range = max(X1) - min(X1)
    i = 0
    while i < len(data):
        X1[i] = (X1[i]) / X1_range
        i += 1

    # hypothesis for linear regression 
    def h_list_func(q0, q1):
        h_list = []
        for x1 in X1:
            h = q0 + q1 * x1
            h_list.append(h)
        return h_list

    # Cost function 
    def minimize(q0, q1):
        total = 0
        for y, x1 in zip(Y, X1):
            difference = (q0 + q1 * x1) - y
            total += difference ** 2
        return total / (2 * len(data))

    # initializing the coefficients of hypothesis
    q0, q1 = 0, 0
    counter = 0

    for iter_ in range(1, iteration + 1):
        counter += 1
        # k : alpha * partial derivative of J(theta) 
        hf = h_list_func(q0, q1)

        sum1, sum2 = 0, 0

        for i, j in zip(hf, Y):
            sum1 += (i - j)
        k0 = alpha / 1000 * sum1
        q0 = q0 - k0

        for i, j, x1 in zip(hf, Y, X1):
            sum2 += (i - j) * x1
        k1 = alpha / 1000 * sum2
        q1 = q1 - k1
      
        # If all abs k values are smaller than the epsilon(err), the loop will stop
        if abs(k0) < err and abs(k1) < err:
            break

    # running the cost function with coefficients (q0, q1)
    cost_func = minimize(q0, q1)

    print(
        f"İteration     :  {counter}",
        f"q0            :  {q0}",
        f"q1            :  {q1}",
        f"Cost Function :  {q0} + ({q1}*x1)",
        f"MSE           :  {cost_func}",
        sep="\n"
    )

    return (q0, q1), counter , cost_func

print("\n\n\nModel 1 :\nq0 + q1*x1")
print("\ntoo small alpha for Model1: 0.001")
model1_1 = GradDescentOne(data, alpha=0.001)
print("\nsuitable alpha for Model1 :1.48")
model1_2 = GradDescentOne(data, alpha=1.48)
print("\ntoo large alpha for Model1: 1.7")
model1_3 = GradDescentOne(data, alpha=1.7)
    






# In[ ]:





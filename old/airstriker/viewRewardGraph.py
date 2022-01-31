import csv
import matplotlib.pyplot as plt

x = []
y = []
with open('output.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
        x.append(row[2])
        y.append(row[3])

plt.plot(x, y, label="line L")
plt.plot()

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Graph Example")
plt.legend()
plt.show()
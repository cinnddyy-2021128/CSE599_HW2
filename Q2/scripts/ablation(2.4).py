import matplotlib.pyplot as plt

# Data
dropout = [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3]
steps =  [17910, 8595, 8790, 12555, 11295, 7080, 12345, 9840, 8925, 7680, 7830, 7215, 6555, 6885, 7140, 7005]

# Plot
plt.figure()
plt.plot(dropout, steps, marker='o')
plt.xlabel('Dropout')
plt.ylabel('Steps for test error → 0 after train error vanished')
plt.title('Dropout vs. Grokking Speed on Modulo Division')
plt.show()

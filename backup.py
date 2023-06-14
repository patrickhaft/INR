vmin = - 4
vmax = 5

distance = vmax - vmin

num_weights = 31

weights = np.linspace(vmin, vmax, num_weights) ** 2


distance_per_weight = distance / (num_weights - 1)


x = np.linspace(-7, 7, 1000)

distance_vmin = x - vmin

num_distance = distance_vmin / distance_per_weight
lower_weight_index = np.clip(num_distance, 0, num_weights - 1).astype(int)
upper_weight_index = np.clip(num_distance + 1, 0, num_weights - 1).astype(int) 


y = weights[lower_weight_index] + (num_distance - lower_weight_index) * (weights[upper_weight_index] - weights[lower_weight_index])
plt.scatter(x, y, s=1, marker=".")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Activation')
plt.grid(True)
plt.show()

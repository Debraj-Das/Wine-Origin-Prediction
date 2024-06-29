data = [[38.89, 0, 0.1],
        [38.89, 0, 0.01],
        [38.89, 0, 0.001],
        [38.89, 0, 0.0001],
        [38.89, 0, 1e-05],
        [97.22, 32, 0.1],
        [94.44, 32, 0.01],
        [66.67, 32, 0.001],
        [27.78, 32, 0.0001],
        [75.0, 32, 1e-05],
        [100.0, 64, 0.1],
        [97.22, 64, 0.01],
        [80.56, 64, 0.001],
        [44.44, 64, 0.0001],
        [55.56, 64, 1e-05]]

sorted_data = sorted(data, key=lambda x: x[0], reverse=True)

for item in sorted_data:
    print(item)

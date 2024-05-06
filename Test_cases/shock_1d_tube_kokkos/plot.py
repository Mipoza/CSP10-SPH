import matplotlib.pyplot as plt

# Replace 'pressures_pos.dat' with your actual file path
string_number = '1'

path_to_file_1 = 'velocities/velocities_pos_new'+ string_number+'.dat'
path_to_file_2 = 'densities/densities_pos_new'+ string_number+'.dat'

path_to_file_3 = 'velocities/velocities_pos_no_visc'+string_number+'.dat'
path_to_file_4 = 'densities/densities_pos_no_visc'+string_number+'.dat'

with open(path_to_file_1, 'r') as file:
    data = file.readlines()

with open(path_to_file_2, 'r') as file:
    data2 = file.readlines()

with open(path_to_file_3, 'r') as file:
    data3 = file.readlines()
#
with open(path_to_file_4, 'r') as file:
    data4 = file.readlines()

# Assuming that the first column is x and the second column is y
x_data = [float(row.split()[0]) for row in data]
y_data = [float(row.split()[1]) for row in data]

x_data2 = [float(row.split()[0]) for row in data2]
y_data2 = [float(row.split()[1]) for row in data2]

x_data3 = [float(row.split()[0]) for row in data3]
y_data3 = [float(row.split()[1]) for row in data3]
#
x_data4 = [float(row.split()[0]) for row in data4]
y_data4 = [float(row.split()[1]) for row in data4]

plt.figure()  # Create a new figure for the first plot
plt.plot(x_data, y_data, color='red')
# plt.plot(x_data3, y_data3, color='blue')

plt.figure()  # Create a new figure for the second plot
plt.plot(x_data2, y_data2, color='red')
# plt.plot(x_data4, y_data4, color='blue')

plt.show()  # Show all figures

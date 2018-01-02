import csv
import matplotlib.pyplot as plt

def calculate_moving_average(years, temps):
    if not years:
        return []
    moving_average = [temps[0]]
    summation_temp = temps[0]
    start_year = years[0]
    for idx in range(1,len(years)):
        summation_temp += temps[idx]
        moving_average.append(
            summation_temp / (1 + years[idx] -start_year))
    return moving_average

if __name__ == '__main__':

    cairo_temp = []
    cairo_years = []
    with open('cairo_data.csv') as cairo_data_file:
        reader = csv.DictReader(cairo_data_file)
        for row in reader:
            cairo_temp.append(float(row['avg_temp']))
            cairo_years.append(int(row['year']))

    global_temp = []
    global_years = []
    with open('global_data.csv') as global_data_file:
        reader = csv.DictReader(global_data_file)
        for row in reader:
            # Ignore years that came before the start year of Cairo's data
            cur_year = int(row['year'])
            if cur_year < int(cairo_years[0]):
                continue
            global_temp.append( float(row['avg_temp']))
            global_years.append( int(row['year']))

    global_temp_moving_avg = calculate_moving_average(
        global_years, global_temp)
    cairo_temp_moving_avg = calculate_moving_average(
        cairo_years, cairo_temp)
    for i in range(1,len(cairo_temp_moving_avg)):
        print (
            global_temp_moving_avg[i]-cairo_temp_moving_avg[i])
    global_handle, = plt.plot(
        global_years, global_temp_moving_avg,
        label = 'Global Avg Temp', color = 'blue',
        linewidth = 4)
    cairo_handle, = plt.plot(
        cairo_years, cairo_temp_moving_avg,
        label = 'Cairo Avg Temp', color = 'red',
        linewidth = 4)
    plt.legend(handles = [global_handle, cairo_handle])
    plt.xlabel('year')
    plt.ylabel('Average Temp (C)')
    plt.ylim([5,25])
    plt.title("Comparison between Cairo and Global avg temp")
    plt.show()



def print_time_spent(start_time, end_time, data_info):
    print("Time for " + data_info[0].upper() + " dataset using " + data_info[1].upper() + ": ", end="")
    print(end_time - start_time)
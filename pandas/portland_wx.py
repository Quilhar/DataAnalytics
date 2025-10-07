
def getting_weather_data(text_file):
    dates = []
    prcp = []
    snow = []
    tmax = []
    tmin = []
    
    days_of_year = [i for i in range(1, 366)]
    with open(text_file, 'r') as file:
        weather_data = file.readlines()

    for line in weather_data[2:]:
        station, date, pr, snwd, sn, tx, tn = line.split()
        
        # turn strings into floats or integers as needed
        pr = int(pr)
        sn = int(sn)

        new_date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]  # convert from YYYYMMDD to MM/DD/YYYY
        
        new_pr = int(pr)/25.4  # convert from mm to inches
        new_sn = int(sn)/25.4  # convert from mm to inches
        
        new_pr = format(new_pr, '.2f')
        new_sn = format(new_sn, '.2f')
        
        # correcting the decimal point placement for tmax and tmin and turning them into floats
        tx = tx[:len(tx)-1] + '.' + tx[len(tx)-1:]
        tn = tn[:len(tn)-1] + '.' + tn[len(tn)-1:]  
        tx = float(tx)
        tn = float(tn)
        
        # celsius to fahrenheit conversion
        new_tx = (tx * 9/5) + 32
        new_tn = (tn * 9/5) + 32
        
        dates.append(new_date)
        prcp.append(float(new_pr))
        snow.append(float(new_sn))
        tmax.append(float(new_tx))
        tmin.append(float(new_tn))

        # Turning weather data into dictionaries
        weather_dict = {'Date': dates, 'Precipitation_(inches)': prcp, 'Snowfall_(inches)': snow, 'Max_Temperature (°F)': tmax, 'Min_Temperature (°F)': tmin}

        return weather_dict

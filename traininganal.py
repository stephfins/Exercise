#save the data as a csv w training zones split by amount of time in each training zone in  a training block
#make sure you have installed stuff (via "pip install")

import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import scipy
from datetime import datetime, timedelta


# This first model is a relatively simple multiple linear regression model that should predict event time based on time in zones
# You can change what time in zones it predicts based on in order to try to see what would optimize your training
def simplelinregmodel():
    data = pd.read_csv('training_data.csv')
    print(data)

    data = data.head(1)  # Using head() instead of slicing to get the first row
    print(data)

    model = LinearRegression()

    X = data[['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']]
    y = data['Event Time']

    model.fit(X, y)

    print(f"Coefs: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    # make a prediction for the new block of 32, 47, 11, 9, 2, 4 hours
    new_block = np.array([[32, 47, 11, 9, 2, 4]])
    prediction = model.predict(new_block)
    print(f"Prediction: {prediction}")

def speeddurcurve():
    distances = np.array([100, 200, 400, 800, 1500, 3000, 5000, 10000, 21100, 42200])
    speeds = np.array([37.6, 37.5, 33.5, 28.5, 26.2, 24.5, 23.8, 22.9, 22.0, 20.9])

    def func(x, a, b):
        return a * np.power(x, b)

    # Overall curve
    fatigue_curve, _ = curve_fit(func, distances, speeds, p0=(50, -1.0))
    params = fatigue_curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=distances, y=speeds, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=distances, y=func(distances, *params), mode='lines', name=f'Fit {round(params[0])} * (x**{round(params[1],2)})'))
    fig.update_layout(title='Speed-Distance Curve')
    fig.update_xaxes(title_text='Distance (km)')
    fig.update_yaxes(title_text='Speed (km/h)')
    fig.show()

def get_data():
    # Build dataframe from simple chapter csv
    data = pd.read_csv('banister_data2.csv')
    print(data)
    return data

def optimize_banister(params, data):
    # Defines function that takes parameters in spreadsheet, then retrieves csv data converting columns load and performance into lists
    # Loops through TSS and calcs ban prediction for day... essentially predicting performance based on training load and past levels of training stress markers

    TSS = data['Load'].to_list()
    Performance = data['Performance'].to_list()
    losses = []
    ctls = [70]
    atls = [70]
    for i in range(len(TSS)):
        ctl = (TSS[i] * (1 - math.exp(-1/params[3]))) + (ctls[i] * (math.exp(-1/params[3])))
        atl = (TSS[i] * (1 - math.exp(-1/params[4]))) + (atls[i] * (math.exp(-1/params[4])))
        ctls.append(ctl)
        atls.append(atl)
        Banister_prediction = params[2] + params[0]*ctl - params[1]*atl
        loss = (Performance[i] - Banister_prediction)**2
        losses.append(loss)
    RMSE = np.nanmean(losses)**0.5  # (rms of losses)
    print(f"k1: {params[0]} k2: {params[1]}, Tau1: {params[3]}, Tau2: {params[4]}, P0:{params[2]}, RMSE: {RMSE}")
    return RMSE

# Assuming you have called get_data to get your DataFrame
data = get_data()
initial_guess = [2, 2, 200, 45, 15]
individual_banister_model = minimize(optimize_banister, x0=initial_guess, bounds=[(0, 5), (0, 5), (100, 300), (20, 60), (10, 20)], args=(data,))
params = individual_banister_model['x']

# export entire training history from TrainingPeaks and use efficiency factor (power/HR) as performance metric
# lets you see how load relates to performance

def calc_vo2_if_bike(row, max_hr, resting_hr, weight):
    # Calculating estimated vo2 max from trainingpeaks csv export
    # Will calculate estimated vo2 from each ride
    if row['WorkoutType'] == 'Bike':
        percent_vo2 = (row['HeartRateAverage'] - resting_hr) / (max_hr - resting_hr)
        vo2_power = row['PowerAverage'] / percent_vo2
        vo2_estimated = (((vo2_power) / 75) * 1000) / weight
        return vo2_estimated

def optimize_banister_tsb(params, data):
    # Defines function that takes parameters in spreadsheet, then retrieves csv data converting columns load and performance into lists
    # Loops through TSS and calcs ban prediction for day... essentially predicting performance based on training load and past levels of training stress markers

    TSS = data['Load'].to_list()
    Performance = data['Performance'].to_list()
    losses = []
    ctls = [70]
    atls = [70]
    tsbs = [0]
    for i in range(len(TSS)):
        ctl = (TSS[i] * (1 - math.exp(-1/params[3]))) + (ctls[i] * (math.exp(-1/params[3])))
        atl = (TSS[i] * (1 - math.exp(-1/params[4]))) + (atls[i] * (math.exp(-1/params[4])))
        tsb = ctl - atl
        ctls.append(ctl)
        atls.append(atl)
        tsbs.append(tsb)
        Banister_prediction = params[2] + params[0] * ctl + params[1] * tsb
        loss = (Performance[i] - Banister_prediction) ** 2
        losses.append(loss)
    RMSE = np.nanmean(losses) ** 0.5  # (rms of losses)
    print(f"k1: {params[0]} k2: {params[1]}, Tau1: {params[3]}, Tau2: {params[4]}, P0:{params[2]}, RMSE: {RMSE}")
    return RMSE

# Assuming you have called get_data to get your DataFrame
data = get_data()
initial_guess = [2, 2, 200, 45, 15]
individual_banister_model = minimize(optimize_banister_tsb, x0=initial_guess, bounds=[(0, 5), (0, 5), (100, 300), (20, 60), (10, 20)], args=(data,))
params = individual_banister_model['x']

# Athletes with higher Tau2 benefit from longer taper 

# Calculate nutritional needs for athlete, reports what one should do for daily carb intake 
def calculate_cho(threshold, intensity_factor, planned_hrs):
    if threshold <= 200:
        cho = 10
    elif threshold <= 240:
        cho = 11
    elif threshold <= 270:
        cho = 12
    elif threshold <= 300:
        cho = 13
    elif threshold <= 330:
        cho = 14
    elif threshold <= 360:
        cho = 15
    else:
        cho = 16
    tss = (intensity_factor**2) * 100 * planned_hrs
    cho_calories = tss * cho
    cho_grams = round(cho_calories / 4)
    return cho_grams

# Calculating Protein needs according to size, training volume and energy balance
def calculate_pro(weight_kg, planned_hours, weight_loss='no'):
    if weight_loss in ['yes', 'gain']:
        pro = 1
    elif planned_hours < 1:
        pro = 0.7
    elif planned_hours < 2:
        pro = 0.8
    elif planned_hours < 2.5:
        pro = 0.9
    else:
        pro = 1
    pro_grams = round((weight_kg * 2.2) * pro)
    return pro_grams

# Estimate basal metabolic rate
def calculate_bmr(weight_kg, height_cm, age, sex):
    adj_factor = 5 if sex == 'male' else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + adj_factor
    return bmr

# Estimate incidental energy expenditure (sedentary = office-> couch, mod active --> lots of incidental activity (10k steps outside of exercise), extremely active = physically demanding job)
def calculate_iee(bmr, activity_level):
    activity_levels = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }
    iee = bmr * activity_levels.get(activity_level, 1.0)
    return iee

# Calculate cycling calorie expenditure
def calculate_cycling_expenditure(power, duration_hours, economy=75):
    liters_o2_per_min = power / economy
    kcals_per_min = liters_o2_per_min * 5
    kcals = kcals_per_min * duration_hours * 60
    return kcals

# Calculate running calorie expenditure
def calculate_running_expenditure(pace_min_km, duration_hours, weight_kg, economy=210):
    try:
        # Calculate total oxygen consumption in liters per minute
        # As (O2 cost per km/pace) to get O2 cost/min
        # Then mult by weight to get ml/min and divide by 1000 to get Liters
        liters_o2_per_min = ((economy / pace_min_km) * weight_kg) / 1000
        
        # Calculate energy cost in kilocalories per minute (5kcal per L O2)
        kcals_per_min = liters_o2_per_min * 5
        
        # Calculate total energy cost for the entire duration
        kcals = kcals_per_min * duration_hours * 60
        
        return kcals
    except ZeroDivisionError:
        # Handle the case where pace_min_km is 0 to avoid division by zero
        return 0

# Assuming a standard economy of 210 ml/kg/km
# You can replace 210 with your own tested economy if available

# calculate_bmr(58, 170, 23, 'female')
# calculate_iee(1700, 'sedentary')
# calculate_running_expenditure(5, 1, 58) # running for 1 hr at 5 min per kilometer weighting 58kg
# calculate_cycling_expenditure(200, 2) # riding for 2hrs at 200 norm power
# calculate_cho(300, 0.67, 3)
#     #for 3hrs combined training at an I.F. of 0.67 for an athlete with a cycling threshold of 300W
#     # We should be shooting for something around 438g of carbohydrate for the day. 
#     # This amounts to 1752 kcal or ~40% of the total energy needs of 4463kcal
# calculate_pro(58, 3)

#code to get a nutrition plan for an athlete
def get_my_nutrition_plan(weight_kg, height_cm, age, sex, activity_level, bike_hrs, bike_power, bike_threshold, run_hrs, run_pace, run_threshold, weight_loss='no'):
    bmr = calculate_bmr(weight_kg, height_cm, age, sex)
    iee = calculate_iee(bmr, activity_level)
    if weight_loss == 'yes':
        iee -= 500
    elif weight_loss == 'gain':
        iee += 500
    bike_kcal = calculate_cycling_expenditure(bike_power, bike_hrs)
    run_kcal = calculate_running_expenditure(run_pace, run_hrs, weight_kg)
    total_kcal = iee + bike_kcal + run_kcal
    try:
        bike_intensity_factor = bike_power / bike_threshold
    except ZeroDivisionError:
        bike_intensity_factor = 0
    try:
        run_intensity_factor = run_threshold / run_pace
    except ZeroDivisionError:
        run_intensity_factor = 0
    average_intensity_factor = ((bike_intensity_factor * bike_hrs) + (run_intensity_factor * run_hrs)) / (bike_hrs + run_hrs)
    total_hrs = bike_hrs + run_hrs
    cho = calculate_cho(bike_threshold, average_intensity_factor, total_hrs)
    pro = calculate_pro(weight_kg, total_hrs, weight_loss)
    fat = round((total_kcal - (cho * 4) - (pro * 4)) / 9)
    nutrition_plan = {'Total Calories': round(total_kcal), 'CHO': cho, 'Protein': pro, 'Fat': fat}
    return nutrition_plan

get_my_nutrition_plan(58, 170, 23, 'female', 'sedentary', 0, 0, 0, 2, 6.9, 4, 'no')

# Input variables time in zone (minutes) for Z1, Z2, Z3, Z4, Z5 per month
# Note: This could also be done from a csv file (e.g. from Training Peaks) using pd.read_csv

def get_block_end(start, block_length):
    end = start + timedelta(days=block_length)
    return end

def sum_TIZ_between_2_dates(data, col, date1, date2):
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    total = data.loc[(data.WorkoutDay >= date1) & (data.WorkoutDay <= date2), col].sum()
    return total

def get_input_data_from_TrainingPeaks(file, block_length_days):
    data = pd.read_csv(file)
    data['WorkoutDay'] = pd.to_datetime(data['WorkoutDay'])
    start = data['WorkoutDay'].values[0]
    end = data['WorkoutDay'].values[-1]
    days = (end - start) / np.timedelta64(1, 'D')
    blocks = int(days / block_length_days)
    block_dates = [start + np.timedelta64(i * block_length_days, 'D') for i in range(blocks + 1)]

    intensities = ['HRZone1Minutes', 'HRZone2Minutes', 'HRZone3Minutes', 'HRZone4Minutes', 'HRZone5Minutes', 'HRZone6Minutes', 'HRZone7Minutes']
    input_data = []
    
    for i in range(len(block_dates) - 1):  # Adjusted the loop range to avoid IndexError
        block_TIZ = []
        for intensity in intensities:
            total_TIZ = sum_TIZ_between_2_dates(data, intensity, block_dates[i], block_dates[i + 1])
            block_TIZ.append(total_TIZ)
        input_data.append(block_TIZ)

    return input_data

def main():
    # Input Data for the model - will look for a workouts.csv file from Training Peaks.
    # If not found, resorts to your manual entry of TIZ by block
    try:
        time_in_zone_by_month = get_input_data_from_TrainingPeaks('workouts.csv', 28)
    except FileNotFoundError:
        time_in_zone_by_month = [
            [1205, 902, 330, 48, 20],
            [1303, 1021, 371, 69, 19],
            [1370, 1311, 380, 53, 24],
            [1389, 1330, 391, 118, 18],
            [1333, 1291, 458, 109, 28],
        ]

    # Output variable - tested FTP each month (replace with your own data)
    FTP_by_month = [248, 279, 295, 310, 303]

    X = time_in_zone_by_month
    y = FTP_by_month

    model = train_model(X, y)

if __name__ == '__main__':
    main()

#save the data as a csv w training zones split by amount of time in each training zone in  a training block
#make sure you have installed pandas, python, and scikit-learn (via "pip install")

import pandas as pandas
from sklearn.linear_model import LinearRegression
import plotly
import numpy as np
import plotly.graph_objects as go
from cipy.optimize import curve_fit
import scipy


# This first model is a relatively simple multiple linear regression model that should predict event time based on time in zones
# You can change what time in zones it predicts based on in order to try to see what would optimize your training
data = pd.read_csv('training_data.csv')
print(data)

data = data[:1]
print(data)

model = LinearRegression()

X = data[['Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5']]

y = data['Event Time']

model.fit(X, y)

print(f"Coefs: {model.coef_}'")
print(f"Intercept: {model.intercept_}")

# make a prediction for the new block of 32, 47, 11, 9, 2, 4 hours
print(f"Prediction: {model.predict([[32, 47, 11, 9, 2, 4]])}")

#this second model is modeling the speed-duration curve

# distances = np.array([100, 200, 400, 800, 1500, 3000, 5000, 10000, 21100, 42200])

# speeds = np.array([37.6, 37.5, 33.5, 28.5, 26.2, 24.5, 23.8, 22.9, 22.0, 20.9])

# def func(x, a, b):
#     return a * np.power(x, b)

# #overall curve

# fatigue_curve = curve_fit(func, distances, speeds, p0=(50, -1.0))

# params = fatigue_curve[0]

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=distances, y=speeds, mode='markers', name='Data'))

# fig.add_trace(go.Scatter(x=distances, y=func(distances, *params), mode='lines', name=f'Fit {round(params[0])} * (x**{round(params[1],2)}'))
# fig.update_layout(title='Speed-Distance Curve')

# fig.update_xaxes(title_text='Distance (km)')
# fig.update_yaxes(title_text='Speed (km/h)')

# fig.show()

def get_data():
    #Build dataframe from simple chapter csv
    data = pd.read_csv('banister_data2.csv')
    print(data)
    return data

def optimize_banister(params):
#defines function that takes parameters in spreadsheet, then retreives csv data converting columns load and performance into lists
# loops through TSS and calcs ban prediction for day... essentially predicting performance based on training load and past levels of training stress markers


    TSS = data['Load'].to_list()
    Performance = data['Performance'].to_list()
    losses = []
    ctls = [70]
    atls = [70]
    for i in range(len(TSS)):
        ctl = (TSS[i] * (1-math.exp(-1/params[3]))) + (ctls[i] * (math.exp(-1/params[3])))
        atl = (TSS[i] * (1-math.exp(-1/params[4]))) + (atls[i] * (math.exp(-1/params[4])))
        ctls.append(ctl)
        atls.append(atl)
        Banister_prediction = params[2] + params[0]*ctl - params[1]*atl
        loss = (Performance[i]- Banister_prediction)**2
        losses.append(loss)
    RMSE = np.nanmean(losses)**0.5 #(rms of losses)
    print(f"k1: {params[0]} k2: {params[1]}, Tau1: {params[3]}, Tau2: {params[4]}, P0:{params[2]}, RMSE: {RMSE}")
    return RMSE

initial_guess = [2, 2, 200, 45, 15]
individual_banister_model = optimize.minimize(optimize_banister, x0=initial_guess, bounds=[(0,5), (0,5), (100,300), (20,60), (10,20)])
params = individual_banister_model['x']

# export entire training history from TrainingPeaks and use efficiency factor (power/HR) as performance metric
# lets you see how load relates to performance

def calc_vo2_if_bike(row, max_hr, resting_hr, weight):
    #calculating estiamted vo2 max from trainingpeaks csv export
    #will calc estimated vo2 from each ride
    if row['WorkoutType'] == 'Bike':
        percent_vo2 = (row['HeartRateAverage'] - resting_hr)/(max_hr - resting_hr)
        vo2_power = row['PowerAverage'] / percent_vo2
        vo2_estimated = (((vo2_power)/75)*1000)/weight
        return v02_estimated


def optimize_banister_tsb(params):
#defines function that takes parameters in spreadsheet, then retreives csv data converting columns load and performance into lists
# loops through TSS and calcs ban prediction for day... essentially predicting performance based on training load and past levels of training stress markers

    TSS = data['Load'].to_list()
    Performance = data['Performance'].to_list()
    losses = []
    ctls = [70]
    atls = [70]
    tsbs=[0]
    for i in range(len(TSS)):
        ctl = (TSS[i] * (1-math.exp(-1/params[3]))) + (ctls[i] * (math.exp(-1/params[3])))
        atl = (TSS[i] * (1-math.exp(-1/params[4]))) + (atls[i] * (math.exp(-1/params[4])))
        tsb = ctl - atl
        ctls.append(ctl)
        atls.append(atl)
        tsbs.append(tsb)
        Banister_prediction = params[2] + params[0]*ctl + params[1]*tsb
        loss = (Performance[i]- Banister_prediction)**2
        losses.append(loss)
    RMSE = np.nanmean(losses)**0.5 #(rms of losses)
    print(f"k1: {params[0]} k2: {params[1]}, Tau1: {params[3]}, Tau2: {params[4]}, P0:{params[2]}, RMSE: {RMSE}")
    return RMSE

initial_guess = [2, 2, 200, 45, 15]
individual_banister_model = optimize.minimize(optimize_banister_tsb, x0=initial_guess, bounds=[(0,5), (0,5), (100,300), (20,60), (10,20)])
params = individual_banister_model['x']

# athletes with higher Tau2 benefit from longer taper 

# Calculate nutritional needs for athlete, reports what one should do for daily carb intake 
def calculate_cho(threshold, intensity_factor, planned_hrs)
    if threshold <= 200:
        cho = 10
    elif threshold <= 240:
        cho = 11
    elif threshold <= 270:
        cho = 12
    elif threshold <= 300:
        cho = 13
    elif threshold <=330:
        cho = 14
    elif threshold <=360:
        cho = 15
    else:
        cho = 16
    tss = (intensity_factor**2) * 100 * planned_hrs
    cho_calories = tss * cho
    cho_grams = round(cho_calories/4)
    return cho_grams

# Calculating Protein needs according to size, training volume and energy balance
def calculate_pro(weight_kg, planned_hours, weight_loss=’no’):    
    if (weight_loss == 'yes') or (weight_loss == 'gain'):
        pro = 1
    elif planned_hours<1:
        pro = 0.7
    elif planned_hours<2:
        pro = 0.8
    elif planned_hours<2.5:
        pro = 0.9
    else:
        pro = 1
    pro_grams = round((weight_kg * 2.2) * pro)

    #estimate basal metabolic rate
def calculate_bmr(weight_kg, height_cm, age, sex):
    if sex == ‘male’:
        adj_factor = 5
    if sex == ‘female’:
        adj_factor = -161 
    bmr = ((10*(weight_kg))+(6.25*(height_cm))- (5*age) + adj_factor)
    return bmr

# estimate incidental energy expenditure (sedentary = office-> couch, mod active --> lots of incidental activity (10k steps outside of exercise), extremely active = physically demanding job)
def calculate_iee(bmr, activity_level):
    if activity_level == 'sedentary':
        iee = bmr * 1.2
    if activity_level == 'lightly_active':
        iee = bmr * 1.375
    if activity_level == 'moderately_active':
        iee = bmr * 1.55
    if activity_level == 'very_active':
        iee = bmr * 1.725
    if activity_level == 'extra_active':
        iee = bmr * 1.9
    return iee  

# calc cycling calorie expenditure
def calculate_cycling_expenditure(power, duration_hours, economy=75):
    liters_o2_per_min = power/economy
    kcals_per_min = liters_o2_per_min * 5
    kcals = kcals_per_min * duration_hours * 60
    return kcals

# calc running calorie expenditure
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
    if weight_loss == ‘yes’:
        iee = iee – 500
    if weight_loss == ‘gain’:
        iee = iee + 500
    bike_kcal = calculate_cycling_expenditure(bike_power, bike_hrs)
    run_kcal = calculate_running_expenditure(run_pace, run_hrs, weight_kg)
    total_kcal = iee + bike_kcal + run_kcal
    try:
        bike_intensity_factor = bike_power/bike_threshold
    except ZeroDivisionError:
        bike_intensity_factor = 0
    try:
        run_intensity_factor = run_threshold/run_pace
    except ZeroDivisionError:
        run_intensity_factor = 0
    average_intensity_factor = ((bike_intensity_factor * bike_hrs) + (run_intensity_factor * run_hrs))/(bike_hrs + run_hrs)
    total_hrs = bike_hrs + run_hrs
    cho = calculate_cho(bike_threshold, average_intensity_factor, total_hrs)
    pro = calculate_pro(weight_kg, total_hrs, weight_loss)
    fat = round((total_kcal - (cho*4) - (pro*4))/9)
    nutrition_plan = {'Total Calories': round(total_kcal), 'CHO': cho, 'Protein': pro, 'Fat': fat}
    return nutrition_plan
get_my_nutrition_plan(58, 170, 23, 'female', 'sedentary', 0, 0, 0, 2, 6.9, 4, 'no')
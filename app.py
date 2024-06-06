from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  


file_path = 'Top Indian Places to Visit.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

data = pd.read_csv(file_path)
columns_to_drop = ['Unnamed: 0', 'Establishment Year', 'Airport with 50km Radius', 'Weekly Off', 'DSLR Allowed', 'Number of google review in lakhs']
data = data.drop(columns=columns_to_drop)


encoders_path = 'label_encoders.joblib'
if os.path.exists(encoders_path):
    label_encoders = joblib.load(encoders_path)
else:
    label_encoders = {}

categorical_columns = ['Zone', 'State', 'City', 'Name', 'Type', 'Significance', 'Best Time to visit']
for column in categorical_columns:
    if column in label_encoders:
        le = label_encoders[column]
        data[column] = le.transform(data[column])
    else:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le


joblib.dump(label_encoders, encoders_path)

def display_valid_options(column_name):
    if column_name in label_encoders:
        return list(label_encoders[column_name].classes_)
    else:
        return []

@app.route('/')
def index():
    zone_options = display_valid_options('Zone')
    type_options = display_valid_options('Type')
    return render_template('index.html', zone_options=zone_options, type_options=type_options)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    zone_input = request.form['zone']
    type_input = request.form['type']

    zone_options = display_valid_options('Zone')
    type_options = display_valid_options('Type')

    if zone_input not in zone_options:
        flash(f"Invalid zone input: {zone_input}. Valid options are: {zone_options}")
        return redirect(url_for('index'))
    if type_input not in type_options:
        flash(f"Invalid type input: {type_input}. Valid options are: {type_options}")
        return redirect(url_for('index'))

    zone_encoded = label_encoders['Zone'].transform([zone_input])[0]
    type_encoded = label_encoders['Type'].transform([type_input])[0]

    filtered_data = data[(data['Zone'] == zone_encoded) & (data['Type'] == type_encoded)]

    decoded_data = filtered_data.copy()
    for column in decoded_data.columns:
        if column in label_encoders:
            decoded_data[column] = label_encoders[column].inverse_transform(decoded_data[column])

    if decoded_data.empty:
        return render_template('results.html', results=None)

    results = decoded_data.to_dict(orient='records')
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

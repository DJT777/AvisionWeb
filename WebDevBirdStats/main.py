from flask import Flask, redirect, url_for, render_template, request, jsonify
import pymysql
import json
import pandas as pd
import numpy as np
from flask_ngrok import run_with_ngrok


app = Flask(__name__, static_url_path='/content/AvisionWeb/WebDevBirdStats/static')
run_with_ngrok(app)


from scipy.stats import norm
def perform_proportion_test(successes, total, null_hypothesis, alternative, confidence):
    # Calculate the sample proportion
    sample_proportion = successes / total

    # Calculate the standard error of the sample proportion
    standard_error = np.sqrt((null_hypothesis * (1 - null_hypothesis)) / total)

    # Calculate the z-score
    z_statistic = (sample_proportion - null_hypothesis) / standard_error

    # Calculate the critical value for the given confidence interval
    confidence_critical_value = norm.ppf(1 - confidence/2)

    # Calculate the critical value for the given confidence level
    if confidence == 0.1:
        critical_value = 1.2816
    elif confidence == 0.05:
        critical_value = 1.6449
    elif confidence == 0.01:
        critical_value = 2.3263

    # Determine if we reject the null hypothesis
    if alternative == "Less Than":
        critical_value = -critical_value
        reject_null_hypothesis = z_statistic < critical_value
    elif alternative == "Greater Than":
        reject_null_hypothesis = z_statistic > critical_value
    else:
        critical_value = norm.ppf(1 - confidence/2)
        reject_null_hypothesis = abs(z_statistic) > critical_value

    # Calculate the p-value
    if alternative == "Less Than":
        p_value = norm.cdf(z_statistic)
    elif alternative == "Greater Than":
        p_value = 1 - norm.cdf(z_statistic)
    else:
        p_value = 2 * (1 - norm.cdf(abs(z_statistic)))


    # Calculate the lower and upper bounds
    margin_of_error = confidence_critical_value * standard_error
    lower_bound = sample_proportion - margin_of_error
    upper_bound = sample_proportion + margin_of_error

    return z_statistic, p_value, critical_value, confidence_critical_value, reject_null_hypothesis, lower_bound, upper_bound


import numpy as np
from scipy.stats import norm

def perform_two_proportion_test(successes1, successes2, total1, total2, null_hypothesis, alternative, confidence):
    # Calculate the sample proportions
    sample_proportion1 = successes1 / total1
    sample_proportion2 = successes2 / total2

    # Calculate the pooled proportion
    pooled_proportion = (successes1 + successes2) / (total1 + total2)

    # Calculate the standard error of the difference in sample proportions
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * ((1 / total1) + (1 / total2)))

    # Calculate the z-score
    z_statistic = (sample_proportion1 - sample_proportion2 - null_hypothesis) / standard_error

    # Calculate the critical value for the given confidence level
    if confidence == 0.1:
        critical_value = 1.2816
    elif confidence == 0.05:
        critical_value = 1.6449
    elif confidence == 0.01:
        critical_value = 2.3263

    # Determine if we reject the null hypothesis
    if alternative == "Less Than":
        reject_null_hypothesis = z_statistic < -critical_value
    elif alternative == "Greater Than":
        reject_null_hypothesis = z_statistic > critical_value
    else:
        critical_value = norm.ppf(1 - confidence/2)
        reject_null_hypothesis = abs(z_statistic) > critical_value

    # Calculate the p-value
    if alternative == "Less Than":
        p_value = norm.cdf(z_statistic)
    elif alternative == "Greater Than":
        p_value = 1 - norm.cdf(z_statistic)
    else:
        p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    # Calculate the lower and upper bounds
    confidence_critical_value = norm.ppf(1 - confidence/2)
    margin_of_error = confidence_critical_value * standard_error
    lower_bound = (sample_proportion1 - sample_proportion2) - margin_of_error
    upper_bound = (sample_proportion1 - sample_proportion2) + margin_of_error

    return z_statistic, p_value, critical_value, confidence_critical_value, reject_null_hypothesis, lower_bound, upper_bound




@app.route("/histogram", methods=['GET'])
def data():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('/content/AvisionWeb/WebDevBirdStats/static/random1000classified.csv')

    # Extract confidences from the dataset
    confidences = df['CONFIDENCE'].astype(float)

    # Take 100 random samples of confidences and sum them
    samples = []
    for i in range(5000):
        sample = np.random.choice(confidences, 100)
        list_sample_sum = sample.sum()
        samples.append([list_sample_sum.tolist()])

    min = np.min(samples)

    # Return the histogram data as JSON
    return render_template("histogram.html", histogramData=samples, min=min)


@app.route("/hypothesis", methods=['GET', 'POST'])
def hypothesis():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv('/content/AvisionWeb/WebDevBirdStats/static/random1000classified.csv')
    # Get the unique species values
    species_values = df['SPECIES'].unique().tolist()


    # Handle the POST request from the Single Test button
    if request.method == 'POST' and 'submit_single_test' in request.form:
        species1 = request.form['species1']
        confidence1 = float(request.form['confidence1'])
        alternative1 = request.form['alternative1']
        null_hypothesis = float(request.form["null_hypothesis1"])

        # Filter the DataFrame to include only rows for the selected species
        subset = df.loc[df['SPECIES'] == species1]

        # Calculate the number of successes (classified) and total trials
        successes = len(subset)

        #calculate number of rows in dataset
        total = len(df)


        # Perform the proportion test using the selected values
        z_statistic, p_value, critical_value, confidence_critical_value, reject_null_hypothesis, lower_bound, upper_bound = perform_proportion_test(successes, total, null_hypothesis, alternative1, confidence1)

        return render_template("hypothesis.html", species_values=species_values, z_statistic=z_statistic,
                               p_value=p_value, critical_value=critical_value, confidence_critical_value=confidence_critical_value,
                               reject_null_hypothesis=reject_null_hypothesis, lower_bound=lower_bound,
                               upper_bound=upper_bound)

    # Handle the POST request from the Two Test button
    if request.method == 'POST' and 'submit_two_test' in request.form:
        species1 = request.form['species1']
        species2 = request.form['species2']
        confidence2 = float(request.form['confidence2'])
        alternative2 = request.form['alternative2']
        null_hypothesis2 = float(request.form["null_hypothesis2"])

        # Filter the DataFrame to include only rows for the selected species
        subset1 = df.loc[df['SPECIES'] == species1]
        subset2 = df.loc[df['SPECIES'] == species2]

        # Calculate the number of successes (classified) and total trials for each subset
        successes1 = len(subset1)
        successes2 = len(subset2)

        # calculate number of rows in dataset
        total = len(df)

        # Calculate the proportions for each subset
        proportion1 = successes1 / total
        proportion2 = successes2 / total

        # Perform the proportion test using the selected values
        z_statistic, p_value, critical_value, confidence_critical_value, reject_null_hypothesis, lower_bound, upper_bound = perform_two_proportion_test(
            successes1, successes2, total, total, null_hypothesis2, alternative2, confidence2)

        return render_template("hypothesis.html", species_values=species_values, z_statistic=z_statistic,
                               p_value=p_value, critical_value=critical_value, confidence_critical_value=confidence_critical_value,
                               reject_null_hypothesis=reject_null_hypothesis, lower_bound=lower_bound,
                               upper_bound=upper_bound)

    # Render the HTML template and pass the unique species values as a list
    return render_template("hypothesis.html", species_values=species_values)

@app.route("/", methods=['GET'])
def home():
    return render_template("base.html")



@app.route("/sum")
def sum():
    totalCount = 0
    species = request.args.get("species")
    with open('/home/dylan/Desktop/repos/AvisionWeb/WebDevBirdStats/static/use_json_data.json', 'r') as f:
        data = json.load(f)
    if species == "all":
        for item in data:
            totalCount += item["COUNT(*)"]
    else:
        totalCount = next((item["COUNT(*)"] for item in data if item["SPECIES"] == species), 0)
    return render_template("sum.html", species=species, sum=totalCount)

import pandas as pd


# Assuming you have a pandas dataframe named "df" containing your dataset
@app.route("/pie", methods=['GET'])
def pie():
    df = pd.read_csv('/content/AvisionWeb/WebDevBirdStats/static/random1000classified.csv')

    chart_data = [("SPECIES", "COUNT(*)")]
    for species, count in df.groupby("SPECIES")["PRIMARY_KEY"].count().items():
        chart_data.append((species, count))

    return render_template("pie.html", json_file=json.dumps(chart_data))

@app.route("/bar", methods=['GET'])
def bar():
    df = pd.read_csv('/content/AvisionWeb/WebDevBirdStats/static/random1000classified.csv')

    chart_data = [("SPECIES", "COUNT(*)")]
    for species, count in df.groupby("SPECIES")["PRIMARY_KEY"].count().items():
        chart_data.append((species, count))

    return render_template("bar.html", json_file=json.dumps(chart_data))


@app.route("/datePie", methods=['GET'])
def sums():
    chart_data = [("DATE", "COUNT(*)")]
    for item in output:
        chart_data.append((item["DATE"].strftime("%Y-%m-%d"), item["COUNT(*)"]))
    return render_template("datepie.html", json_file=json.dumps(chart_data))

@app.route("/confPie", methods=['GET'])
def conf():

    output = mysqlconnectConf()
    chart_data = [("ROUNDEDCONFIDENCE", "COUNT")]
    for item in output:
        chart_data.append((str(item["ROUNDEDCONFIDENCE"]), item["COUNT"]))
    return render_template("bar.html", json_file=json.dumps(chart_data))




if __name__ == '__main__':
    app.run()

if __name__ == "__main__":
    app.run(debug=True)

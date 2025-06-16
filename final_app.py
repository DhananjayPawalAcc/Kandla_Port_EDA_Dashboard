from flask import Flask, render_template, jsonify, request
import pandas as pd 
import joblib
import requests
import markdown2

app= Flask(__name__)

PORT_CSV_FILE= "response_portcall_269_simple.csv"
BERTH_CSV_FILE= "response_berthcall_269_kandla.csv"

def load_port_data():
    df=pd.read_csv(PORT_CSV_FILE, on_bad_lines='skip')
    df.columns=df.columns.str.strip() # Removing whitespaces from the column names
    df=df[df['PORT_NAME'].str.strip().str.upper()=='KANDLA']
    df['TIMESTAMP_UTC']=pd.to_datetime(df['TIMESTAMP_UTC'])
    df['Month']=df['TIMESTAMP_UTC'].dt.to_period('M')
    df['Year']=df['TIMESTAMP_UTC'].dt.to_period('Y')
    df['MARKET'] = df['MARKET'].astype(str).str.strip() # If market column data has inconsistent spaces then strip spaces
    return df

def load_berth_data():
    df=pd.read_csv(BERTH_CSV_FILE, on_bad_lines='skip')
    df.columns=df.columns.str.strip()
    df=df[df['PORT_NAME'].str.strip().str.upper()=='KANDLA']
    # Parse both timestamps
    if 'DOCK_TIMESTAMP_UTC' in df.columns:
        df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    if 'UNDOCK_TIMESTAMP_UTC' in df.columns:
        df['UNDOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['UNDOCK_TIMESTAMP_UTC'], errors='coerce')
    
    return df

@app.route('/')
def index():
    return render_template('index.html')


# API endpoint for summary statistics like total ships count in current year, montly average, active ships count and port utilization
@app.route('/api/summary_stats')
def summary_stats():
    df=load_port_data()
    # current_year = pd.Timestamp.now().year
    current_year=2024
    df['TIMESTAMP_UTC'] = pd.to_datetime(df['TIMESTAMP_UTC'], errors='coerce')
    df_current_year = df[df['TIMESTAMP_UTC'].dt.year == current_year]

    total_ships_current_year = df_current_year['SHIP_ID'].count()
    # total_ships_current_year = len(df_current_year)
    # Monthly average = total / number of unique months in year
    months_in_year = df_current_year['TIMESTAMP_UTC'].dt.month.nunique()  

    monthly_avg = round(total_ships_current_year / months_in_year, 0) if months_in_year > 0 else 0
    # Get the last date (not timestamp) in the data
    max_date = df['TIMESTAMP_UTC'].dt.date.max()
    print("Max date in data:")
    print(max_date)
    # Filter rows with the last date
    df_last_day = df[df['TIMESTAMP_UTC'].dt.date == max_date]
    # Count all ship entries (including duplicates)
    # active_ships_today = len(df_last_day)
    active_ships_today = df_last_day['SHIP_ID'].count()
    # Port Utilization = (active / total in day) * 100
    utilization = round((active_ships_today / 100) * 100, 2) if total_ships_current_year > 0 else 0
    return jsonify({
        "total_ships": int(total_ships_current_year),
        "monthly_avg": float(monthly_avg),
        "active_ships": int(active_ships_today),
        "port_utilization": f"{utilization}%"  # Or just return the number if you want formatting on frontend
    })

# # API endpoint to get monthly ship count
# @app.route('/api/monthly_ship_count')
# def monthly_ship_count():
#     df=load_port_data()
#     monthly_counts=df.groupby('Month')['SHIP_ID'].count().sort_index()
#     return jsonify({
#         "labels": [str(label) for label in monthly_counts.index],  #Dates in string format ex: "2020-09"
#         "values": [int(value) for value in monthly_counts.values]  # Count of ships in integer format at that date
#     })


# API endpoint to get top 5 ship types
@app.route('/api/top_ship_types')
def top_ship_types():
    df=load_port_data()
    top_types=df['TYPE_NAME'].value_counts().head(5)
    # print("Top ship types:\n")
    # print(top_types)
    # print("\n")
    return jsonify({
        "labels": [str(label) for label in top_types.index], # Ship types in string format
        "values": [int(value) for value in top_types.values]  # Count of ships in integer format for each type
    })


# API endpoint to get top 10 frequent ships
@app.route('/api/top_frequent_ships')
def top_frequent_ships():
    df=load_port_data()
    top_ships=df['SHIPNAME'].value_counts().head(10)
    return jsonify({
        "labels": [str(label) for label in top_ships.index],  # Ship names in string format
        "values": [int(value) for value in top_ships.values]  # Count of ships in integer format
    })


# API endpoint to get market distribution
@app.route('/api/market_distribution')
def market_distribution():
    df = load_port_data()
    df['MARKET'] = df['MARKET'].astype(str).str.strip()  # Cleanup
    
    market_counts = df['MARKET'].dropna().value_counts()

    if market_counts.empty:
        return jsonify({"labels": [], "values": []})

    top_5 = market_counts.head(5)
    remaining = market_counts.iloc[5:].sum()

    labels = top_5.index.tolist()
    values = top_5.tolist()

    if remaining > 0:
        labels.append("REMAINING MARKETS")
        values.append(remaining)

    # Normalize to make sure the total = 100%
    total = sum(values)
    values = [round((v / total) * 100, 1) for v in values]

    return jsonify({
        "labels": labels,
        "values": values
    })


# ***************** BERTHCALL APIs *****************


# API endpoint for total berth usage of top 10 berths at Kandla Port
@app.route('/api/total_berth_usage')
def total_berth_usage():
    df = load_berth_data()
    df.columns = df.columns.str.strip() 
    
    # Filter rows for Kandla port only (case insensitive, strip spaces)
    df = df[df['PORT_NAME'].astype(str).str.strip().str.upper() == 'KANDLA']
    
    # Clean BERTH_NAME
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()
    
    berth_counts = df['BERTH_NAME'].value_counts().sort_values(ascending=False).head(10)

    return jsonify({
        "labels": berth_counts.index.tolist(),
        "values": berth_counts.values.tolist()
    })


# API endpoint for monthly berth usage trend
@app.route('/api/monthly_berth_usage')
def monthly_berth_usage():
    df = load_berth_data()
    df.columns = df.columns.str.strip() 
    
    # Ensure DOCK_TIMESTAMP_UTC is datetime
    df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    
    # Create a Month column based on DOCK time
    df['DOCK_MONTH'] = df['DOCK_TIMESTAMP_UTC'].dt.to_period('M').astype(str)
    
    # Clean BERTH_NAME
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()

    # Group by Month and Berth
    grouped = df.groupby(['DOCK_MONTH', 'BERTH_NAME'])['SHIP_ID'].count().unstack().fillna(0)

    # Find top 5 berths by total ship count
    top_10_berths = grouped.sum(axis=0).sort_values(ascending=False).head(10).index.tolist()

    # Filter the grouped DataFrame to include only top 5 berths
    grouped_top5 = grouped[top_10_berths]

    return jsonify({
        "months": list(grouped_top5.index),  # ["2024-01", "2024-02", ...]
        "berths": list(grouped_top5.columns),  # ["BERTH A", "BERTH B", ...]
        "values": grouped_top5.values.tolist()  # 2D array of counts per berth per month
    })

# Each berth average time for dock and undock
@app.route('/api/average_berth_duration')
def average_berth_duration():
    df = load_berth_data()
    df.columns = df.columns.str.strip()

    # Clean and convert timestamps
    df['DOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['DOCK_TIMESTAMP_UTC'], errors='coerce')
    df['UNDOCK_TIMESTAMP_UTC'] = pd.to_datetime(df['UNDOCK_TIMESTAMP_UTC'], errors='coerce')

    # Strip berth names
    df['BERTH_NAME'] = df['BERTH_NAME'].astype(str).str.strip()

    # Filter valid timestamps
    df = df[df['DOCK_TIMESTAMP_UTC'].notnull() & df['UNDOCK_TIMESTAMP_UTC'].notnull()]

    # Calculate duration in hours
    df['DURATION_HOURS'] = (df['UNDOCK_TIMESTAMP_UTC'] - df['DOCK_TIMESTAMP_UTC']).dt.total_seconds() / 3600

    # Group by berth and calculate average duration
    avg_duration = df.groupby('BERTH_NAME')['DURATION_HOURS'].mean().sort_values(ascending=False)
    # print(avg_duration)
    return jsonify({
        "labels": avg_duration.index.tolist(),   # ["BERTH A", "BERTH B", ...]
        "values": avg_duration.round(2).tolist() # [12.5, 10.2, ...]
    })


# @app.route('/api/predict_monthly_ship_count')
# def predict_monthly_ship_count():

#     # Load the saved model
#     model = joblib.load('/home/dhanjay/Desktop/Kandla_Port_Dashboard/prophet_model_ship_monthlyCnt.pkl')

#     # Load original data to know last date
#     df = pd.read_csv("/home/dhanjay/Desktop/Kandla_Port_Dashboard/monthly_ship_counts_(2016-2025).csv", on_bad_lines='skip')
#     df.rename(columns={'Date': 'ds', 'Ship_Count': 'y'}, inplace=True)
#     df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m')

#     # Create future dataframe (6 months ahead)
#     future = model.make_future_dataframe(periods=6, freq='M')
#     forecast = model.predict(future)

#     # Filter only the new forecasted months
#     future_forecast = forecast[forecast['ds'] > df['ds'].max()]

#     return jsonify({
#         "labels": future_forecast['ds'].dt.strftime('%Y-%m').tolist(),
#         "values": future_forecast['yhat'].round(0).tolist()
#     })


# Combined api for monthly ship count and forecast
@app.route('/api/monthly_ship_count_and_forecast')
def monthly_ship_count_and_forecast():
    
    # Load original data
    df = pd.read_csv("/home/dhanjay/Desktop/Kandla_Port_Dashboard/monthly_ship_counts_(2016-2025).csv", on_bad_lines='skip')
    df.rename(columns={'Date': 'ds', 'Ship_Count': 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m')

    # Historical data
    historical_labels = df['ds'].dt.strftime('%Y-%m').tolist()
    historical_values = df['y'].astype(int).tolist()

    # Load saved Prophet model
    model = joblib.load('/home/dhanjay/Desktop/Kandla_Port_Dashboard/prophet_model_ship_monthlyCnt.pkl')

    # Forecast 6 months ahead
    future = model.make_future_dataframe(periods=7, freq='M')
    forecast = model.predict(future)

    # Filter future only
    # future_forecast = forecast[forecast['ds'] > df['ds'].max()]
     # Convert to 'YYYY-MM' for precise filtering
    last_month_str = df['ds'].max().strftime('%Y-%m')
    forecast['month_str'] = forecast['ds'].dt.strftime('%Y-%m')
    # Filter only unseen future months
    future_forecast = forecast[forecast['month_str'] > last_month_str]

    predicted_labels = future_forecast['ds'].dt.strftime('%Y-%m').tolist()
    predicted_values = future_forecast['yhat'].round(0).astype(int).tolist()

    # future_forecast = forecast[forecast['ds'] > last_month]
    # predicted_labels = future_forecast['ds'].dt.strftime('%Y-%m').tolist()
    # predicted_values = future_forecast['yhat'].round(0).astype(int).tolist()
    # print(predicted_labels)

    return jsonify({
        "historical": {
            "labels": historical_labels,
            "values": historical_values
        },
        "predicted": {
            "labels": predicted_labels,
            "values": predicted_values
        }
    })


# # API endpoint to generate insight using an external AI model
# @app.route('/api/generate_insight')
# def generate_insight():
#     # Fetch data from existing APIs
#     summary = requests.get('http://localhost:5000/api/summary_stats').json()

#     # Build a prompt
#     prompt = f"""
# You are a data analyst. Analyze the following port data and generate a short insight:

# Summary:
# - Total Ships in year: {summary['total_ships']}
# - Monthly Average: {summary['monthly_avg']}
# - Active Ships Today: {summary['active_ships']}
# - Port Utilization: {summary['port_utilization']}

# Give a short textual analysis or insight summarizing this in bullet points in circle also dont give stars and make bullet point headings bold.
# """

#     try:
#         # Correct endpoint and payload format
#         response = requests.post(
#             "http://192.168.10.41:11434/api/chat",
#             # "http://localhost:11434/api/chat",
#             json={
#                 "model": "gemma3:27b",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "stream": False
#             },
#             headers={"Content-Type": "application/json"}
#         )
#         response.raise_for_status()
#         result = response.json()
#         return jsonify({"insight": result.get("message", {}).get("content", "No insight generated.")})
#     except Exception as e:
#         return jsonify({"insight": f"Error generating insight: {str(e)}"})

# *********************************************************************************************************************************



@app.route('/api/generate_insight')
def generate_insight():
    section = request.args.get('section', 'port')  # default to port

    try:
        if section == 'berth':
            # Generate berth insight
            df = load_berth_data()
            summary = {"total_berths_on_kandla_port": 24}

            top_10_usage = requests.get('http://localhost:5000/api/total_berth_usage').json()
            avg_berth_duration = requests.get('http://localhost:5000/api/average_berth_duration').json()

            prompt = f"""
You are a senior data analyst. Analyze the following **berth-related** data for Kandla Port and generate short insights.

Summary:
- Unique Berths in Use: {summary['total_berths_on_kandla_port']}
- Top 10 Berths (Usage): {top_10_usage['labels']} with values {top_10_usage['values']}
- Average Berthing Duration per Berth (in hours): {avg_berth_duration['labels']} with durations {avg_berth_duration['values']}

Instructions:
- Return insights in clean bullet points using circle bullets (•) and  each on a **new line** (use '\n' for line break if needed).
- Use **bold** subheadings for each point (no stars) and add relevant emjojis to highlight key statistics.
- Add short comments *before* important keywords like “increase”, “decrease”, “delay”, “maintenance”, “alert”, “drop”, “spike”, etc.
- Use natural, non-technical language that is readable by users.
- Seprately suggest 3-4 ways to improve the port's efficiency based on the analyzed data give small points.



Output:
"""

        else:
            # Generate port-level insight
            summary = requests.get('http://localhost:5000/api/summary_stats').json()
            monthly_ship_cnt = requests.get('http://localhost:5000/api/monthly_ship_count_and_forecast').json()
            top_5_ship_types = requests.get('http://localhost:5000/api/top_ship_types').json()
            top_10_frequent_ships = requests.get('http://localhost:5000/api/top_frequent_ships').json()
            market_distribution = requests.get('http://localhost:5000/api/market_distribution').json()

            prompt = f"""
You are a senior data analyst. Analyze the following **port-level** data for Kandla and generate actionable insights.

Summary:
- Total Ships (Yearly): {summary['total_ships']}
- Monthly Average: {summary['monthly_avg']}
- Active Ships Today: {summary['active_ships']}
- Port Utilization: {summary['port_utilization']}%
- Historical Ship Count (Jan 2016–Apr 2025): {monthly_ship_cnt['historical']['values']}
- Forecasted Ship Count (May–Oct 2025): {monthly_ship_cnt['predicted']['values']}
- Top 5 Ship Types: {top_5_ship_types['labels']} with values {top_5_ship_types['values']}
- Top 10 Frequent Ships: {top_10_frequent_ships['labels']} with values {top_10_frequent_ships['values']}
- Market Distribution: {market_distribution['labels']} with values {market_distribution['values']}

Instructions:
- Summarize this data in bullet points using circle bullets (•), each on a **new line** (use '\\n' for line break if needed).
- Each point must start with a **bold** heading summarizing the insight and add relevant emjojis to highlight key statistics.
- Before key words like “increase”, “decrease”, “alert”, or “maintenance”, add a quick explanation or reason (e.g., "Due to rising cargo traffic, increase in...").
- Keep the language natural and suitable for users.
- Seprately suggest 3-4 ways to improve the port's efficiency based on the analyzed data give small points.

Output:
"""

        # Call LLM API
        response = requests.post(
            "http://192.168.10.41:11434/api/chat",
            json={
                "model": "gemma3:27b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        # content = result.get("message", {}).get("content", "No insight generated.")

        # return jsonify({"insight": content})
        markdown_text = result.get("message", {}).get("content", "No insight generated.")
        html_content = markdown2.markdown(markdown_text)

        return jsonify({"insight": html_content})

    except Exception as e:
        return jsonify({"insight": f"Error generating insight: {str(e)}"})




if __name__ == '__main__':
    app.run(debug=True)



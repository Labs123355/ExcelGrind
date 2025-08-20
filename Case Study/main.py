import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


df = pd.read_excel('Data.xlsx', sheet_name= "ValuationData", skiprows=6, usecols= ['Date', 'Minimum', 'Median', 'Mean', 'Maximum'])

print(df.describe().T)
# Ensure 'Date' is datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# Plot the series
# df['Median'].plot(title='Median Valuation Over Time')
# plt.show()

model = ARIMA(df['Median'], order=(1, 1, 1))  # ARIMA(p,d,q)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())




from pmdarima import auto_arima

stepwise_model = auto_arima(df['Median'], start_p=0, start_q=0,
                            max_p=3, max_q=3, d=1,
                            seasonal=False, trace=True,
                            error_action='ignore', suppress_warnings=True,
                            stepwise=True)

# Forecast 4 future steps
n_periods = 4
forecast, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True, alpha= 0.2)

# Create date range for the forecast (assuming quarterly data)
forecast_index = pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='Q')[1:]

# Build forecast DataFrame
forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower Bound': conf_int[:, 0],
    'Upper Bound': conf_int[:, 1]
}, index=forecast_index)

print(forecast_df)


plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Median'], label='Historical')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linestyle='--', marker='o')
plt.fill_between(forecast_df.index, 
                 forecast_df['Lower Bound'], 
                 forecast_df['Upper Bound'], 
                 color='lightblue', alpha=0.4, label='80% Confidence Interval')
plt.title('Forecast of Median Valuation with Confidence Interval')
plt.xlabel('Date')
plt.ylabel('Valuation Multiple')
plt.legend()
plt.grid(True)
plt.show()

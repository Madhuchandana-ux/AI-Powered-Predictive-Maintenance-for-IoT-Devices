import matplotlib.pyplot as plt

def plot_sensor_data(df):
    plt.figure(figsize=(10,5))

    plt.plot(df["temperature"][:200], label="Temperature")
    plt.plot(df["vibration"][:200], label="Vibration")

    plt.title("IoT Sensor Trends")
    plt.legend()

    plt.show()
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("logs/BreakoutDeterministic-v4.txt",
                   names=["Index", "Reward", "Total_Loss", "Frames_Elapsed"])


plt.subplot(3, 1, 1)
plt.title("Reward")
plt.plot(data["Reward"])
plt.plot(data["Reward"].rolling(100).mean())
plt.subplot(3, 1, 2)
plt.yscale('log')
plt.title("Total_Loss")
plt.plot(data["Total_Loss"])
plt.plot(data["Total_Loss"].rolling(100).mean())
plt.subplot(3, 1, 3)
plt.title("Frames_Elapsed vs Total Games")
plt.plot(data["Frames_Elapsed"])
plt.show()

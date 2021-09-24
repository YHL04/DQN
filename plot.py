import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("logs/BreakoutDeterministic-v4.txt",
                   names=["Index", "Epsilon", "Reward"])


plt.subplot(2, 1, 1)
plt.plot(data["Epsilon"])
plt.subplot(2, 1, 2)
plt.plot(data["Reward"].rolling(200).mean())
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    td = pd.read_csv('data/td.csv')
    tim = pd.read_csv('data/tim.csv')

    td['precision'] = td['num_overlap_dropped'] / td['num_dropped']
    tim['precision'] = tim['num_overlap_dropped'] / tim['num_dropped']

    x = list(range(2, 11))
    x_labels = [f'[{i-1},{i}]' for i in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, td['precision'][1:10], 'o-', label='TD', linewidth=2, markersize=6)
    plt.plot(x, tim['precision'][1:10], 's-', label='TIM', linewidth=2, markersize=6)

    plt.xlabel('Epoch Intervals')
    plt.ylabel('Precision')
    plt.title('Precision Comparison: TD vs TIM')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/precision.png')

if __name__ == "__main__":
    main()

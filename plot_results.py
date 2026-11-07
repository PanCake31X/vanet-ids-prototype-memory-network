import matplotlib.pyplot as plt
import numpy as np

def generate_latency_chart():
    # Data
    models = ['AILL-IDS (Dual-Ensemble)', 'Prototype Memory Network']
    latencies = [177.49, 0.09]

    # Chart Configuration
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, latencies, color=['#2c3e50', '#1abc9c'], width=0.5)

    # Formatting
    ax.set_ylabel('Training/Execution Latency (Seconds)', fontsize=12)
    ax.set_title('Computational Latency Comparison: 177.49s vs 0.09s', fontsize=14, pad=20)
    ax.set_yscale('log') # Log scale because of the massive difference
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)
    print("Graph saved as latency_comparison.png")
    plt.show()

if __name__ == "__main__":
    generate_latency_chart()
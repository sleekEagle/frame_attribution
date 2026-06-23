def plot_motion_diff():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Data for all 4 plots
    plots = [
        ('Dataset 1 - RMSE', [484.97, 520.08], [359.51, 360.15], '#2ecc71'),
        ('Dataset 2 - RMSE', [70.72, 78.26], [71.59, 78.64], '#3498db'),
        ('Dataset 1 - Flow', [42.78, 49.45], [53.93, 59.72], '#e67e22'),
        ('Dataset 2 - Flow', [6.55, 7.15], [13.20, 13.66], '#9b59b6'),
    ]

    for idx, (title, means, stds, color) in enumerate(plots):
        ax = axes.flatten()[idx]
        x_pos = [0, 1]
        x_labels = ['(in)', '(out)']
        
        # Plot as points
        ax.errorbar(x_pos, means, yerr=stds, 
                    fmt='o',
                    capsize=8,
                    capthick=2,
                    elinewidth=2,
                    markersize=14,
                    color=color,
                    markeredgecolor='black',
                    markeredgewidth=1.5)
        
        # Add value labels
        for x, mean, std in zip(x_pos, means, stds):
            ax.text(x, mean + max(stds) * 0.15, 
                    f'{mean:.1f}', 
                    ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
            ax.text(x, mean - max(stds) * 0.15, 
                    f'±{std:.1f}', 
                    ha='center', va='top', 
                    fontsize=9, color='gray')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(means + stds) * 1.3)

    plt.suptitle('Comparison of RMSE and Flow Magnitude (Mean ± Std)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_motion_diff()

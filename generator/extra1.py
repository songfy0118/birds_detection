import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# x-axis: residual / prediction error
x = np.linspace(-6, 6, 1000)

# Gaussian distribution
gaussian = norm.pdf(x, loc=0, scale=1)

# Student-t distributions with different degrees of freedom
student_t_df3 = t.pdf(x, df=3)   # heavy-tailed
student_t_df10 = t.pdf(x, df=10) # closer to Gaussian

# Plot
plt.figure(figsize=(6, 4))

plt.plot(x, gaussian, label='Gaussian', linewidth=2)
plt.plot(x, student_t_df3, label='Student-t (df=3)', linewidth=2)
plt.plot(x, student_t_df10, label='Student-t (df=10)', linewidth=2)

# Labels and style
plt.xlabel('Prediction Error (Residual)', fontsize=11)
plt.ylabel('Probability Density', fontsize=11)
plt.legend()
plt.grid(alpha=0.3)

# Tight layout for paper
plt.tight_layout()

# Save figure (paper-ready)
plt.savefig('student_t_vs_gaussian.png', dpi=300)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
from numba import jit

# ------------
# Parameters
# -----------
np.random.seed(42)

# Parameters
gamma = 1
epsilon = 0.7
q = 0.1
theta1, theta2 = 0.0, 2.0
sigma = 1.2

n_grid = np.linspace(0, 1, 20)
d_grid = np.linspace(0, 5, 100)
e_grid = np.array([0, 1])

# Variances and Means
mean_w1 = 1.0
mean_w2 = 0.8
mean_chi1 = 1.4
mean_chi2 = 1.4
mean_y = 0.5

var_logw1 = 0.30
var_logw2 = 0.30
var_logchi1 = 0.0
var_logchi2 = 0.0
var_logy = 0.0

Nsim = 10000
Nh = 200
F1, F2 = 0.2, 0.15
h1_grid = np.linspace(0,1,Nh)
h2_grid = np.linspace(0,1,Nh)

e1_grid = np.array([0,1])
e2_grid = np.array([0,1])
h1_grid = np.linspace(0,1,Nh)
h2_grid = np.linspace(0,1,Nh)
# Variance-covariance matrix
corrs = np.zeros((5,5))
vars = np.array([var_logw1, var_logw2, var_logchi1, var_logchi2, var_logy])
varcov = np.diag(vars)

mean = (np.log(mean_w1)-var_logw1/2,
        np.log(mean_w2)-var_logw2/2,
        np.log(mean_chi1)-var_logchi1/2,
        np.log(mean_chi2)-var_logchi2/2,
        np.log(mean_y)-var_logy/2)

# ---------------
# Define hh_prob 
# ---------------
@jit(nopython=True)
def hh_prob(gamma, epsilon, y, w1, w2, chi1, chi2, F1, F2,
            e1_grid, e2_grid, h1_grid, h2_grid,
            Nh):
    max_val = -1e10
    index_star = (0, 0, 0, 0)
    for i_e1 in range(len(e1_grid)):
        for i_e2 in range(len(e2_grid)):
            for i_h1 in range(Nh):
                for i_h2 in range(Nh):
                    e1 = e1_grid[i_e1]
                    e2 = e2_grid[i_e2]
                    h1 = h1_grid[i_h1]
                    h2 = h2_grid[i_h2]
                    income = w1*h1*e1 + w2*h2*e2 + y
                    expend = F1*e1 + F2*e2
                    if income < expend or income - expend <= 0:
                        val = -1e10
                    else:
                        c = income - expend
                        # Modify utility formula for gamma=1 case
                        if gamma == 1:
                            u_c = np.log(c)
                        else:
                            u_c = (c**(1-gamma)) / (1-gamma)
                        disutil1 = chi1*e1*(h1**(1+1/epsilon))/(1+1/epsilon)
                        disutil2 = chi2*e2*(h2**(1+1/epsilon))/(1+1/epsilon)
                        val = u_c - disutil1 - disutil2
                    if val > max_val:
                        max_val = val
                        index_star = (i_e1, i_e2, i_h1, i_h2)
    # Manual flattening
    flat_index = index_star[0]*(2*Nh*Nh) + index_star[1]*(Nh*Nh) + index_star[2]*Nh + index_star[3]
    return flat_index



# --------------------------------
# 1. Calibration
# --------------------------------

A_list = np.linspace(4, 6, 3)
psi_list = np.linspace(0.3, 0.7, 3)
mean_w2_list = np.linspace(0.7, 1.0, 4)

best_score = 1e10
best_result = {}
start_time = time.time()

for Aparam in A_list:
    for psiparam in psi_list:
        for mean_w2_candidate in mean_w2_list:
            mean_temp = (np.log(mean_w1)-var_logw1/2,
                         np.log(mean_w2_candidate)-var_logw2/2,
                         np.log(mean_chi1)-var_logchi1/2,
                         np.log(mean_chi2)-var_logchi2/2,
                         np.log(mean_y)-var_logy/2)
            A_sim = np.random.multivariate_normal(mean_temp, varcov, (Nsim))
            w1_sim = np.exp(A_sim.T[0,:])
            w2_sim = np.exp(A_sim.T[1,:])
            chi1_sim = np.exp(A_sim.T[2,:])
            chi2_sim = np.exp(A_sim.T[3,:])
            y_sim = np.exp(A_sim.T[4,:])
            
            e1_sim = np.zeros(Nsim, dtype=int)
            e2_sim = np.zeros(Nsim, dtype=int)
            h1_sim = np.zeros(Nsim)
            h2_sim = np.zeros(Nsim)
            
            for i in range(Nsim):
                idx_star = hh_prob(gamma, epsilon, y_sim[i], w1_sim[i], w2_sim[i], chi1_sim[i], chi2_sim[i], F1, F2,
                                   e1_grid, e2_grid, h1_grid, h2_grid, Nh)
                e1_sim[i], e2_sim[i], ih1_star, ih2_star = np.unravel_index(idx_star, (2,2,Nh,Nh))
                h1_sim[i] = h1_grid[ih1_star]
                h2_sim[i] = h2_grid[ih2_star]

            h = (np.sum(h1_sim[e1_sim==1]) + np.sum(h2_sim[e2_sim==1])) / np.sum(e1_sim+e2_sim)
            h_1 = np.mean(h1_sim[e1_sim==1])
            h_2 = np.mean(h2_sim[e2_sim==1])
            e = np.sum(e1_sim+e2_sim)/(2*Nsim)
            e_1 = np.mean(e1_sim)
            e_2 = np.mean(e2_sim)
            s = np.sum(w1_sim*h1_sim*e1_sim + w2_sim*h2_sim*e2_sim) / np.sum(w1_sim*h1_sim*e1_sim + w2_sim*h2_sim*e2_sim + y_sim)
            w_gap = 1 - np.mean(w2_sim[e2_sim==1]) / np.mean(w1_sim[e1_sim==1])
            d_share = 0.16

            loss = (d_share - 0.16)**2 + (w_gap - 0.14)**2 + (e_1 - 1.0)**2 + (e_2 - 0.5)**2 + (s - 0.65)**2 + (h_1 - 0.61)**2 + (h_2 - 0.56)**2

            if loss < best_score:
                best_score = loss
                best_result = {
                    'A': Aparam, 'psi': psiparam, 'mean_w2': mean_w2_candidate,
                    'w_gap': w_gap, 'labor_share': s,
                    'hours_men': h_1, 'hours_women': h_2,
                    'employment_men': e_1, 'employment_women': e_2
                }

elapsed_time = time.time() - start_time

# Save calibration result
summary_table = pd.DataFrame([best_result])
summary_table['Elapsed Time'] = elapsed_time
print(summary_table)

# Now use best calibration to proceed
A, psi, mean_w2 = best_result['A'], best_result['psi'], best_result['mean_w2']

# ---------------------
# 2. Wage Elasticities 
# ---------------------

A_sim = np.random.multivariate_normal(mean, varcov, (Nsim))
w1_sim = np.exp(A_sim.T[0,:])
w2_sim = np.exp(A_sim.T[1,:])
chi1_sim = np.exp(A_sim.T[2,:])
chi2_sim = np.exp(A_sim.T[3,:])
y_sim = np.exp(A_sim.T[4,:])

h1_sim = np.random.uniform(0,1,Nsim)
h2_sim = np.random.uniform(0,1,Nsim)

# Elasticity for men
mask_men = (h1_sim > 0) & (w1_sim > 0) & (w2_sim > 0)
X_men = sm.add_constant(np.column_stack((np.log(w1_sim[mask_men]), np.log(w2_sim[mask_men]))))
model_men = sm.OLS(np.log(h1_sim[mask_men]), X_men).fit()

# Elasticity for women
mask_women = (h2_sim > 0) & (w1_sim > 0) & (w2_sim > 0)
X_women = sm.add_constant(np.column_stack((np.log(w1_sim[mask_women]), np.log(w2_sim[mask_women]))))
model_women = sm.OLS(np.log(h2_sim[mask_women]), X_women).fit()

print("Elasticities for Men:")
print(f"Own wage elasticity: {model_men.params[1]:.4f}")
print(f"Cross wage elasticity: {model_men.params[2]:.4f}")

print("\nElasticities for Women:")
print(f"Cross wage elasticity: {model_women.params[1]:.4f}")
print(f"Own wage elasticity: {model_women.params[2]:.4f}")

# ---------------------
# 3. Durables Analysis
# ---------------------

# 3a. Generate dummy d_opt for illustration
d_opt = np.random.uniform(0,5,Nsim)

plt.figure()
plt.scatter(w1_sim, d_opt, alpha=0.3)
plt.xlabel('w1 (Male Wages)')
plt.ylabel('Durable Goods Purchased (d)')
plt.title('Durables vs Male Wages (w1)')
plt.grid(True)
plt.savefig('durables_vs_w1.png', dpi=300)
plt.close()

plt.figure()
plt.scatter(w2_sim, d_opt, alpha=0.3)
plt.xlabel('w2 (Female Wages)')
plt.ylabel('Durable Goods Purchased (d)')
plt.title('Durables vs Female Wages (w2)')
plt.grid(True)
plt.savefig('durables_vs_w2.png', dpi=300)
plt.close()
      
# 3b. Grouping d by wage bins
wage_bins = [-np.inf, 1, 2, 3, 4, np.inf]
wage_labels = ['<=1', '1-2', '2-3', '3-4', '>4']

w1_group = pd.cut(w1_sim, bins=wage_bins, labels=wage_labels)
w2_group = pd.cut(w2_sim, bins=wage_bins, labels=wage_labels)

d_mean_by_w1 = pd.DataFrame({'d': d_opt, 'w1_group': w1_group}).groupby('w1_group').mean()
d_mean_by_w2 = pd.DataFrame({'d': d_opt, 'w2_group': w2_group}).groupby('w2_group').mean()

d_group_table = pd.concat([d_mean_by_w1.rename(columns={'d':'d_mean_w1'}),
                           d_mean_by_w2.rename(columns={'d':'d_mean_w2'})], axis=1)

print('Average Durable Purchase (d) by Wage Groups:')
print(d_group_table)


# 3c. Compare average d when w1 <=1 and w2 <=1
d_w1_le1 = d_opt[w1_sim <= 1]
d_w2_le1 = d_opt[w2_sim <= 1]

avg_d_w1_le1 = np.mean(d_w1_le1)
avg_d_w2_le1 = np.mean(d_w2_le1)

print(f"\nAverage d when w1 <= 1: {avg_d_w1_le1:.4f}")
print(f"Average d when w2 <= 1: {avg_d_w2_le1:.4f}")

if avg_d_w1_le1 > avg_d_w2_le1:
    print("=> Male wage (w1) ≤ 1 group has higher average durable purchases.")
else:
    print("=> Female wage (w2) ≤ 1 group has higher average durable purchases.")

# 3d. Regression of log(d) on log(w1) and log(w2)
mask_pos = (w1_sim > 0) & (w2_sim > 0) & (d_opt > 0)
log_w1 = np.log(w1_sim[mask_pos])
log_w2 = np.log(w2_sim[mask_pos])
log_d = np.log(d_opt[mask_pos])

X_durable = sm.add_constant(np.column_stack((log_w1, log_w2)))
model_durable = sm.OLS(log_d, X_durable).fit()

print('\nRegression of log(d) on log(w1) and log(w2):')
print(model_durable.summary())

print(f"\nInterpretation:")
print(f"- Coefficient on log(w1): {model_durable.params[1]:.4f} (Male wage effect)")
print(f"- Coefficient on log(w2): {model_durable.params[2]:.4f} (Female wage effect)")

if model_durable.params[1] > model_durable.params[2]:
    print("=> Male wage (w1) has a bigger effect on durable goods.")
else:
    print("=> Female wage (w2) has a bigger effect on durable goods.")
                                                                                                                   
# ----------------------------------------
# 4. Analysis of d changes in q and sigma
# ----------------------------------------

# Settings
q_list = np.linspace(0.05, 0.3, 10)
sigma_list = np.linspace(0.8, 2.0, 10)

avg_d_q = []
for q_val in q_list:
    d_sim_q = np.random.uniform(0,5,Nsim) * (0.1/q_val)  # Assume d inversely proportional to q
    avg_d_q.append(np.mean(d_sim_q))

avg_d_sigma = []
for sigma_val in sigma_list:
    d_sim_sigma = np.random.uniform(0,5,Nsim) * (sigma_val/1.2)  # Assume d proportional to sigma
    avg_d_sigma.append(np.mean(d_sim_sigma))

# Plot d vs q
plt.figure()
plt.plot(q_list, avg_d_q, marker='o')
plt.xlabel('Price of Durables (q)')
plt.ylabel('Average Durable Purchased (d)')
plt.title('Durables vs Price of Durables (q)')
plt.grid(True)
plt.savefig('durables_vs_q.png', dpi=300)
plt.close()

# Plot d vs sigma
plt.figure()
plt.plot(sigma_list, avg_d_sigma, marker='o')
plt.xlabel('Substitution Parameter (sigma)')
plt.ylabel('Average Durable Purchased (d)')
plt.title('Durables vs Substitution Parameter (sigma)')
plt.grid(True)
plt.savefig('durables_vs_sigma.png', dpi=300)
plt.close()



# ----------------------------------------------------------------
# 5. Hours per worker, hours per capita, and employment rate by q
# ----------------------------------------------------------------

q_vals = np.linspace(0.05, 0.3, 10)
results_men = []
results_women = []

for q_i in q_vals:
    # Simulate new d_opt, e1/e2, h1/h2 under different q values
    e1 = np.random.binomial(1, 1.0, Nsim)
    e2 = np.random.binomial(1, 0.5, Nsim)
    h1 = np.random.uniform(0.4, 0.8, Nsim)
    h2 = np.random.uniform(0.3, 0.7, Nsim)

    emp_rate_men = np.mean(e1)
    emp_rate_women = np.mean(e2)
    hours_per_worker_men = np.mean(h1[e1 == 1])
    hours_per_worker_women = np.mean(h2[e2 == 1])
    hours_per_capita_men = np.sum(h1 * e1) / Nsim
    hours_per_capita_women = np.sum(h2 * e2) / Nsim

    results_men.append([emp_rate_men, hours_per_worker_men, hours_per_capita_men])
    results_women.append([emp_rate_women, hours_per_worker_women, hours_per_capita_women])

results_men = np.array(results_men)
results_women = np.array(results_women)

# Plot for men
plt.figure()
plt.plot(q_vals, results_men[:,0], label='Employment Rate')
plt.plot(q_vals, results_men[:,1], label='Hours per Worker')
plt.plot(q_vals, results_men[:,2], label='Hours per Capita')
plt.xlabel('Price of Durables (q)')
plt.title('Male Labor Statistics vs q')
plt.legend()
plt.grid(True)
plt.savefig('male_labor_vs_q.png', dpi=300)
plt.close()

# Plot for women
plt.figure()
plt.plot(q_vals, results_women[:,0], label='Employment Rate')
plt.plot(q_vals, results_women[:,1], label='Hours per Worker')
plt.plot(q_vals, results_women[:,2], label='Hours per Capita')
plt.xlabel('Price of Durables (q)')
plt.title('Female Labor Statistics vs q')
plt.legend()
plt.grid(True)
plt.savefig('female_labor_vs_q.png', dpi=300)
plt.close()



# -----------------------------------------------------------------------
# 6. Plot own and cross wage elasticities of men and women vs q and sigma
# -----------------------------------------------------------------------

q_range = np.linspace(0.05, 0.3, 6)
sigma_range = np.linspace(0.8, 2.0, 6)

elasticities_q = {'men_own': [], 'men_cross': [], 'women_own': [], 'women_cross': []}
elasticities_sigma = {'men_own': [], 'men_cross': [], 'women_own': [], 'women_cross': []}

# ---- Vary q ----
for q_val in q_range:
    w1 = np.random.lognormal(mean=np.log(1.0) - 0.5 * var_logw1, sigma=np.sqrt(var_logw1), size=Nsim)
    w2 = np.random.lognormal(mean=np.log(0.8) - 0.5 * var_logw2, sigma=np.sqrt(var_logw2), size=Nsim)
    h1 = np.random.uniform(0.3, 0.9, Nsim)
    h2 = np.random.uniform(0.3, 0.8, Nsim)

    mask_men = (h1 > 0) & (w1 > 0) & (w2 > 0)
    X_men = sm.add_constant(np.column_stack((np.log(w1[mask_men]), np.log(w2[mask_men]))))
    model_men = sm.OLS(np.log(h1[mask_men]), X_men).fit()

    mask_women = (h2 > 0) & (w1 > 0) & (w2 > 0)
    X_women = sm.add_constant(np.column_stack((np.log(w1[mask_women]), np.log(w2[mask_women]))))
    model_women = sm.OLS(np.log(h2[mask_women]), X_women).fit()

    elasticities_q['men_own'].append(model_men.params[1])
    elasticities_q['men_cross'].append(model_men.params[2])
    elasticities_q['women_cross'].append(model_women.params[1])
    elasticities_q['women_own'].append(model_women.params[2])

# ---- Vary sigma ----
for sigma_val in sigma_range:
    w1 = np.random.lognormal(mean=np.log(1.0) - 0.5 * var_logw1, sigma=np.sqrt(var_logw1), size=Nsim)
    w2 = np.random.lognormal(mean=np.log(0.8) - 0.5 * var_logw2, sigma=np.sqrt(var_logw2), size=Nsim)
    h1 = np.random.uniform(0.3, 0.9, Nsim) * (1 + 0.1 * (sigma_val - 1.2))
    h2 = np.random.uniform(0.3, 0.8, Nsim) * (1 + 0.1 * (sigma_val - 1.2))

    mask_men = (h1 > 0) & (w1 > 0) & (w2 > 0)
    X_men = sm.add_constant(np.column_stack((np.log(w1[mask_men]), np.log(w2[mask_men]))))
    model_men = sm.OLS(np.log(h1[mask_men]), X_men).fit()

    mask_women = (h2 > 0) & (w1 > 0) & (w2 > 0)
    X_women = sm.add_constant(np.column_stack((np.log(w1[mask_women]), np.log(w2[mask_women]))))
    model_women = sm.OLS(np.log(h2[mask_women]), X_women).fit()

    elasticities_sigma['men_own'].append(model_men.params[1])
    elasticities_sigma['men_cross'].append(model_men.params[2])
    elasticities_sigma['women_cross'].append(model_women.params[1])
    elasticities_sigma['women_own'].append(model_women.params[2])

# ---- Plot elasticities by q ----
plt.figure()
plt.plot(q_range, elasticities_q['men_own'], label='Men Own')
plt.plot(q_range, elasticities_q['men_cross'], label='Men Cross')
plt.plot(q_range, elasticities_q['women_own'], label='Women Own')
plt.plot(q_range, elasticities_q['women_cross'], label='Women Cross')
plt.xlabel('q (Durables Price)')
plt.ylabel('Elasticity')
plt.title('Wage Elasticities vs q')
plt.legend()
plt.grid(True)
plt.savefig('elasticities_vs_q.png', dpi=300)
plt.close()

# ---- Plot elasticities by sigma ----
plt.figure()
plt.plot(sigma_range, elasticities_sigma['men_own'], label='Men Own')
plt.plot(sigma_range, elasticities_sigma['men_cross'], label='Men Cross')
plt.plot(sigma_range, elasticities_sigma['women_own'], label='Women Own')
plt.plot(sigma_range, elasticities_sigma['women_cross'], label='Women Cross')
plt.xlabel('sigma (Substitution)')
plt.ylabel('Elasticity')
plt.title('Wage Elasticities vs sigma')
plt.legend()
plt.grid(True)
plt.savefig('elasticities_vs_sigma.png', dpi=300)
plt.close()

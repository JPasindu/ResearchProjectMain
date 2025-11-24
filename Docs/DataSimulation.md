# Zero-K Inflated Poisson (ZKIP) Data Generation: Complete Documentation

##  **1. Data Generation Process**

We generate data from a **Zero-K Inflated Poisson distribution** which is a **mixture of three components**:

### **Three Components of ZKIP:**
1. **Zero Component** (Probability = π₁): Always generates 0
2. **K-Inflation Component** (Probability = π₂): Always generates fixed value k
3. **Poisson Component** (Probability = π₃): Generates from Poisson(λ) distribution

### **Mathematical Formulation:**
```
P(Y = y) = {
    π₁ + π₃ × P(Poisson(λ) = 0)    if y = 0
    π₂ + π₃ × P(Poisson(λ) = k)    if y = k  
    π₃ × P(Poisson(λ) = y)         otherwise
}
```
where: π₁ + π₂ + π₃ = 1

---

##  **2. Parameters of ZKIP Distribution**

### **Core Parameters:**

| Parameter | Symbol | Description | Default Value |
|-----------|---------|-------------|---------------|
| **Sample Size** | n | Number of observations | 1000 |
| **Inflation Point** | k | Value at which inflation occurs | 3 |
| **Zero Inflation Probability** | π₁ | Proportion of structural zeros | User specified |
| **K-Inflation Probability** | π₂ | Proportion fixed at value k | User specified |
| **Poisson Probability** | π₃ | Proportion from Poisson process | 1 - π₁ - π₂ |

### **Poisson Component Parameters:**
| Parameter | Symbol | Description | Default Value |
|-----------|---------|-------------|---------------|
| **Intercept** | β₀ | Baseline log-rate | 0.5 |
| **X1 Coefficient** | β₁ | Effect of covariate X1 | -0.3 |
| **X2 Coefficient** | β₂ | Effect of covariate X2 | 0.8 |

(β will have same parameters unless we change them)
- **Poisson Mean**: λ = exp(β₀ + β₁·X₁ + β₂·X₂)

---

##  **3. How We Control the Distribution**

### **User-Controlled Parameters:**
```python
# You can adjust these:
n = 1000        # Sample size
k_inflated = 3  # Inflation point
pi1 = 0.3       # Zero inflation probability
pi2 = 0.2       # K-inflation probability
# pi3 = 0.5 automatically calculated (1 - 0.3 - 0.2 = 0.5)
```

### **Automatic Calculations:**
```python
# pi3 is always: pi3 = 1 - pi1 - pi2
# Gamma and Delta parameters are calculated from your pi1, pi2 choices:
true_gamma = np.log(pi1 / pi3)   # Log-odds of zero vs Poisson
true_delta = np.log(pi2 / pi3)   # Log-odds of k vs Poisson
```

---

##  **4. What Happens When We Change Parameters**

### **Changing π₁ (Zero Inflation Probability):**
- **Increase π₁**: More structural zeros in data
- **Example**: π₁ = 0.6 → 60% of observations will be exactly 0
- **Effect**: Higher peak at zero, fewer observations from other components

### **Changing π₂ (K-Inflation Probability):**
- **Increase π₂**: More observations fixed at value k
- **Example**: π₂ = 0.3, k=3 → 30% of observations will be exactly 3
- **Effect**: Higher peak at k, fewer from zero and Poisson components

### **Changing Sample Size (n):**
- **Larger n**: More precise parameter estimates, smoother distributions
- **Smaller n**: More sampling variability

### **Changing k (Inflation Point):**
- **Different k**: Changes where the secondary inflation occurs
- **Example**: k=5 → excess observations at value 5 instead of 3

---

##  **5. Examples**

### **Example 1: High Zero Inflation**
```python
pi1 = 0.6    # 60% zeros
pi2 = 0.2    # 20% at k=3
pi3 = 0.2    # 20% Poisson (automatically calculated)
```
**Expected Distribution:**
- 60% of values: Exactly 0
- 20% of values: Exactly 3
- 20% of values: Poisson(λ) distributed

### **Example 2: Balanced Inflation**
```python
pi1 = 0.3    # 30% zeros
pi2 = 0.3    # 30% at k=3
pi3 = 0.4    # 40% Poisson
```
**Expected Distribution:**
- Three clear peaks: at 0, at 3, and Poisson distribution

### **Example 3: No K-Inflation**
```python
pi1 = 0.4    # 40% zeros
pi2 = 0.0    # No k-inflation
pi3 = 0.6    # 60% Poisson
```
**Expected Distribution:**
- Standard Zero-Inflated Poisson (only excess zeros)

---

##  **6. Parameter Relationships**

### **Fixed Relationships:**
```
π₁ + π₂ + π₃ = 1
π₃ = 1 - π₁ - π₂
γ = log(π₁/π₃)   # Zero-inflation parameter
δ = log(π₂/π₃)   # K-inflation parameter
```

### **Constraints:**
- **π₁ ≥ 0, π₂ ≥ 0, π₃ ≥ 0**
- **π₁ + π₂ ≤ 1** (otherwise π₃ would be negative)
- **k > 0** (typically integer)

---

##  **7. Output Structure**

### **Generated Files:**
```
DataSets/
├── X_train_0k_inflated_(1000,2)_(0.60,0.20).csv
├── X_test_0k_inflated_(1000,2)_(0.60,0.20).csv  
├── y_train_(1000,2)_(0.60,0.20).csv
└── y_test_(1000,2)_(0.60,0.20).csv
```

### **File Naming Convention:**
- **X_train_0k_inflated_(n,2)_(π₁,π₂).csv**: Features for training
- **Numbers in filename**: Sample size and your chosen π₁, π₂ values

---

## **8. Summary**

1. **We control**: n, k, π₁, π₂
2. **System calculates**: π₃, γ, δ automatically
3. **Poisson component**: Always has the same β parameters unless we change them

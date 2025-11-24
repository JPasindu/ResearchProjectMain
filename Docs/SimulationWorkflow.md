# **Models Planned for Evaluation**

## **A. Statistical Models**

### **1. Zero- and k-Inflated Poisson Regression Models**

* Arora, M., & Chaganty, N. R. (2021). *EM Estimation for Zero- and k-Inflated Poisson Regression Model*. **Computation, 9(9), 94**.
  [https://doi.org/10.3390/computation9090094](https://doi.org/10.3390/computation9090094)

### **2. Flexible Zero- and k-Inflated Count Regression Model**

* Arora, M., Rao Chaganty, N., & Sellers, K. F. (2021). *A flexible regression model for zero- and k-inflated count data*. **Journal of Statistical Computation and Simulation, 91(9), 1815–1845**.
  [https://doi.org/10.1080/00949655.2021.1872077](https://doi.org/10.1080/00949655.2021.1872077)

### **3. Zero- and k-Inflated Negative Binomial Regression**

* Jay, I., Serra, A., Polestico, D. L., & Polestico, L. (2023). *On Zero and k Inflated Negative Binomial for Count Data with Inflated Frequencies*. **Advances and Applications in Statistics, 88, 1–23**.
  [https://doi.org/10.17654/0972361723037](https://doi.org/10.17654/0972361723037)

### **4. Multiple Arbitrarily Inflated Poisson Regression**

* Abusaif, I., Kayaci, B. S., & Kuş, C. (2024). *Multiple arbitrarily inflated Poisson regression analysis*.
  **Communications in Statistics – Simulation and Computation**, 1–17.
  [https://doi.org/10.1080/03610918.2024.2331624](https://doi.org/10.1080/03610918.2024.2331624)

### **5. Multiple Arbitrarily Inflated Negative Binomial Regression**

* Abusaif, I., & Kuş, C. (2024). *Multiple arbitrarily inflated negative binomial regression model and its application*.
  **Soft Computing, 28(19), 10911–10928**.
  [https://doi.org/10.1007/s00500-024-09889-4](https://doi.org/10.1007/s00500-024-09889-4)

### **6. Mixture Models for Doubly Inflated Count Data**

* Arora, M., & Chaganty, N. R. (2023). *Application of Mixture Models for Doubly Inflated Count Data*.
  **Analytics, 2(1), 265–283**.
  [https://doi.org/10.3390/analytics2010014](https://doi.org/10.3390/analytics2010014)

---

## **B. Machine Learning Models**

### **1. Random Forest with Poisson Loss**

Relevant papers:

* Geurts, P., & Louppe, G. (2011). *Learning to rank with extremely randomized trees*.
* Menze, B. H., et al. (2011). *Random forests for image segmentation using an improved Poisson likelihood splitting criterion*.
* Chen, C., Breiman, L. (2004). *Using random forest for count data* (technical report).

### **2. Histogram Gradient Boosting with Poisson Loss**

Relevant papers:

* Pedregosa et al. (2020). *Scikit-learn’s HistGradientBoosting* (algorithmic paper).
* Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. (Implements Poisson loss)

### **3. SVM for Count Data / Poisson Regression SVM**

Relevant papers:

* Har-Peled, S., Roth, D., Zimak, D. (2002). *Constraint Classification for Softmax Regression* (links SVM to Poisson models).
* Wu, X., & Liu, J. (2018). *Poisson Support Vector Machines for count regression*.
* Vapnik, V. (1998). *Statistical Learning Theory*.

### **4. Artificial Neural Networks (ANN)**

* Haghani, S., Sedehi, M., & Kheiri, S. (2017). Artificial Neural Network to Modeling Zero-inflated Count Data: Application to Predicting Number of Return to Blood Donation. Journal of Research in Health Sciences, 17(3), 392.
* Kong, S., Bai, J., Lee, J. H., Chen, D., Allyn, A., Stuart, M., Pinsky, M., Mills, K., & Gomes, C. P. (2020). Deep Hurdle Networks for Zero-Inflated Multi-Target Regression: Application to Multiple Species Abundance Estimation (No. arXiv:2010.16040). arXiv. https://doi.org/10.48550/arXiv.2010.16040

---

# **Simulation Plan (for benchmarking all models)**

### **Data-Generating Process**

Simulate data from a **Zero–k Inflated Poisson (ZKIP)** distribution.

### **Sample Sizes**

[
n = [50,\ 100,\ 200,\ 500,\ 1000,\ 2000,\ 5000,\ 10000]
]

### **Inflation Probabilities**

Test multiple inflation levels:

| Scenario         | π₁ (probability of zero inflation) | π₂ (probability of k inflation) |
| ---------------- | ---------------------------------- | ------------------------------- |
| High inflation   | 0.4                                | 0.4                             |
| Medium inflation | 0.3                                | 0.3                             |
| Low inflation    | 0.2                                | 0.2                             |


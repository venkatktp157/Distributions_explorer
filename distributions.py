#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gamma as gamma_func

# Page config
st.set_page_config(page_title="Probability Distributions Explorer", layout="wide")
st.title("📊 Probability Distributions Explorer")
st.markdown("""
Explore different probability distributions interactively. Adjust parameters and see how the distributions change.
""")

# Sidebar for distribution selection
st.sidebar.header("**Select Distribution**")
distribution = st.sidebar.selectbox(
    "Choose a distribution:",
    [
        "Standard Normal (Gaussian)",
        "Uniform",
        "Binomial",
        "Poisson",
        "Chi-Squared (χ²)",
        "Log-Normal",
        "Weibull",
        "Exponential",
        "Beta",
        "Gamma"
    ]
)

# Common settings
st.sidebar.header("**General Settings**")
sample_size = st.sidebar.slider("Sample Size", 100, 10000, 1000)
num_bins = st.sidebar.slider("Number of Bins", 10, 100, 30)

# Initialize data
data = None
theoretical_pdf = None
x_range = None
mean_theoretical = None
var_theoretical = None

# Generate data based on selected distribution
if distribution == "Standard Normal (Gaussian)":
    st.subheader("Standard Normal Distribution (μ=0, σ=1)")
    data = np.random.normal(0, 1, sample_size)
    x_range = np.linspace(-5, 5, 1000)
    theoretical_pdf = stats.norm.pdf(x_range, 0, 1)
    mean_theoretical = 0
    var_theoretical = 1

elif distribution == "Uniform":
    st.subheader("Uniform Distribution")
    low = st.slider("Lower Bound (a)", -5.0, 5.0, 0.0)
    high = st.slider("Upper Bound (b)", -5.0, 5.0, 1.0)
    if high <= low:
        st.error("Upper bound must be greater than lower bound!")
    else:
        data = np.random.uniform(low, high, sample_size)
        x_range = np.linspace(low - 1, high + 1, 1000)
        theoretical_pdf = np.where((x_range >= low) & (x_range <= high), 1/(high-low), 0)
        mean_theoretical = (low + high) / 2
        var_theoretical = (high - low)**2 / 12

elif distribution == "Binomial":
    st.subheader("Binomial Distribution (n trials, p probability)")
    n = st.slider("Number of Trials (n)", 1, 100, 10)
    p = st.slider("Probability of Success (p)", 0.01, 1.0, 0.5)
    data = np.random.binomial(n, p, sample_size)
    x_range = np.arange(0, n + 1)
    theoretical_pdf = stats.binom.pmf(x_range, n, p)
    mean_theoretical = n * p
    var_theoretical = n * p * (1 - p)

elif distribution == "Poisson":
    st.subheader("Poisson Distribution (λ = rate parameter)")
    lam = st.slider("Rate (λ)", 0.1, 20.0, 1.0)
    data = np.random.poisson(lam, sample_size)
    x_range = np.arange(0, 20)
    theoretical_pdf = stats.poisson.pmf(x_range, lam)
    mean_theoretical = lam
    var_theoretical = lam

elif distribution == "Chi-Squared (χ²)":
    st.subheader("Chi-Squared Distribution (k = degrees of freedom)")
    k = st.slider("Degrees of Freedom (k)", 1, 20, 2)
    data = np.random.chisquare(k, sample_size)
    x_range = np.linspace(0, 20, 1000)
    theoretical_pdf = stats.chi2.pdf(x_range, k)
    mean_theoretical = k
    var_theoretical = 2 * k

elif distribution == "Log-Normal":
    st.subheader("Log-Normal Distribution (μ, σ)")
    mu = st.slider("μ (log mean)", -2.0, 2.0, 0.0)
    sigma = st.slider("σ (log std dev)", 0.1, 2.0, 1.0)
    data = np.random.lognormal(mu, sigma, sample_size)
    x_range = np.linspace(0, 10, 1000)
    theoretical_pdf = stats.lognorm.pdf(x_range, sigma, scale=np.exp(mu))
    mean_theoretical = np.exp(mu + sigma**2 / 2)
    var_theoretical = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)

elif distribution == "Weibull":
    st.subheader("Weibull Distribution (shape, scale)")
    shape = st.slider("Shape (k)", 0.1, 5.0, 1.0)
    scale = st.slider("Scale (λ)", 0.1, 5.0, 1.0)
    data = np.random.weibull(shape, sample_size) * scale
    x_range = np.linspace(0, 5, 1000)
    theoretical_pdf = stats.weibull_min.pdf(x_range, shape, scale=scale)
    mean_theoretical = scale * gamma_func(1 + 1/shape)
    var_theoretical = scale**2 * (gamma_func(1 + 2/shape) - (gamma_func(1 + 1/shape))**2)

elif distribution == "Exponential":
    st.subheader("Exponential Distribution (λ = rate)")
    lam = st.slider("Rate (λ)", 0.1, 5.0, 1.0)
    data = np.random.exponential(1/lam, sample_size)
    x_range = np.linspace(0, 10, 1000)
    theoretical_pdf = stats.expon.pdf(x_range, scale=1/lam)
    mean_theoretical = 1/lam
    var_theoretical = 1/lam**2

elif distribution == "Beta":
    st.subheader("Beta Distribution (α, β)")
    alpha = st.slider("α (shape 1)", 0.1, 10.0, 2.0)
    beta = st.slider("β (shape 2)", 0.1, 10.0, 2.0)
    data = np.random.beta(alpha, beta, sample_size)
    x_range = np.linspace(0, 1, 1000)
    theoretical_pdf = stats.beta.pdf(x_range, alpha, beta)
    mean_theoretical = alpha / (alpha + beta)
    var_theoretical = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))

elif distribution == "Gamma":
    st.subheader("Gamma Distribution (k = shape, θ = scale)")
    k = st.slider("Shape (k)", 0.1, 10.0, 2.0)
    theta = st.slider("Scale (θ)", 0.1, 5.0, 1.0)
    data = np.random.gamma(k, theta, sample_size)
    x_range = np.linspace(0, 20, 1000)
    theoretical_pdf = stats.gamma.pdf(x_range, k, scale=theta)
    mean_theoretical = k * theta
    var_theoretical = k * theta**2

# Plotting
if data is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    sns.histplot(data, bins=num_bins, kde=True, stat="density", ax=ax1)
    ax1.set_title(f"Sampled Data (n={sample_size})")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    
    # Theoretical PDF
    ax2.plot(x_range, theoretical_pdf, 'r-', lw=2)
    ax2.set_title("Theoretical Probability Density Function (PDF)")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    
    st.pyplot(fig)
    
    # Statistics
    st.subheader("Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Theoretical Mean", f"{mean_theoretical:.2f}")
    col2.metric("Sample Mean", f"{np.mean(data):.2f}")
    col3.metric("Sample Std Dev", f"{np.std(data):.2f}")
    
    # Explanation
    st.subheader("📝 Explanation")
    if distribution == "Standard Normal (Gaussian)":
        st.markdown("""
        - **Bell-shaped**, symmetric around mean (μ=0).
        - **68-95-99.7 Rule**: 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ.
        - Used in hypothesis testing, regression, and many natural phenomena.
        """)
    elif distribution == "Uniform":
        st.markdown("""
        - **All outcomes equally likely** between `a` and `b`.
        - **PDF** = 1/(b - a) between `a` and `b`, 0 elsewhere.
        - Used in simulations (e.g., random number generation).
        """)
    elif distribution == "Binomial":
        st.markdown("""
        - **Discrete distribution** counting successes in `n` trials.
        - **Mean** = n·p, **Variance** = n·p(1-p).
        - Used in A/B testing, quality control.
        """)
    elif distribution == "Poisson":
        st.markdown("""
        - Models **rare events** (e.g., calls per hour, accidents per day).
        - **Mean = Variance = λ**.
        - Used in queuing theory, insurance, and event modeling.
        """)
    elif distribution == "Chi-Squared (χ²)":
        st.markdown("""
        - Sum of squares of `k` independent standard normal variables.
        - Used in **hypothesis testing** (e.g., goodness-of-fit tests).
        - **Mean = k**, **Variance = 2k**.
        """)
    elif distribution == "Log-Normal":
        st.markdown("""
        - **Right-skewed**, models variables that are multiplicative products.
        - Used in finance (stock prices), biology (size distributions).
        - If X ~ Lognormal(μ, σ), then ln(X) ~ Normal(μ, σ).
        """)
    elif distribution == "Weibull":
        st.markdown("""
        - Models **failure rates** (reliability engineering).
        - **k < 1**: Decreasing failure rate (early failures).
        - **k = 1**: Exponential distribution (constant failure rate).
        - **k > 1**: Increasing failure rate (aging/wear-out).
        """)
    elif distribution == "Exponential":
        st.markdown("""
        - Models **time between events** in a Poisson process.
        - **Memoryless property**: P(X > s + t | X > s) = P(X > t).
        - Used in survival analysis, queuing theory.
        """)
    elif distribution == "Beta":
        st.markdown("""
        - Defined on **[0, 1]**, flexible shapes.
        - Used in Bayesian statistics (prior for binomial probability).
        - **α=β=1**: Uniform distribution.
        """)
    elif distribution == "Gamma":
        st.markdown("""
        - Generalization of **Exponential & Chi-Squared** distributions.
        - Models **waiting times** for multiple Poisson events.
        - Used in insurance claims, rainfall modeling.
        """)
else:
    st.warning("Please select a valid distribution.")


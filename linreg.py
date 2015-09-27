# load numpy and pandas for data manipulation
import numpy as np
import pandas as pd


# load statsmodels as alias ``sm``
import statsmodels.api as sm

# load the longley dataset into a pandas data frame - first column (year) used as row labels
df = pd.read_csv('https://github.com/rhanberry/linear-regression-script-project/blob/master/data_folder/cancer_tax_state.csv', index_col = 0)
df.head()

y = df.Rate  # response
X = df.TAX  # predictor
X = sm.add_constant(X)  # Adds a constant term to the predictor
X.head()


est = sm.OLS(y, X)


est = est.fit()
est.summary()


est.params


# Make sure that graphics appear inline in the iPython notebook
import pylab

# We pick 100 hundred points equally spaced from the min to the max
X_prime = np.linspace(X.TAX.min(), X.TAX.max(), 100)[:, np.newaxis]
X_prime = sm.add_constant(X_prime)  # add constant as we did before

# Now we calculate the predicted values
y_hat = est.predict(X_prime)

plt.scatter(X.GNP, y, alpha=0.3)  # Plot the raw data
plt.xlabel("Cigarette excise tax")
plt.ylabel("Lung and bronchus cancer rate")
plt.plot(X_prime[:, 1], y_hat, 'r', alpha=0.9)  # Add the regression line, colored in red


# import formula api as alias smf
import statsmodels.formula.api as smf

# formula: response ~ predictors
est = smf.ols(formula='Rate ~ TAX', data=df).fit()
est.summary()



# Fit the no-intercept model
est_no_int = smf.ols(formula='Employed ~ GNP - 1', data=df).fit()

# We pick 100 hundred points equally spaced from the min to the max
X_prime_1 = pd.DataFrame({'GNP': np.linspace(X.GNP.min(), X.GNP.max(), 100)})
X_prime_1 = sm.add_constant(X_prime_1)  # add constant as we did before

y_hat_int = est.predict(X_prime_1)
y_hat_no_int = est_no_int.predict(X_prime_1)

fig = plt.figure(figsize=(8,4))
splt = plt.subplot(121)

splt.scatter(X.GNP, y, alpha=0.3)  # Plot the raw data
plt.ylim(30, 100)  # Set the y-axis to be the same
plt.xlabel("Cigarette excise tax")
plt.ylabel("Lung and bronchus cancer rate")
plt.title("With intercept")
splt.plot(X_prime[:, 1], y_hat_int, 'r', alpha=0.9)  # Add the regression line, colored in red

splt = plt.subplot(122)
splt.scatter(X.GNP, y, alpha=0.3)  # Plot the raw data
plt.xlabel("Cigarette excise tax")
plt.ylabel("Lung and bronchus cancer rate")
plt.title("Without intercept")
splt.plot(X_prime[:, 1], y_hat_no_int, 'r', alpha=0.9)  # Add the regression line, colored in red
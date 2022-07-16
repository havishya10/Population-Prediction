import pandas as pd
import matplotlib.pyplot as plt

# function to train the model
def populationreg(x_train, y_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    return reg


country = input("input the Country Name: ").title()
year = int(input("Input the year for the population to be predicted: "))
data = pd.read_csv('unpop.csv')
country_list = data['Country Name'].tolist()

if country in country_list:
    data = data.loc[data['Country Name'] == country]
    data.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    data = data.T
    data.dropna(inplace=True)
    data = data.reset_index()
    x1 = data.iloc[:, 0]
    y1 = data.iloc[:, 1]
    x = x1.to_numpy().reshape(-1, 1).astype(int)
    y = y1.to_numpy().reshape(-1, 1).astype(int)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    reg = populationreg(x_train, y_train)
    predict_pop = int(reg.coef_[0][0] * year + reg.intercept_[0])
    print("The expected population of %s by year %d is %d " % (country, year, predict_pop))
    # plotting graph
    plt.scatter(x_train, y_train, color="red")  # plotting actual observations ie training data
    plt.plot(x_train, reg.predict(x_train),color="green")  # plotting the regression line, x_train on x axis and predictions of x_train on y axia
    plt.title("Linear Regression model on population prediction")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.show()

else:
    print("Inputed country is not in the data base, Kindly check the spelling")

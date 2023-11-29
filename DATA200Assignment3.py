import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np


class AutoSidebar():
    def __init__(self):
        self.__sidebar = []

    def header(self, text, tag):
        st.header(text, tag)
        self.__sidebar.append({"type": "header", "text": f"[{text}](#{tag})"})

    def subheader(self, text, tag):
        st.subheader(text, tag)
        self.__sidebar.append({"type": "subheader", "text": f"[{text}](#{tag})"})

    def title(self, text, tag):
        st.title(text, tag)
        self.__sidebar.append({"type": "title", "text": f"[{text}](#{tag})"})

    def make(self):
        for s in self.__sidebar:
            if s["type"] == "header":
                st.sidebar.header(s["text"])
            elif s["type"] == "subheader":
                st.sidebar.subheader("> "+s["text"])
            else:
                st.sidebar.title(s["text"])

sidebar = AutoSidebar()

df = pd.read_csv("Price.csv")
sidebar.title("House Price Prediction", "home")
st.write("Objective: Build a classification model that predicts house prices. The target column to predict is 'Price'")
st.write("To get started, lets take a look at the dataset")
st.write("Here is the dataset in full")
st.write(df)
st.write("Summary analysis of each feature of dataset")
st.write(df.describe())

condition_counts = df['condition'].value_counts().sort_index()
bedrooms_counts = df['bedrooms'].value_counts().sort_index()


# Create a bar chart for the distribution of property conditions
st.markdown("---")
sidebar.header("Features", "Features")
sidebar.subheader("Condition", "Condition")
st.write("Distribution of Property Conditions")
fig, ax = plt.subplots()
condition_counts.plot(kind='bar', color='skyblue')
plt.title("Distribution of Property Conditions")
plt.xlabel('Condition')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Rotate x-axis labels if needed
st.pyplot(fig)  # Display the Matplotlib plot in Streamlit
st.write("""As we can see from our above chart, houses which are in condition 3 are more common and houses which are in
condition 1 are in least number""")

sidebar.subheader("Bedrooms", "Bedrooms")
st.write("Distribution of Number of bedrooms")
fig, ax = plt.subplots()
ax.bar(bedrooms_counts.index, bedrooms_counts)
plt.title("Number of properties by bedroom count")
plt.xlabel("Bedroom number")
plt.ylabel("Count")
plt.xticks(rotation=0)
st.pyplot(fig)
st.write("""As we can see from our above chart, houses which have 3 bedrooms are most common and houses which
have bedrooms 9 are least common""")


st.markdown("---")
sidebar.header("Correlation Heatmap of all Numerical Features", "Correlation_Heatmap")
st.write("Correlation Heatmap for all the numerical features")
# Create a subset DataFrame with only the numerical columns
numerical_df = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]
fig, ax = plt.subplots()
# Create the correlation heatmap
sns.heatmap(numerical_df.corr(), annot=True, vmin = -1, vmax = 1, cmap='coolwarm', fmt=".2f", annot_kws={"size": 5}, linewidths=0.5, square=True)
plt.title('Correlation Heatmap')
st.pyplot(fig)
st.write("""The heatmap reveals that the pairs of variables are positively correlated (indicated by darker shades),
such as price with itself and other attributes with itself. It also shows negative correlations between floors and condition and between condition and year built.""")

st.markdown("---")
sidebar.header("Distribution for Price", "Price_Distribution")
st.write("The distribution of the numerical feature “price”")
#create price dataframe
price = df['price']
fig, ax = plt.subplots()
sns.histplot(price, color='blue')
# Add labels and a title
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Price')
st.pyplot(fig)
st.write("""The histogram reveals that the distribution of property prices is right skewed, meaning that most properties have lower prices, and there are relatively fewer properties with
extremely high prices. According to the skewness, the market is characterized by a broader range of property values. The mean price is higher than the median, indicating the influence of higher-priced
properties on the distribution. Data quality issues are evident in properties with a recorded price of $0, which requires further investigation.""")

st.markdown("---")
sidebar.header("Covariance matrix of the numerical features", "Covariance_Matrix")
st.write("Covariance matrix of the numerical features")
numerical_features = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]  # Select numerical columns
cov_matrix = numerical_features.cov()
sns.heatmap(cov_matrix, annot=True,  cmap='coolwarm', linewidths=0.5, annot_kws={"size": 5})
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('covariance matrix Heatmap')
st.pyplot(fig)
st.write("""This suggests that "price" does not have strong linear relationships with the other variables, and they are not highly correlated in terms of covariance.
This can be valuable information when analyzing the relationships between these variables, as it implies that changes in the other variables are not strongly associated with changes in the price.""")

st.markdown("---")
sidebar.header("Interactive Linear Regression", "Linear_Regression")
drop=st.selectbox("What type of feature you want to choose?",options=("sqft_lot", "sqft_living", "yr_built"),index=0,help="Choose a feature option in dropdown")

    
if drop=="sqft_lot":
    # Choose the feature to plot against 'price'
    feature_name = 'sqft_lot'
    fig, ax = plt.subplots()
    # Create the scatter plot
    sns.scatterplot(x=feature_name, y='price', data=df, alpha=0.7, color='b')
    # Fit a regression line
    sns.regplot(x=feature_name, y='price', data=df, scatter=False, color='r')
    # Set labels and title
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature_name} vs. Price')
    # Show the plot
    sqft_lot = plt.show()
    st.pyplot(sqft_lot)
    st.write("""The scatter plot shows a positive correlation between the size of the lot (sqft_lot) and the price of the house (price). This means that larger plots tend to be more expensive.
     There are a few possible explanations for this. First, larger lots may be more desirable to buyers because they offer more space for outdoor activities, such as gardening, entertaining, or
      building a pool. Second, larger lots may be in more desirable neighborhoods, which can drive up prices.
     Third, larger lots may be more expensive to develop, which can also contribute to higher prices. The scatter plot also shows a lot of variation in prices, even for houses with similar lot sizes.
     This suggests that there are other factors that also influence house prices, such as the condition of the house, the number of bedrooms and bathrooms, and the location of the house.""")

elif drop=="sqft_living":
    # Choose the feature to plot against 'price'
    feature_name = 'sqft_living'
    fig, ax = plt.subplots()
    # Create the scatter plot
    sns.scatterplot(x=feature_name, y='price', data=df, alpha=0.7, color='b')
    # Fit a regression line
    sns.regplot(x=feature_name, y='price', data=df, scatter=False, color='r')
    # Set labels and title
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature_name} vs. Price')
    # Show the plot
    sqft_living = plt.show()
    st.pyplot(sqft_living)
    st.write("""A positive correlation (coefficient of 0.43) between sqft_living and price indicates a moderate positive relationship between the two variables:
     The positive value indicates there is a positive correlation. A value of 0 means no correlation, while +1/-1 indicate perfect positive/negative correlations.
     So, a value of 0.43 is reasonably close to 0, indicating a moderate positive correlation. What this signifies is,  As the sqft_living increases, price also tends to increase on average. Bigger
     homes sell for more. But the relationship is not extremely strong, since 0.43 is moderately distant from 1. There is some variance in price not explained directly by living space. Other factors
     also significantly influence home price besides just size. Location, economic trends, interest rates etc. likely also play a role. So, in summary, a 0.43 correlation between sqft_living and price
     indicates that larger homes moderately tend to sell for more on average, but living space is not the only driver of home prices in this market.
     """)


elif drop=="yr_built":
    # Choose the feature to plot against 'price'
    feature_name = 'yr_built'
    fig, ax = plt.subplots()
    # Create the scatter plot
    sns.scatterplot(x=feature_name, y='price', data=df, alpha=0.7, color='b')
    # Fit a regression line
    sns.regplot(x=feature_name, y='price', data=df, scatter=False, color='r')
    # Set labels and title
    plt.xlabel(feature_name)
    plt.ylabel('Price')
    plt.title(f'Scatter Plot of {feature_name} vs. Price')
    # Show the plot
    yr_built = plt.show()
    st.pyplot(yr_built)
    st.write("""A correlation coefficient of 0.02 between yr_built and price indicates almost no correlation between the two variables:
       The value is very close to 0, which indicates no linear relationship. Positive values between 0 and 0.3 indicate a weak positive correlation. So, what this signifies is that knowing when the
       home was built does not help in predicting how much the home will sell for. The year built alone has minimal correlation with the final sales price. This also suggests factors like size,
       neighborhood, renovations etc. matter more than age in determining prices. A home built in 1950 may sell for more than one built in 1995 depending on other attributes. In summary, a 0.02
       correlation coefficient tells us that year built does not have a measurable correlation with sale price in this data. The age of the home alone cannot predict the final sales value well. """)



sidebar.make()

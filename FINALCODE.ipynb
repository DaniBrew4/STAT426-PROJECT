{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. Pose a prediction question that can be answered with data and a machine learning model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve\n",
    "import sklearn\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import preprocessing\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.pipeline import make_pipeline\n",
    "import requests\n",
    "import pandas as pd\n",
    "from pandas.io.json import json_normalize"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Part 1\n",
    "Walk through the data science process:\n",
    "\n",
    "### 1. Pose a prediction question that can be answered with data and a machine learning model\n",
    "Can I predict wine quality from its chemical features and whether or not it comes from it the wine or red data set?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Collect data to answer your question via webscraping, APIs and/or combining several readily available dataset (i.e. kaggle, uci ML repo, etc.)\n",
    "I had to change to a ready dataset from the web because the original datasets on \"https://data.cityofnewyork.us/data.json\" moved. The order of their data sets changed and one of the ones that I was using was removed from the website on December 9th."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'\n",
    "red_wine = pd.read_csv(url, sep = ';')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'\n",
    "white_wine = pd.read_csv(url2, sep = ';')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.88</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.6</td>\n",
       "      <td>0.098</td>\n",
       "      <td>25.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>0.9968</td>\n",
       "      <td>3.20</td>\n",
       "      <td>0.68</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>7.8</td>\n",
       "      <td>0.76</td>\n",
       "      <td>0.04</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0.092</td>\n",
       "      <td>15.0</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0.9970</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.65</td>\n",
       "      <td>9.8</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>11.2</td>\n",
       "      <td>0.28</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.075</td>\n",
       "      <td>17.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>0.9980</td>\n",
       "      <td>3.16</td>\n",
       "      <td>0.58</td>\n",
       "      <td>9.8</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.4</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.00</td>\n",
       "      <td>1.9</td>\n",
       "      <td>0.076</td>\n",
       "      <td>11.0</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.9978</td>\n",
       "      <td>3.51</td>\n",
       "      <td>0.56</td>\n",
       "      <td>9.4</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0            7.4              0.70         0.00             1.9      0.076   \n",
       "1            7.8              0.88         0.00             2.6      0.098   \n",
       "2            7.8              0.76         0.04             2.3      0.092   \n",
       "3           11.2              0.28         0.56             1.9      0.075   \n",
       "4            7.4              0.70         0.00             1.9      0.076   \n",
       "\n",
       "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "1                 25.0                  67.0   0.9968  3.20       0.68   \n",
       "2                 15.0                  54.0   0.9970  3.26       0.65   \n",
       "3                 17.0                  60.0   0.9980  3.16       0.58   \n",
       "4                 11.0                  34.0   0.9978  3.51       0.56   \n",
       "\n",
       "   alcohol  quality  \n",
       "0      9.4        5  \n",
       "1      9.8        5  \n",
       "2      9.8        5  \n",
       "3      9.8        6  \n",
       "4      9.4        5  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "red_wine.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>quality</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7.0</td>\n",
       "      <td>0.27</td>\n",
       "      <td>0.36</td>\n",
       "      <td>20.7</td>\n",
       "      <td>0.045</td>\n",
       "      <td>45.0</td>\n",
       "      <td>170.0</td>\n",
       "      <td>1.0010</td>\n",
       "      <td>3.00</td>\n",
       "      <td>0.45</td>\n",
       "      <td>8.8</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>6.3</td>\n",
       "      <td>0.30</td>\n",
       "      <td>0.34</td>\n",
       "      <td>1.6</td>\n",
       "      <td>0.049</td>\n",
       "      <td>14.0</td>\n",
       "      <td>132.0</td>\n",
       "      <td>0.9940</td>\n",
       "      <td>3.30</td>\n",
       "      <td>0.49</td>\n",
       "      <td>9.5</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8.1</td>\n",
       "      <td>0.28</td>\n",
       "      <td>0.40</td>\n",
       "      <td>6.9</td>\n",
       "      <td>0.050</td>\n",
       "      <td>30.0</td>\n",
       "      <td>97.0</td>\n",
       "      <td>0.9951</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.44</td>\n",
       "      <td>10.1</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7.2</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.32</td>\n",
       "      <td>8.5</td>\n",
       "      <td>0.058</td>\n",
       "      <td>47.0</td>\n",
       "      <td>186.0</td>\n",
       "      <td>0.9956</td>\n",
       "      <td>3.19</td>\n",
       "      <td>0.40</td>\n",
       "      <td>9.9</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.2</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.32</td>\n",
       "      <td>8.5</td>\n",
       "      <td>0.058</td>\n",
       "      <td>47.0</td>\n",
       "      <td>186.0</td>\n",
       "      <td>0.9956</td>\n",
       "      <td>3.19</td>\n",
       "      <td>0.40</td>\n",
       "      <td>9.9</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "0            7.0              0.27         0.36            20.7      0.045   \n",
       "1            6.3              0.30         0.34             1.6      0.049   \n",
       "2            8.1              0.28         0.40             6.9      0.050   \n",
       "3            7.2              0.23         0.32             8.5      0.058   \n",
       "4            7.2              0.23         0.32             8.5      0.058   \n",
       "\n",
       "   free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "0                 45.0                 170.0   1.0010  3.00       0.45   \n",
       "1                 14.0                 132.0   0.9940  3.30       0.49   \n",
       "2                 30.0                  97.0   0.9951  3.26       0.44   \n",
       "3                 47.0                 186.0   0.9956  3.19       0.40   \n",
       "4                 47.0                 186.0   0.9956  3.19       0.40   \n",
       "\n",
       "   alcohol  quality  \n",
       "0      8.8        6  \n",
       "1      9.5        6  \n",
       "2     10.1        6  \n",
       "3      9.9        6  \n",
       "4      9.9        6  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "white_wine.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4898 entries, 0 to 4897\n",
      "Data columns (total 12 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   fixed acidity         4898 non-null   float64\n",
      " 1   volatile acidity      4898 non-null   float64\n",
      " 2   citric acid           4898 non-null   float64\n",
      " 3   residual sugar        4898 non-null   float64\n",
      " 4   chlorides             4898 non-null   float64\n",
      " 5   free sulfur dioxide   4898 non-null   float64\n",
      " 6   total sulfur dioxide  4898 non-null   float64\n",
      " 7   density               4898 non-null   float64\n",
      " 8   pH                    4898 non-null   float64\n",
      " 9   sulphates             4898 non-null   float64\n",
      " 10  alcohol               4898 non-null   float64\n",
      " 11  quality               4898 non-null   int64  \n",
      "dtypes: float64(11), int64(1)\n",
      "memory usage: 459.3 KB\n"
     ]
    }
   ],
   "source": [
    "white_wine.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1599 entries, 0 to 1598\n",
      "Data columns (total 12 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   fixed acidity         1599 non-null   float64\n",
      " 1   volatile acidity      1599 non-null   float64\n",
      " 2   citric acid           1599 non-null   float64\n",
      " 3   residual sugar        1599 non-null   float64\n",
      " 4   chlorides             1599 non-null   float64\n",
      " 5   free sulfur dioxide   1599 non-null   float64\n",
      " 6   total sulfur dioxide  1599 non-null   float64\n",
      " 7   density               1599 non-null   float64\n",
      " 8   pH                    1599 non-null   float64\n",
      " 9   sulphates             1599 non-null   float64\n",
      " 10  alcohol               1599 non-null   float64\n",
      " 11  quality               1599 non-null   int64  \n",
      "dtypes: float64(11), int64(1)\n",
      "memory usage: 150.0 KB\n"
     ]
    }
   ],
   "source": [
    "red_wine.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Clean / wrangle your data\n",
    "### 4. Create features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Adding a categorical variable\n",
    "red_wine['type'] = \"red\"  \n",
    "white_wine['type'] = \"white\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "wine = pd.concat([red_wine,white_wine])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 6497 entries, 0 to 4897\n",
      "Data columns (total 13 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   fixed acidity         6497 non-null   float64\n",
      " 1   volatile acidity      6497 non-null   float64\n",
      " 2   citric acid           6497 non-null   float64\n",
      " 3   residual sugar        6497 non-null   float64\n",
      " 4   chlorides             6497 non-null   float64\n",
      " 5   free sulfur dioxide   6497 non-null   float64\n",
      " 6   total sulfur dioxide  6497 non-null   float64\n",
      " 7   density               6497 non-null   float64\n",
      " 8   pH                    6497 non-null   float64\n",
      " 9   sulphates             6497 non-null   float64\n",
      " 10  alcohol               6497 non-null   float64\n",
      " 11  quality               6497 non-null   int64  \n",
      " 12  type                  6497 non-null   object \n",
      "dtypes: float64(11), int64(1), object(1)\n",
      "memory usage: 710.6+ KB\n"
     ]
    }
   ],
   "source": [
    "wine.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Explore the data through Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Scale the data with the MaxMinScaler from sklearn.preprocessing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.cluster import KMeans"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 7.4 ,  0.7 ,  0.  , ...,  0.56,  9.4 ,  5.  ],\n",
       "       [ 7.8 ,  0.88,  0.  , ...,  0.68,  9.8 ,  5.  ],\n",
       "       [ 7.8 ,  0.76,  0.04, ...,  0.65,  9.8 ,  5.  ],\n",
       "       ...,\n",
       "       [ 6.5 ,  0.24,  0.19, ...,  0.46,  9.4 ,  6.  ],\n",
       "       [ 5.5 ,  0.29,  0.3 , ...,  0.38, 12.8 ,  7.  ],\n",
       "       [ 6.  ,  0.21,  0.38, ...,  0.32, 11.8 ,  6.  ]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "numWine = wine.loc[:,'fixed acidity':'quality']\n",
    "numWine.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0.29752066, 0.41333333, 0.        , ..., 0.19101124, 0.20289855,\n",
       "        0.33333333],\n",
       "       [0.33057851, 0.53333333, 0.        , ..., 0.25842697, 0.26086957,\n",
       "        0.33333333],\n",
       "       [0.33057851, 0.45333333, 0.02409639, ..., 0.24157303, 0.26086957,\n",
       "        0.33333333],\n",
       "       ...,\n",
       "       [0.2231405 , 0.10666667, 0.11445783, ..., 0.13483146, 0.20289855,\n",
       "        0.5       ],\n",
       "       [0.14049587, 0.14      , 0.18072289, ..., 0.08988764, 0.69565217,\n",
       "        0.66666667],\n",
       "       [0.18181818, 0.08666667, 0.22891566, ..., 0.05617978, 0.55072464,\n",
       "        0.5       ]])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "scaler = MinMaxScaler()\n",
    "scaled = scaler.fit_transform(numWine)\n",
    "X = scaled\n",
    "X"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Perform a cluster analysis on these wines using k-means.\n",
    "##### Look at a plot of WCSS versus number of clusters to help choose the optimal number of clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "wcss = []\n",
    "for k in range(1,20):\n",
    "    kmeans = KMeans(n_clusters = k)\n",
    "    kmeans.fit(X)\n",
    "    wcss.append(kmeans.inertia_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 0, 'Number of clusters')"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(np.arange(1,20), wcss, marker = \"o\")\n",
    "plt.xlabel('Number of clusters')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#####  To me it seems that the elbow is closest to 5 groups or clusters"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Make a plot of the first two principal components colored by predicted cluster label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.collections.PathCollection at 0x7f4a741cea10>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.decomposition import PCA \n",
    "pca = PCA(n_components = 2)\n",
    "pca.fit(X)\n",
    "pccomp = pca.transform(X)\n",
    "plt.scatter(pccomp[:,0], pccomp[:,1], c = kmeans.labels_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### It apears that there is not much seperation between the groups (at least in a 2D array). Let's see if we can by way of some models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6. Analyze the data with multiple machine learning approaches"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 6497 entries, 0 to 4897\n",
      "Data columns (total 13 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   fixed acidity         6497 non-null   float64\n",
      " 1   volatile acidity      6497 non-null   float64\n",
      " 2   citric acid           6497 non-null   float64\n",
      " 3   residual sugar        6497 non-null   float64\n",
      " 4   chlorides             6497 non-null   float64\n",
      " 5   free sulfur dioxide   6497 non-null   float64\n",
      " 6   total sulfur dioxide  6497 non-null   float64\n",
      " 7   density               6497 non-null   float64\n",
      " 8   pH                    6497 non-null   float64\n",
      " 9   sulphates             6497 non-null   float64\n",
      " 10  alcohol               6497 non-null   float64\n",
      " 11  quality               6497 non-null   int64  \n",
      " 12  type                  6497 non-null   object \n",
      "dtypes: float64(11), int64(1), object(1)\n",
      "memory usage: 710.6+ KB\n"
     ]
    }
   ],
   "source": [
    "wine.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#SPLIT the data into testing data and training data\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n",
    "from sklearn.linear_model import LinearRegression, Lasso, Ridge\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "from sklearn.tree import DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(6497, 13)\n",
      "(6497,)\n"
     ]
    }
   ],
   "source": [
    "y = wine.loc[:,'quality']\n",
    "wine2 = wine.drop('quality', axis=1)\n",
    "x = wine2\n",
    "x = pd.get_dummies(x)\n",
    "print(x.shape)\n",
    "print(y.shape)\n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>fixed acidity</th>\n",
       "      <th>volatile acidity</th>\n",
       "      <th>citric acid</th>\n",
       "      <th>residual sugar</th>\n",
       "      <th>chlorides</th>\n",
       "      <th>free sulfur dioxide</th>\n",
       "      <th>total sulfur dioxide</th>\n",
       "      <th>density</th>\n",
       "      <th>pH</th>\n",
       "      <th>sulphates</th>\n",
       "      <th>alcohol</th>\n",
       "      <th>type_red</th>\n",
       "      <th>type_white</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>536</th>\n",
       "      <td>7.9</td>\n",
       "      <td>0.345</td>\n",
       "      <td>0.51</td>\n",
       "      <td>15.3</td>\n",
       "      <td>0.047</td>\n",
       "      <td>54.0</td>\n",
       "      <td>171.0</td>\n",
       "      <td>0.99870</td>\n",
       "      <td>3.09</td>\n",
       "      <td>0.51</td>\n",
       "      <td>9.1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3479</th>\n",
       "      <td>5.9</td>\n",
       "      <td>0.320</td>\n",
       "      <td>0.28</td>\n",
       "      <td>4.7</td>\n",
       "      <td>0.039</td>\n",
       "      <td>34.0</td>\n",
       "      <td>94.0</td>\n",
       "      <td>0.98964</td>\n",
       "      <td>3.22</td>\n",
       "      <td>0.57</td>\n",
       "      <td>13.1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3234</th>\n",
       "      <td>6.6</td>\n",
       "      <td>0.250</td>\n",
       "      <td>0.34</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.054</td>\n",
       "      <td>22.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>0.99338</td>\n",
       "      <td>3.26</td>\n",
       "      <td>0.47</td>\n",
       "      <td>10.4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3597</th>\n",
       "      <td>6.6</td>\n",
       "      <td>0.190</td>\n",
       "      <td>0.28</td>\n",
       "      <td>11.8</td>\n",
       "      <td>0.042</td>\n",
       "      <td>54.0</td>\n",
       "      <td>137.0</td>\n",
       "      <td>0.99492</td>\n",
       "      <td>3.18</td>\n",
       "      <td>0.37</td>\n",
       "      <td>10.8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4531</th>\n",
       "      <td>7.3</td>\n",
       "      <td>0.280</td>\n",
       "      <td>0.54</td>\n",
       "      <td>12.9</td>\n",
       "      <td>0.049</td>\n",
       "      <td>62.0</td>\n",
       "      <td>162.5</td>\n",
       "      <td>0.99840</td>\n",
       "      <td>3.06</td>\n",
       "      <td>0.45</td>\n",
       "      <td>9.1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1512</th>\n",
       "      <td>7.3</td>\n",
       "      <td>0.220</td>\n",
       "      <td>0.49</td>\n",
       "      <td>9.4</td>\n",
       "      <td>0.034</td>\n",
       "      <td>29.0</td>\n",
       "      <td>134.0</td>\n",
       "      <td>0.99390</td>\n",
       "      <td>2.99</td>\n",
       "      <td>0.32</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2579</th>\n",
       "      <td>6.4</td>\n",
       "      <td>0.280</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.7</td>\n",
       "      <td>0.156</td>\n",
       "      <td>49.0</td>\n",
       "      <td>106.0</td>\n",
       "      <td>0.99354</td>\n",
       "      <td>3.10</td>\n",
       "      <td>0.37</td>\n",
       "      <td>9.2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3310</th>\n",
       "      <td>6.3</td>\n",
       "      <td>0.300</td>\n",
       "      <td>0.29</td>\n",
       "      <td>2.1</td>\n",
       "      <td>0.048</td>\n",
       "      <td>33.0</td>\n",
       "      <td>142.0</td>\n",
       "      <td>0.98956</td>\n",
       "      <td>3.22</td>\n",
       "      <td>0.46</td>\n",
       "      <td>12.9</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1177</th>\n",
       "      <td>7.1</td>\n",
       "      <td>0.660</td>\n",
       "      <td>0.00</td>\n",
       "      <td>2.4</td>\n",
       "      <td>0.052</td>\n",
       "      <td>6.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.99318</td>\n",
       "      <td>3.35</td>\n",
       "      <td>0.66</td>\n",
       "      <td>12.7</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1692</th>\n",
       "      <td>7.0</td>\n",
       "      <td>0.240</td>\n",
       "      <td>0.30</td>\n",
       "      <td>4.2</td>\n",
       "      <td>0.040</td>\n",
       "      <td>41.0</td>\n",
       "      <td>213.0</td>\n",
       "      <td>0.99270</td>\n",
       "      <td>3.28</td>\n",
       "      <td>0.49</td>\n",
       "      <td>11.8</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5197 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  \\\n",
       "536             7.9             0.345         0.51            15.3      0.047   \n",
       "3479            5.9             0.320         0.28             4.7      0.039   \n",
       "3234            6.6             0.250         0.34             3.0      0.054   \n",
       "3597            6.6             0.190         0.28            11.8      0.042   \n",
       "4531            7.3             0.280         0.54            12.9      0.049   \n",
       "...             ...               ...          ...             ...        ...   \n",
       "1512            7.3             0.220         0.49             9.4      0.034   \n",
       "2579            6.4             0.280         0.56             1.7      0.156   \n",
       "3310            6.3             0.300         0.29             2.1      0.048   \n",
       "1177            7.1             0.660         0.00             2.4      0.052   \n",
       "1692            7.0             0.240         0.30             4.2      0.040   \n",
       "\n",
       "      free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  \\\n",
       "536                  54.0                 171.0  0.99870  3.09       0.51   \n",
       "3479                 34.0                  94.0  0.98964  3.22       0.57   \n",
       "3234                 22.0                 141.0  0.99338  3.26       0.47   \n",
       "3597                 54.0                 137.0  0.99492  3.18       0.37   \n",
       "4531                 62.0                 162.5  0.99840  3.06       0.45   \n",
       "...                   ...                   ...      ...   ...        ...   \n",
       "1512                 29.0                 134.0  0.99390  2.99       0.32   \n",
       "2579                 49.0                 106.0  0.99354  3.10       0.37   \n",
       "3310                 33.0                 142.0  0.98956  3.22       0.46   \n",
       "1177                  6.0                  11.0  0.99318  3.35       0.66   \n",
       "1692                 41.0                 213.0  0.99270  3.28       0.49   \n",
       "\n",
       "      alcohol  type_red  type_white  \n",
       "536       9.1         0           1  \n",
       "3479     13.1         0           1  \n",
       "3234     10.4         0           1  \n",
       "3597     10.8         0           1  \n",
       "4531      9.1         0           1  \n",
       "...       ...       ...         ...  \n",
       "1512     11.0         0           1  \n",
       "2579      9.2         0           1  \n",
       "3310     12.9         0           1  \n",
       "1177     12.7         1           0  \n",
       "1692     11.8         0           1  \n",
       "\n",
       "[5197 rows x 13 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1420    4\n",
       "1796    5\n",
       "4433    5\n",
       "4656    7\n",
       "383     6\n",
       "       ..\n",
       "705     5\n",
       "1007    5\n",
       "1601    6\n",
       "4625    5\n",
       "662     6\n",
       "Name: quality, Length: 1300, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Test Linear Regression, KNN, and Decision Tree Regressor Models"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 7. Evaluate each model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Linear Regression Model's Test score:  0.30746325191269097\n",
      "Linear Regression Model's Train score:  0.3719299743574145\n",
      "KNN Regression Model's Test score:  0.3444642454391532\n",
      "KNN Regression Model's Train score:  0.5763907539054585\n",
      "DT Regression Model's Test score:  0.07186808090241703\n",
      "DT Regression Model's Train score:  1.0\n"
     ]
    }
   ],
   "source": [
    "lr_pipe = Pipeline(steps = [(\"feature\",PolynomialFeatures(interaction_only=True)),\n",
    "                         (\"scalar\",StandardScaler()),(\"model\",LinearRegression(n_jobs=-1))])\n",
    "lr_pipe.fit(x_train,y_train)\n",
    "knn_pipe = Pipeline(steps = [(\"feature\",PolynomialFeatures(interaction_only=True)),\n",
    "                         (\"scalar\",StandardScaler()),(\"model\",KNeighborsRegressor(n_jobs=-1))])\n",
    "knn_pipe.fit(x_train,y_train)\n",
    "dt_pipe = Pipeline(steps = [(\"feature\",PolynomialFeatures(interaction_only=True)),\n",
    "                         (\"scalar\",StandardScaler()),(\"model\",DecisionTreeRegressor())])\n",
    "dt_pipe.fit(x_train,y_train)\n",
    "print(\"Linear Regression Model's Test score: \",lr_pipe.score(x_test,y_test))\n",
    "print(\"Linear Regression Model's Train score: \",lr_pipe.score(x_train,y_train))\n",
    "print(\"KNN Regression Model's Test score: \",knn_pipe.score(x_test,y_test))\n",
    "print(\"KNN Regression Model's Train score: \",knn_pipe.score(x_train,y_train))\n",
    "print(\"DT Regression Model's Test score: \",dt_pipe.score(x_test,y_test))\n",
    "print(\"DT Regression Model's Train score: \",dt_pipe.score(x_train,y_train))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[21:52:02] WARNING: ../src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.\n",
      "Training Accuracy: 0.990379064845103\n",
      "Testing Accuracy: 0.6915384615384615\n"
     ]
    }
   ],
   "source": [
    "#Further Model Evaluation\n",
    "from xgboost import XGBClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "xgb = XGBClassifier(nthread=-1)\n",
    "xgb.fit(x_train, y_train)\n",
    "\n",
    "y_pred = xgb.predict(x_test)\n",
    "xgb_test_predictions = [round(value) for value in y_pred]\n",
    "    \n",
    "y_pred = xgb.predict(x_train)\n",
    "xgb_train_predictions = [round(value) for value in y_pred]\n",
    "\n",
    "print(\"Training Accuracy:\", accuracy_score(xgb_train_predictions, y_train)) #Accuracy of the model when training.\n",
    "print(\"Testing Accuracy:\", accuracy_score(xgb_test_predictions, y_test)) # Accuracy of the test.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf = RandomForestClassifier(n_estimators = 20, random_state = 42,n_jobs=-1)\n",
    "rf.fit(x_train, y_train); #n_estimators is the number of decision trees being used."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Testing Accuracy: 0.6684615384615384\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"Testing Accuracy:\", rf.score(x_test, y_test) )#Accuracy of the test.\n",
    "\n",
    "train_score_array = []\n",
    "test_score_array = []\n",
    "\n",
    "for k in range (1, 21):\n",
    "    rf = RandomForestClassifier(n_estimators = k, random_state = 42,n_jobs=-1)\n",
    "    rf.fit(x_train, y_train)\n",
    "    train_score_array.append(rf.score(x_train, y_train))\n",
    "    test_score_array.append(rf.score(x_test, y_test))\n",
    "x_axis = range(1,21) # x_axis values\n",
    "%matplotlib inline\n",
    "#x-values, y-values, Name for legend, color\n",
    "plt.plot(x_axis, train_score_array , label = \"Train Score\", c= \"g\") #Plots a green line\n",
    "plt.plot(x_axis, test_score_array, label = \"Test Score\", c= \"b\")  #Plots a blue line\n",
    "plt.xlabel('# of decision trees')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "y_pred = rf.predict(x_test)\n",
    "rf_test_pred = [round(value) for value in y_pred]\n",
    "\n",
    "y_pred = rf.predict(x_train)\n",
    "rf_train_pred = [round(value) for value in y_pred]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "While KNN, Linear Regression and DT didn't do that well, XGB Boost and Random Forest Classifiers did much better. They both were prone to over fitting the training data, but were able to get up towards 70% accuracy on the testing data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 8. Answer the original question"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Although I can predict wine quality from it's chemical features and whether it was red or white wine....I cannot do it very well. I had hoped that adding the feature for red or white wine would help increase the predicting power."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 9. Understand and explain potential sources of bias in how your data/model answers your question of interest"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are several possible sources of Bias in my models. One of them is that they are slightly over fitting the training data as you can see in the graph above, but there isn't a strong downward curve in the testing data, which may also indicate that I didn't over fit the data. Another possible source of bias in my model is multicollinearity. I didn't check to see if some of the features needed to be dropped or combined due to hiving a high multilinearity. This is very possible in some of the fields like fixed acidity and PH. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Communicate the highlights of your work in a markdown report (this should be the Readme file of a Github repository)\n",
    "### Post all your work (including clean, well-documented, and reproducible code) in a public Github repository"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Part 2\n",
    "Record a 5 minute video presentation of your project. Your presentation should be self contained, meaning that someone could watch your video and know the main purpose and conclusions of your work without having seen your Github repository. You should clearly define the question you are answer / purpose of the project as well as the main highlights and/or conclusions.\n",
    "\n",
    "Rules\n",
    "\n",
    "### Your project should be original for this class\n",
    "### Your project should be individual work\n",
    "### You can work with a group (of no more than 4 people) for data collection, but you must pose your own research question\n",
    "### Your project should be original. You can be inspired by something you have seen on kaggle, Github, another class, work, etc. but should be original work.\n",
    "### For full points, don’t just use a readily available dataset but either:\n",
    "    combine two or more readily available sources of data, OR\n",
    "    collect your own data\n",
    "###  #1 and #2 will be due by November 20\n",
    "#### The final project will be due December 10"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

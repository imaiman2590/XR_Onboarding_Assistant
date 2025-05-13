# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, 100)
        self.lstm = nn.LSTM(100, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm(x)
        x = self.dropout(hn[-1])  # Use the last hidden state
        x = self.fc(x)
        return x

# Custom Dataset
class QueryDataset(Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

# Function to train the model
def train_model(data):
    # Prepare data
    X = data['query']
    y = data['chart_type']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create a bag of words model
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X).toarray()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.LongTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)

    # Create the model
    hidden_size = 64  # Adjusted hidden size
    model = LSTMClassifier(input_size=len(vectorizer.vocabulary_), hidden_size=hidden_size, num_classes=len(label_encoder.classes_))
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50  # Increase epochs for better training
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training completed.")
    return model, vectorizer, label_encoder

# Function to make predictions
def predict(model, vectorizer, label_encoder, query):
    model.eval()
    query_vectorized = vectorizer.transform([query]).toarray()
    query_tensor = torch.LongTensor(query_vectorized)
    with torch.no_grad():
        output = model(query_tensor)
    _, predicted = torch.max(output, 1)
    return label_encoder.inverse_transform(predicted.numpy())[0]

# Expanded sample data for training
data = {
    'query': [
        'show me a bar chart', 
        'give me a line plot of sales', 
        'create a pie chart for expenses',
        'generate a scatter plot for sales vs profit', 
        'point plot of profits over time',
        'give me a histogram of age distribution',
        'display a box plot for the data',
        'show a dual axis chart for revenue and expenses',
        'plot a heatmap of correlation',
        'create a stacked bar chart of categories',
        'line chart for monthly sales',
        'scatter plot comparing temperature and sales',
        'show the percentage distribution of categories in a pie chart',
        'display trends over the years in a line plot',
        'give me a scatter plot for height vs weight',
        'show a bar graph of total sales by category',
        'create a multi-line chart to compare sales across different regions',
        'show a waterfall chart for profit analysis',
        'plot the distribution of salaries with a box plot',
        'show me a donut chart of market share',
        'create a radar chart for performance metrics',
        'give me a violin plot of age by gender',
        'display a funnel chart for sales conversion rates',
        'show a 3D scatter plot of data points',
        'generate a time series chart for stock prices',
        'create a Gantt chart for project timelines',
        'display a Pareto chart for defects in production',
        'show a candlestick chart for stock prices',
        'create a geographical heat map for sales data',
        'plot a bubble chart to represent sales and profit',
        'generate a slope chart for comparing two periods',
        'show me a line chart for website traffic over a month',
        'display a stacked area chart for energy consumption',
        'create a bar chart comparing sales of different products',
        'show the correlation between two variables with a scatter plot',
        'give me a pie chart showing market share by company',
        'create a time series analysis of temperature changes',
        'display a line graph of user registrations over time',
        'show a histogram of test scores',
        'create a box plot to compare heights of students in different classes',
        'plot a heatmap of the performance of different sales teams',
        'show a candlestick chart of stock prices over the last year',
        'create a map visualizing sales by region',
        # Additional data
        'show me a stacked line chart for different regions',
        'create a bubble chart to show profits and sales',
        'display a column chart of sales by month',
        'generate a box plot for sales data',
        'show the trend of expenses over the years',
        'display a line chart of revenue growth',
        'create a chart comparing actual vs target sales',
        'show a radar chart for product features rating',
        'create a heatmap for customer satisfaction levels',
        'plot a line graph for average temperatures over years',
        'display the sales distribution in a histogram',
        'generate a bar graph for total units sold by product',
        'show the breakdown of expenses in a pie chart',
        'create a line chart for daily active users',
        'display a candlestick chart for cryptocurrency prices',
        'plot the average sales per region with a bubble chart',
        'show a time series forecast for next quarter sales',
        'create a heatmap for website traffic by hour of the day',
        'display a dual-axis chart for costs and revenue over time',
        'generate a 3D surface plot for sales data',
        'show me the distribution of user ages in a histogram',
        'create a funnel chart to visualize conversion rates'
    ],
    'chart_type': [
        'bar', 'line', 'pie', 'scatter', 'point',
        'histogram', 'box', 'dual', 'heatmap', 'stacked_bar',
        'line', 'scatter', 'pie', 'line', 'scatter', 'bar',
        'line', 'waterfall', 'box', 'pie', 'radar',
        'violin', 'funnel', 'scatter3d', 'time_series', 'gantt',
        'pareto', 'candlestick', 'heatmap', 'bubble', 'slope',
        'line', 'area', 'bar', 'scatter', 'pie',
        'line', 'line', 'box', 'heatmap', 'candlestick', 'map',
        'line', 'bubble', 'column', 'box', 'line',
        'line', 'bar', 'radar', 'heatmap', 'line',
        'line', 'pie', 'line', 'candlestick', 'bubble',
        'line', 'heatmap', 'dual', 'surface', 'histogram',
        'bar', 'pie', 'line', 'candlestick', 'bubble'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Train the model
model, vectorizer, label_encoder = train_model(df)

# Save the model, vectorizer, and label encoder
torch.save(model.state_dict(), 'chart_model.pth')
import pickle
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

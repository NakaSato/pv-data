import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Configure matplotlib for better font handling
def configure_fonts():
    # List of fonts that typically have good Unicode support
    supported_fonts = ['Roboto', 'Segoe UI', 'Tahoma', 'DejaVu Sans', 'Liberation Sans']
    
    # Enable fontconfig for better system font discovery
    plt.rcParams['svg.fonttype'] = 'none'
    
    # Check which of these fonts are available on the system
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in supported_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = font
            print(f"Using {font} for better Unicode character support")
            return font
    
    # If none of the preferred fonts are available, use the default sans-serif
    plt.rcParams['font.family'] = 'sans-serif'
    return 'sans-serif'

# Helper function to safely format labels with special characters
def safe_label(text):
    # Replace degree celsius symbol with plain text if needed
    return text.replace('°C', ' Celsius').replace('\N{DEGREE CELSIUS}', ' Celsius')

# Configure fonts before loading data
font_name = configure_fonts()

# Load the data - first peek at the file to understand the actual format
try:
    # First read the file without parsing dates to inspect the format
    df_peek = pd.read_csv('./raw/inverter/processed/processed_INVERTER_02_2025-04-04_2025-04-05.csv', 
                          nrows=5, low_memory=False)
    
    print("Sample Time values:", df_peek['Time'].tolist())
    
    # Now load the full data with the appropriate date parsing
    df = pd.read_csv('./raw/inverter/processed/processed_INVERTER_02_2025-04-04_2025-04-05.csv',
                    low_memory=False)
    
    # Convert the Time column to datetime after loading
    df['Time'] = pd.to_datetime(df['Time'], format='mixed', errors='coerce')
    
    # Check for and handle missing values after conversion
    print(f"Missing timestamps after conversion: {df['Time'].isna().sum()}")
    df.dropna(subset=['Time'], inplace=True)
    
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Data Preparation
# Select relevant columns
inverter_data = df[['Time', 'Pac(W)', 'Ppv(W)', 'VacR(V)', 'VacS(V)', 'VacT(V)', 
                   'VacRS(V)', 'VacST(V)', 'VacTR(V)', 'IacR(A)', 'IacS(A)', 
                   'IacT(A)', 'Fac(Hz)', 'INVTemp(℃)', 'PIDStatus']].copy()

# Rename columns using the mapping provided
inverter_data = inverter_data.rename(columns={
    'Time': 'Timestamp',
    'Pac(W)': 'AC_Power',
    'Ppv(W)': 'DC_Power', 
    'VacR(V)': 'Voltage_AC_R',
    'VacS(V)': 'Voltage_AC_S',
    'VacT(V)': 'Voltage_AC_T',
    'VacRS(V)': 'Voltage_AC_RS',
    'VacST(V)': 'Voltage_AC_ST',
    'VacTR(V)': 'Voltage_AC_TR',
    'IacR(A)': 'Current_AC_R',
    'IacS(A)': 'Current_AC_S',
    'IacT(A)': 'Current_AC_T',
    'Fac(Hz)': 'Frequency',
    'INVTemp(℃)': 'Temperature',
    'PIDStatus': 'Status'
})

# Calculate average voltage and current for simplicity
inverter_data['Voltage_AC'] = inverter_data[['Voltage_AC_R', 'Voltage_AC_S', 'Voltage_AC_T']].mean(axis=1)
inverter_data['Current_AC'] = inverter_data[['Current_AC_R', 'Current_AC_S', 'Current_AC_T']].mean(axis=1)
inverter_data.columns = ['Timestamp', 'AC_Power', 'DC_Power', 'Voltage_AC', 'Current_AC', 'Frequency', 'Temperature', 'Status']

# Convert timestamp to datetime and set as index
inverter_data['Timestamp'] = pd.to_datetime(inverter_data['Timestamp'])
inverter_data.set_index('Timestamp', inplace=True)

# Filter out nighttime data (when AC_Power is 0)
inverter_data = inverter_data[inverter_data['AC_Power'] > 0]

# Feature Engineering
# Calculate efficiency (DC to AC conversion)
inverter_data['Efficiency'] = inverter_data['AC_Power'] / inverter_data['DC_Power'] * 100

# Handle any infinite values from division
inverter_data.replace([np.inf, -np.inf], np.nan, inplace=True)
inverter_data.dropna(inplace=True)

# Visualize the data
plt.figure(figsize=(15, 8))
plt.plot(inverter_data.index, inverter_data['AC_Power'], label='AC Power (W)')
plt.plot(inverter_data.index, inverter_data['DC_Power'], label='DC Power (W)')
plt.title('Solar Inverter Power Output')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.legend()
plt.grid()
plt.show()

# Prepare data for LSTM
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), :])
        y.append(data[i+look_back, 0])  # Predicting AC_Power
    return np.array(X), np.array(y)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(inverter_data[['AC_Power', 'DC_Power', 'Voltage_AC', 
                                                'Current_AC', 'Frequency', 'Temperature', 'Efficiency']])

# Split into train and test sets
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

# Create sequences
look_back = 60  # Number of previous time steps to use
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Create PyTorch DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Get output from last time step
        out = self.fc(out)
        return out

# Initialize model
input_size = X_train.shape[2]  # Number of features
hidden_size = 50
num_layers = 1
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping variables
patience = 5
min_val_loss = float('inf')
counter = 0
early_stop = False
best_model_state = None

# Training history
train_losses = []
val_losses = []

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss = val_loss / len(test_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Early stopping
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        counter = 0
        best_model_state = model.state_dict().copy()
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            model.load_state_dict(best_model_state)
            early_stop = True
            break

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
model.eval()
with torch.no_grad():
    train_predict = model(X_train).numpy()
    test_predict = model(X_test).numpy()

# Inverse transform predictions
train_predict_full = np.zeros((train_predict.shape[0], scaled_data.shape[1]))
train_predict_full[:, 0] = train_predict.flatten()
train_predict = scaler.inverse_transform(train_predict_full)[:, 0]

y_train_full = np.zeros((y_train.shape[0], scaled_data.shape[1]))
y_train_full[:, 0] = y_train.numpy().flatten()
y_train = scaler.inverse_transform(y_train_full)[:, 0]

test_predict_full = np.zeros((test_predict.shape[0], scaled_data.shape[1]))
test_predict_full[:, 0] = test_predict.flatten()
test_predict = scaler.inverse_transform(test_predict_full)[:, 0]

y_test_full = np.zeros((y_test.shape[0], scaled_data.shape[1]))
y_test_full[:, 0] = y_test.numpy().flatten()
y_test = scaler.inverse_transform(y_test_full)[:, 0]

# Calculate RMSE
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# Plot predictions
plt.figure(figsize=(15, 8))
plt.plot(inverter_data.index[look_back:train_size-1], y_train, label='Actual Train')
plt.plot(inverter_data.index[look_back:train_size-1], train_predict, label='Predicted Train')
plt.plot(inverter_data.index[train_size+look_back:-1], y_test, label='Actual Test')
plt.plot(inverter_data.index[train_size+look_back:-1], test_predict, label='Predicted Test')
plt.title('Solar Inverter AC Power Prediction')
plt.xlabel('Time')
plt.ylabel(safe_label('AC Power (W)'))
plt.legend()
plt.grid()
plt.show()

# Save the model for future use
torch.save(model.state_dict(), 'solar_inverter_lstm.pth')
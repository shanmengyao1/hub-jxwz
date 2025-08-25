import torch
import numpy as np
import matplotlib.pyplot as plt


# 生成 sin(x) 数据

x_min, x_max = -2 * np.pi, 2 * np.pi
X_numpy = np.linspace(x_min, x_max, 200).reshape(-1, 1)  # 200 个点
y_numpy = np.sin(X_numpy)  # 对应的 sin(x)




X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin(x) 数据生成完成。")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)



class SinNet(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(SinNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)



model = SinNet(hidden_dim=64)
print("模型结构：")
print(model)
print("---" * 10)


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 3000
for epoch in range(num_epochs):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 300 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)


model.eval()
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.plot(X_numpy, y_numpy, 'b-', label='True sin(x)', linewidth=2)
plt.plot(X_numpy, y_predicted, 'r--', label='Fitted by Neural Network', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Fitting sin(x)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


final_loss = loss_fn(model(X), y).item()
print(f"最终 MSE 损失: {final_loss:.6f}")
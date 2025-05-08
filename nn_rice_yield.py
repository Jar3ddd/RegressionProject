import pandas as pd
import numpy as np

import jax.numpy as jnp
from jax import random, jit, value_and_grad

import flax.linen as nn
from flax.training import train_state

import optax
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt


df_full    = pd.read_csv('data/preprocessed_rice_data.csv')     # for overall stats at the end
df_train   = pd.read_csv('data/training_data.csv')
df_test    = pd.read_csv('data/testing_data.csv')

cat_cols = ['Rice_Cultivar', 'Replicate']
df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
df_full = pd.get_dummies(df_full, columns=cat_cols, drop_first=True)

feature_cols = [
    'Nitrogen_Rate','Heading_100','Thermal','NDWI','EVI','NAVI','GNDVI','Cigreen',
    'RENDVI','TGI','SAVI','CI_RedEdge','BI','SCI','GLI','NGRDI','SI','VARI',
    'HUE','BGI','PSRI','RVI','TVI','CVI','NDVI','DVI','NRCT','Vegetation_Fraction',
    'contrast_90','dissimilarity_90','homogeneity_90','ASM_90','energy_90','correlation_90'
] + [c for c in df_train.columns if c.startswith('Rice_Cultivar_') or c.startswith('Replicate_')]


X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train['Yield'].values.astype(np.float32).reshape(-1,1)

X_test  = df_test[feature_cols].values.astype(np.float32)
y_test  = df_test['Yield'].values.astype(np.float32).reshape(-1,1)

X_full = df_full[feature_cols].values.astype(np.float32)
y_full = df_full['Yield'].values.astype(np.float32).reshape(-1,1)


scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train = scaler_X.transform(X_train)
X_test  = scaler_X.transform(X_test)

y_train = scaler_y.transform(y_train)
y_test  = scaler_y.transform(y_test)


class MLP(nn.Module):
    hidden_dims: list = (16, 16, 8)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def train_epoch(state, X, y, batch_size=32):
    n = X.shape[0]
    batch_losses = []

    for start in range(0, n, batch_size):
        end = start + batch_size
        Xb = jnp.array(X[start:end])
        yb = jnp.array(y[start:end])
        state, loss = train_step(state, (Xb, yb))
        batch_losses.append(loss)

    batch_losses = jnp.stack(batch_losses)
    return state, jnp.mean(batch_losses), batch_losses

def eval_epoch(state, X, y, batch_size=32):
    n = X.shape[0]
    batch_losses = []

    for start in range(0, n, batch_size):
        end = start + batch_size
        Xb = jnp.array(X[start:end])
        yb = jnp.array(y[start:end])
        loss = eval_step(state.params, (Xb, yb))
        batch_losses.append(loss)

    batch_losses = jnp.stack(batch_losses)
    return jnp.mean(batch_losses), batch_losses  


def create_train_state(rng, learning_rate):
    model = MLP()
    params = model.init(rng, jnp.ones([1, X_train.shape[1]]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def mse_loss(params, batch):
    Xb, yb = batch
    preds = MLP().apply({'params': params}, Xb)
    return jnp.mean((preds - yb)**2)

@jit
def train_step(state, batch):
    loss, grads = value_and_grad(mse_loss)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jit
def eval_step(params, batch):
    return mse_loss(params, batch)


# Hyperparams
rng = random.PRNGKey(0)
state = create_train_state(rng, learning_rate=1e-3)
batch_size = 32
num_epochs = 50

X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
X_test  = jnp.array(X_test)
y_test  = jnp.array(y_test)

n_train = X_train.shape[0]

for epoch in range(1, num_epochs+1):
    # Shuffle the training set
    perm = random.permutation(rng, n_train)    # retruns a vector
    rng, _ = random.split(rng)
    X_shuf = X_train[perm]
    y_shuf = y_train[perm]

    state, train_loss, train_losses = train_epoch(state, X_shuf, y_shuf, batch_size)
    test_loss, test_losses   = eval_epoch(state, X_test, y_test, batch_size)

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")



# Predict on test set
preds = MLP().apply({'params': state.params}, X_test)
preds = scaler_y.inverse_transform(np.array(preds))
y_true = scaler_y.inverse_transform(np.array(y_test))

rmse = np.sqrt(mean_squared_error(y_true, preds))
mae  = mean_absolute_error(y_true, preds)
mape = mean_absolute_percentage_error(y_true, preds)*100
print(f"\nTest RMSE: {rmse:.3f}")
print(f"Test MAE : {mae:.3f}")
print(f"Test MAPE: {mape:.1f}%")

#mean_yield = np.mean(y_true)
#var_yield  = np.var(y_true, ddof=0)  

mean_yield   = df_full['Yield'].mean()
var_yield    = df_full['Yield'].var(ddof=0)

print(f"Mean Yield    : {mean_yield:.3f} t/ha")
print(f"Yield Variance: {var_yield:.3f} (t/ha)^2")
print("Yield range:", df_full['Yield'].min(), "to", df_full['Yield'].max(), "t/ha")

# Quick scatter
print("\nClose Predicted vs True yield plot to continue with the K-fold Test")
plt.scatter(y_true, preds, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--')
plt.xlabel("True Yield")
plt.ylabel("Predicted Yield")
plt.title("MLP Regression Performance")
plt.show()



# K-fold Test

K = 10 # Number of folds
kf = KFold(n_splits=K, shuffle=True, random_state=0)

fold_rmse = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_full, y_full), start=1):
    print(f"\n=== Fold {fold}/{K} ===")

    X_tr_raw, X_te_raw = X_full[train_idx], X_full[test_idx]
    y_tr_raw, y_te_raw = y_full[train_idx], y_full[test_idx]

    scaler_X = StandardScaler().fit(X_tr_raw)
    scaler_y = StandardScaler().fit(y_tr_raw)
    X_tr = scaler_X.transform(X_tr_raw)
    X_te = scaler_X.transform(X_te_raw)
    y_tr = scaler_y.transform(y_tr_raw)
    y_te = scaler_y.transform(y_te_raw)

    X_tr_j, y_tr_j = jnp.array(X_tr), jnp.array(y_tr)
    X_te_j, y_te_j = jnp.array(X_te), jnp.array(y_te)

    rng = random.PRNGKey(fold)  
    state = create_train_state(rng, learning_rate=1e-3)

    for epoch in range(1, num_epochs+1):
        rng, perm_key = random.split(rng)
        perm = random.permutation(perm_key, X_tr_j.shape[0])
        X_shuf = X_tr_j[perm]
        y_shuf = y_tr_j[perm]

        state, train_loss, _ = train_epoch(state, X_shuf, y_shuf, batch_size)

    test_loss, _ = eval_epoch(state, X_te_j, y_te_j, batch_size)

    preds = MLP().apply({'params': state.params}, X_te_j)
    preds = scaler_y.inverse_transform(np.array(preds)).ravel()
    y_true = scaler_y.inverse_transform(np.array(y_te_j)).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    print(f"Fold {fold} RMSE: {rmse:.4f}")
    fold_rmse.append(rmse)

# Summarize across folds
mean_rmse = np.mean(fold_rmse)
std_rmse  = np.std(fold_rmse)
print(f"\nK‑Fold CV results: mean RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")

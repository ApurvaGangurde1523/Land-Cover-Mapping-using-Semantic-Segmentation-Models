{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "afd0b226-3d2c-4a90-b4a9-fdd850c97769",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n",
    "from sklearn.utils import shuffle\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "import torch, torch.nn as nn\n",
    "from torch.utils.data import DataLoader, TensorDataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e0af0e37-39cc-4ae0-b053-4de72634526e",
   "metadata": {},
   "outputs": [],
   "source": [
    "path = r\"C:\\Users\\Apurva Gangurde\\Downloads\\archive (2)\\rainfall in india 1901-2015.csv\"\n",
    "df = pd.read_csv(path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "005f6046-25ad-4e5d-ba25-bf18e540f475",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.fillna(df.mean(numeric_only=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ee9d1e37-e003-42f8-ac73-2c89afe8e60b",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.random.seed(42)\n",
    "df[\"Temperature\"] = 28 + 2*np.sin(2*np.pi*(df[\"YEAR\"]%12)/12) + np.random.normal(0,1,len(df))\n",
    "df[\"SoilMoisture\"] = 0.35 + np.random.normal(0,0.03,len(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5a961256-3262-4be0-85c0-5177d392df6f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data prepared: (4080, 27)\n"
     ]
    }
   ],
   "source": [
    "df[\"RainAnomaly\"] = df.groupby(\"SUBDIVISION\")[\"ANNUAL\"].transform(lambda x: x - x.mean())\n",
    "df[\"DroughtIndex\"] = -df[\"RainAnomaly\"] / df[\"RainAnomaly\"].abs().max()\n",
    "df[\"PrevRain\"] = df.groupby(\"SUBDIVISION\")[\"ANNUAL\"].shift(1)\n",
    "df[\"PrevDrought\"] = df.groupby(\"SUBDIVISION\")[\"DroughtIndex\"].shift(1)\n",
    "df[\"Rain5yrAvg\"] = df.groupby(\"SUBDIVISION\")[\"ANNUAL\"].transform(lambda x: x.rolling(5,1).mean())\n",
    "df[\"RainAnomalyRatio\"] = df[\"RainAnomaly\"] / (df[\"PrevRain\"] + 1e-6)\n",
    "df = df.dropna().reset_index(drop=True)\n",
    "\n",
    "print(\"Data prepared:\", df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d5114175-9790-4a10-ba4f-846e7c19e742",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "humid = [\"KERALA\",\"ASSAM & MEGHALAYA\",\"WEST BENGAL\",\"ODISHA\"]\n",
    "semi_arid = [\"GUJARAT REGION\",\"SAURASHTRA & KUTCH\",\"WEST RAJASTHAN\",\"EAST RAJASTHAN\"]\n",
    "arid = [\"PUNJAB\",\"HARYANA\",\"BIHAR\",\"EAST UTTAR PRADESH\",\"WEST UTTAR PRADESH\"]\n",
    "peninsular = [\"VIDARBHA\",\"MARATHWADA\",\"MADHYA MAHARASHTRA\",\"CHHATTISGARH\",\"TELANGANA\",\"RAYALSEEMA\"]\n",
    "\n",
    "clusters = {\n",
    "    \"Humid\": humid,\n",
    "    \"SemiArid\": semi_arid,\n",
    "    \"Arid\": arid,\n",
    "    \"Peninsular\": peninsular\n",
    "}\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "37f3ea44-8047-4b83-a176-d310b7f6c724",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "class DroughtLSTM(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):\n",
    "        super().__init__()\n",
    "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "        self.fc = nn.Linear(hidden_size, 1)\n",
    "    def forward(self, x):\n",
    "        out, _ = self.lstm(x)\n",
    "        out = self.dropout(out[:, -1, :])\n",
    "        return self.fc(out)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1324072b-fb87-466c-ba57-bb4db28f698a",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def train_lstm(X_train, y_train, input_size, epochs=60, seq_len=10):\n",
    "    DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "    # Prepare sequences\n",
    "    Xs, ys = [], []\n",
    "    for i in range(len(X_train)-seq_len):\n",
    "        Xs.append(X_train[i:i+seq_len])\n",
    "        ys.append(y_train[i+seq_len])\n",
    "    Xs, ys = np.array(Xs), np.array(ys)\n",
    "\n",
    "    ds = TensorDataset(torch.Tensor(Xs), torch.Tensor(ys))\n",
    "    loader = DataLoader(ds, batch_size=32, shuffle=True)\n",
    "\n",
    "    model = DroughtLSTM(input_size=input_size).to(DEVICE)\n",
    "    opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
    "    loss_fn = nn.MSELoss()\n",
    "\n",
    "    for ep in range(1, epochs+1):\n",
    "        model.train(); total=0\n",
    "        for xb, yb in loader:\n",
    "            xb, yb = xb.to(DEVICE), yb.to(DEVICE)\n",
    "            opt.zero_grad()\n",
    "            loss = loss_fn(model(xb), yb)\n",
    "            loss.backward(); opt.step()\n",
    "            total += loss.item()\n",
    "        if ep%10==0:\n",
    "            print(f\"Epoch {ep}/{epochs} | Loss: {total/len(loader):.5f}\")\n",
    "    return model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b6f76542-126d-4111-8e2e-44ac2483df28",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Training for cluster: Humid\n",
      "Epoch 10/60 | Loss: 0.02980\n",
      "Epoch 20/60 | Loss: 0.02864\n",
      "Epoch 30/60 | Loss: 0.02653\n",
      "Epoch 40/60 | Loss: 0.02881\n",
      "Epoch 50/60 | Loss: 0.02956\n",
      "Epoch 60/60 | Loss: 0.02558\n",
      " Humid Cluster: RMSE=0.0560 | R²=0.8081 | SMAPE=53.38% | Accuracy≈46.62%\n",
      "\n",
      " Training for cluster: SemiArid\n",
      "Epoch 10/60 | Loss: 0.02620\n",
      "Epoch 20/60 | Loss: 0.02917\n",
      "Epoch 30/60 | Loss: 0.02630\n",
      "Epoch 40/60 | Loss: 0.02687\n",
      "Epoch 50/60 | Loss: 0.02884\n",
      "Epoch 60/60 | Loss: 0.02922\n",
      " SemiArid Cluster: RMSE=0.0302 | R²=0.7917 | SMAPE=42.59% | Accuracy≈57.41%\n",
      "\n",
      " Training for cluster: Arid\n",
      "Epoch 10/60 | Loss: 0.02491\n",
      "Epoch 20/60 | Loss: 0.02648\n",
      "Epoch 30/60 | Loss: 0.02988\n",
      "Epoch 40/60 | Loss: 0.02591\n",
      "Epoch 50/60 | Loss: 0.02561\n",
      "Epoch 60/60 | Loss: 0.02727\n",
      " Arid Cluster: RMSE=0.0187 | R²=0.8765 | SMAPE=28.07% | Accuracy≈71.93%\n",
      "\n",
      " Training for cluster: Peninsular\n",
      "Epoch 10/60 | Loss: 0.03151\n",
      "Epoch 20/60 | Loss: 0.03261\n",
      "Epoch 30/60 | Loss: 0.03127\n",
      "Epoch 40/60 | Loss: 0.02972\n",
      "Epoch 50/60 | Loss: 0.03129\n",
      "Epoch 60/60 | Loss: 0.03163\n",
      " Peninsular Cluster: RMSE=0.0123 | R²=0.9335 | SMAPE=24.31% | Accuracy≈75.69%\n"
     ]
    }
   ],
   "source": [
    "\n",
    "results = []\n",
    "for cname, region_list in clusters.items():\n",
    "    print(f\"\\n Training for cluster: {cname}\")\n",
    "    data = df[df[\"SUBDIVISION\"].isin(region_list)].copy()\n",
    "\n",
    "    features = [\"ANNUAL\",\"PrevRain\",\"PrevDrought\",\"Rain5yrAvg\",\"RainAnomalyRatio\",\"Temperature\",\"SoilMoisture\"]\n",
    "    X = data[features].values\n",
    "    y = data[\"DroughtIndex\"].values.reshape(-1,1)\n",
    "\n",
    "    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()\n",
    "    X_scaled = scaler_x.fit_transform(X)\n",
    "    y_scaled = scaler_y.fit_transform(y)\n",
    "\n",
    "    split = int(0.8*len(X_scaled))\n",
    "    X_train, X_test = X_scaled[:split], X_scaled[split:]\n",
    "    y_train, y_test = y_scaled[:split], y_scaled[split:]\n",
    "\n",
    "    # Random Forest\n",
    "    rf = RandomForestRegressor(n_estimators=400, random_state=42)\n",
    "    rf.fit(X_train, y_train.ravel())\n",
    "    pred_rf = rf.predict(X_test).reshape(-1,1)\n",
    "\n",
    "    # LSTM\n",
    "    model = train_lstm(X_train, y_train, input_size=X.shape[1], epochs=60, seq_len=10)\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        seqs, ys = [], []\n",
    "        for i in range(len(X_test)-10):\n",
    "            seqs.append(X_test[i:i+10])\n",
    "            ys.append(y_test[i+10])\n",
    "        seqs, ys = np.array(seqs), np.array(ys)\n",
    "        preds_lstm = model(torch.Tensor(seqs)).cpu().numpy()\n",
    "\n",
    "    # Align predictions\n",
    "    min_len = min(len(pred_rf), len(preds_lstm))\n",
    "    y_true = scaler_y.inverse_transform(y_test[-min_len:])\n",
    "    pred_rf_i = scaler_y.inverse_transform(pred_rf[-min_len:])\n",
    "    pred_lstm_i = scaler_y.inverse_transform(preds_lstm[-min_len:])\n",
    "\n",
    "    # Ensemble\n",
    "    pred_ens = 0.7*pred_rf_i + 0.3*pred_lstm_i\n",
    "\n",
    "      # Ensemble evaluation (inside cluster loop)\n",
    "    rmse = np.sqrt(mean_squared_error(y_true, pred_ens))\n",
    "    r2 = r2_score(y_true, pred_ens)\n",
    "\n",
    "    # --- Fixed indentation for SMAPE ---\n",
    "    smape = np.mean(2.0 * np.abs(y_true - pred_ens) /\n",
    "                    (np.abs(y_true) + np.abs(pred_ens) + 1e-6)) * 100\n",
    "    acc = 100 - smape\n",
    "\n",
    "    print(f\" {cname} Cluster: RMSE={rmse:.4f} | R²={r2:.4f} | SMAPE={smape:.2f}% | Accuracy≈{acc:.2f}%\")\n",
    "    results.append([cname, rmse, r2, acc])\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "d8b2fe96-5e4a-4fd8-a4dd-6d858d89da82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Cluster-wise performance:\n",
      "      Cluster      RMSE        R2   Accuracy\n",
      "0       Humid  0.055999  0.808132  46.621629\n",
      "1    SemiArid  0.030233  0.791730  57.411157\n",
      "2        Arid  0.018737  0.876550  71.928644\n",
      "3  Peninsular  0.012255  0.933541  75.685251\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmoAAAHDCAYAAACUKTbEAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAOvtJREFUeJzt3Qm8lOP///FP+6bSoo02pEUkRUVlKbJLWUpU9E1IiYhQKctJUrakaEMLoWxfkUIh0SISJaKSytaiXc3/8b6+v3v+M3PmnM4y55yrc17Px2M6zXbPPfd9z32/72u784VCoZABAADAO/lzegYAAAAQH0ENAADAUwQ1AAAATxHUAAAAPEVQAwAA8BRBDQAAwFMENQAAAE8R1AAAADxFUAMAAPAUQQ1e+vnnny1fvnw2fPjwTE2na9euVqNGjTR/3sSJEzP1eQAsQ7+3+++/3z2Wl+n7azkAkQhqyBTtaLVzWbRoUdznzzzzTKtfv77lFh999JH7vq+++mqqr/vnn39s0KBB7ruXKFHCypUrZyeddJLdeuuttmHDhvCBKi03vTb4XN1eeumluJ95+umnu+cTtbxj56NUqVJ2xhln2DvvvJPidhDvdvfdd6f5M6+88kr3nrvuuish3wHZT9tqu3btrFKlSla4cGGrUKGCXXzxxfb666+br/773/9maUD66quv7JprrrGqVatakSJFrGzZsta6dWubMGGC7d+/37KD9jv6jpoXHFoK5vQMAFnpueeeswMHDmTrZ+7bt89atmxp33//vXXp0sV69erlgtu3335rU6ZMscsuu8xOOeUUe/HFF6Pe99hjj9n69ett5MiRUY8fccQRLqxJ0aJF3TS004+k5z/77DP3fCKdc8451rlzZ9MlgX/55RcbPXq0O+i+++671qZNm2SvHzJkiNWsWTPqsbQGx23bttlbb73lSkCnTp1qQ4cOzfMlLIcanZxoG6hVq5b16NHDqlevbn/++acLQu3bt7fJkyfb1VdfHfe99913X7pCfSJp/kaNGpUlYe3555+3G2+80SpWrGjXXnutWzbbt2+3OXPmWLdu3ey3336ze+65x7IjqA0ePNj9vnTSiEMHQQ250o4dO1xJVqFChbL9s2fOnGlLly6Ne1DavXu37d27181bbNiaNm2a/f3338kej3TBBRfYm2++aX/88YeVL18+/LjCmw4EOghoGgcr8TjrrLNszZo1B60WPu6446LmRwfbevXq2RNPPBE3qJ1//vnWuHFjy4jXXnvNlS6MHz/ezj77bJs3b54rwfONQqvWY7FixSwvOdj3VimzQtrll1/utsfI396dd95p7733njuJSUnBggXdLTf5/PPPXUhr1qyZC4MlS5YMP9enTx9XE7F8+XLLDftaZB2qPpGtdOBt0KBB3Odq164d9+CvEiadmesAoffH7tjUDu2www6zH3/80QUZ7Qw7deoUfi42jGzZssU9Xrp0aTv88MNdqZceSxTNR1AVGUslXqpCzKhLL73UVZ1Mnz496nEdGFVtWKBAActKdevWdQEx+I6JpGCrEjyFSH2O7sejkkp9V5U0apvQdnPvvfdGvebXX391pRVVqlRxy0ulfDfddJMLyam1hwqqcIMSTNH2c9FFF7mgoRCqzxwzZox7TlVXCpWq3tPnKMSq1DEelUJq+9X2qW1Apapab0FJlILN77//nux9N9xwg9tOFZJSEvwGfvrpJ/cb0oFT313BSQErkkqYH3/8cTv++OPd9qiAr9Kv2ICf2veOZ8CAAa5KT0E73gmS5kvTS0m8daL7t9xyi9vetWw1Dwo933zzjXte83Pssce676FmFpHrTebPn29XXHGFVatWza0fVT3edttttmvXrqhlp9K04POCW3qXVzwqwdK0tC1HhrSAlqs+P71tbOMtq9mzZ1vz5s3dtqJtQb+LoKROJ2fa3uS6664Lf8fINoILFy608847z+0Xixcv7rbVTz/9NO7nrlixwp2ElilTxn0mslbuOn1Bjtm6dasr5YkVewatov/u3bu7sBVZJfbll1/aqlWrXPVHpBdeeMFVE/Ts2dMdqFSSowOjdtTaYQb+/fdfdyDQTkMdELSjiUcHLYWdTz75xJ3pKhDMmDHDhbVEUagM5l3fJ5HVd/pemn9VDSp4yLJly1y1qqpYvv76a8vq9awD1DHHHJPm7SCy5C+1apkPP/zQJk2a5O537NjRBfSnn37atXMK6Pu1aNHCBQEFGB3EFBpVZfrQQw+Fp3Xqqae68K3X1KlTxwU3lfjs3LkzanpptXLlSjdPOkBr+9VBUBTKdAC/5JJLXGmQ5uPmm292B3dtswEdEK+//nr32v79+7uDqUpdZ82a5Q54+l0oVL388ssumAQULDXfKsk8WLW2SiN1oG3atKkNGzbMTVsBUL8NTTug76D50QG7d+/ermRVy1nzowNzZMhK6XvH+uGHH1yA1neMF0gyQ2FLpcjB8kxKSnKBr1+/fvbMM8+45a1tUt9Znz937tzwexXwtM71W1E70S+++MKeeuop18QgONnRd9M2o6AT2xwhvcsrkj5X1ZtqBqGgmJX0+9cyOfHEE926VihdvXp1OGhpP6fHBw4c6H4T+g3Jaaed5v5qmak0vFGjRm6byZ8/f/gkRMtfv6dICr8qvX/44YeTnQggC4SATJgwYYJ+panejj/++PDrt2zZEipatGjorrvuippO7969QyVKlAj9888/7v6aNWvce4sVKxZav359+HULFy50j992223hx7p06eIeu/vuu5PNn56rXr16+P7MmTPda4cNGxZ+7N9//w21aNHCPa7vk5oPP/zQvW769Okpvmbnzp2h2rVru9fps7t27RoaN25caNOmTalO+8ILL4ya15Q+9+233w7ly5cvtHbtWvfcnXfeGTr66KPd/88444yo5Z3atLSMU6PXdOvWLfT777+HNm/eHFq0aFHovPPOc48/+uijad4O0mL48OFuXW/bts3dX7VqlXvvjBkzol7XsmXLUMmSJUO//PJL1OMHDhwI/79z586h/Pnzh7788stknxO8btCgQXHnLfgekctG60SPzZo1K+66jtWmTZvw+gi2ec1zkyZNQrt27Upxvps1a+ZeE+n11193n611lprgN9CrV6+oaWubKly4sFuHMn/+fPe6yZMnR71f3y328dS+d6w33njDvXbkyJGhtAh+35G/t3jrRPeLFCkStT7GjBnjHq9UqVJ4e5H+/fsnW3fx1k9SUpL7/URuQz179oy7PaRnecVatmyZe82tt956kKUR/X21HFLaf6W0rLTcdT9Yz/Ho9xBvH6ftpFatWm67jdwetexq1qwZOuecc5J9bseOHdP8nZB5VH0iIVR1oDPS2JvO8CKpWD0oEQrOxFQSoJKEtm3bJmvroMeOPPLI8H2d2TVp0sS194gVlDClRu9TyUfka1VdqAb/iaLqGVUjqF2O6Gxc1XCVK1d2n7Nnz55MTf/cc891VUxq06ZlqL8q9UhJUMoV3HRfVAoR+bg6PMQaN26cq2JU1Z6qaVRCoJKM22+/Pc3bQVqoaujCCy8Ml8bobF1n95HVn6oWVLs1lZrEllAEpZYqyVIbQXV4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ccQR1rBhQ2vdurVVrVo16+YUAAAgj0lTidquXbvswQcfdEHsggsusHfffde2bNliBQoUsNWrV9ugQYOsZs2a7rnPP/88oTP466+/2jXXXGPlypWzYsWK2QknnGCLFi0KPx8KhWzgwIFWuXJl97wC4w8//JDQeQAAAPC2RO24446zZs2a2XPPPWfnnHOOFSpUKNlrVMI2ZcoU69Chg917773WvXv3TM/c33//baeffrqdddZZLhyq9E4hrEyZMuHXDBs2zJ588kmbNGmSC4sDBgywNm3a2IoVK6xo0aKZngcAAICcki+kIqmD+O6776xu3bppmuC+ffts7dq1dswxx2R65u6++2779NNPXVVrPJr1KlWqWN++fe2OO+5wj23dutUqVqxoEydOdKExLbZt22alS5d27y1VqlSm5xsAUpNvcL6cngWkIjTooIfFhBicb3C2fA7Sb1BokGWl9OSONFV9pjWkiUrbEhHS5M0337TGjRvbFVdcYRUqVHBt4VSqF1izZo1t3LjRVXcG9MWbNGliCxYsSHG6e/bscQsp8gYAAJBren3++++/NmrUKBei2rVrZ4899pjt3r07oTP3008/2ejRo61WrVr23nvv2U033WS9e/d21ZyikCYqQYuk+8Fz8SQlJblAF9zoBAEAAA75Xp+RFJhWrVrlQpqqO1944QXXyH/q1KkJmzn1MlWJ2sMPP+zuq0Rt+fLl9uyzz1qXLl0yPN3+/fvb7bffHr6vEjXCGgAAOGSD2owZM+yyyy4L33///fdt5cqVruenqAF/06ZNEzpz6slZr169ZNWwr732mvt/pUqV3N9Nmza51wZ0/6STTkpxukWKFHE3AACAXFH1OX78eGvbtq1t2LDB3T/55JPtxhtvtFmzZtlbb71l/fr1s1NOOSWhM6cenwqDkVSKV716dfd/9fJUWJszZ05U6djChQtdL1UAAIA8EdQUxjp27GhnnnmmPfXUUzZ27FjXU0FDcWhIDFUdaniORLrtttvcuGyq+tR4bZq+Prdnz57u+Xz58lmfPn3cGG/qePDNN99Y586dXU9QhUoAAIA800btqquuclWcKj3TX7UVUyeCrKISOlW5qk3ZkCFDXAna448/bp06dQq/RvOyY8cOu+GGG9wgvM2bN3elfIyhBgAA8sQ4avHMmzfPlWydd9559sADDxzSwYhx1ABkJ8ZR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/a4jV2q6du1qbdu2zbZ5QuZl9zpLyzEjryGo5eEfWnYEJI1U/dtvv1n9+vWz7DPwP+r5fNNNN1m1atVczysN+NymTRv79NNPs/RztX7PP//8ZI/36NHDDUY9ffr0NE/ryy+/tBtuuCHBc4iM0LWStf4uvPDCNL3+jjvuiBrPEtm3f9d+XDeNK3rsscfakCFD7N9//03I9J944gmbOHFiQqaFjCGoIUtpR6/AULBgrhgJxmvt27e3pUuX2qRJk2zVqlX25ptv2plnnml//vlnln6u1m9sl/ydO3fatGnTrF+/fjZ+/PiDTmPv3r3u7xFHHGHFixfPsnlF2o0bN8569epl8+bNsw0bNqT4Og0coFBw2GGHucvwIfudd9557oTphx9+sL59+7rSzUcffTRho+cfalfu2ft/+5PcgqCGFKstVAWlqqjYkrmHH37YjVGnH29w5nbnnXda2bJl7aijjrIJEyakWoz93//+14477jgrVqyYnXXWWe41yByVis6fP98eeeQRt0yrV6/uBn/WdW0vueSS8Gv+85//uDCkS7ycffbZtmzZsmTbgIKVSuV04L355ptt//79NmzYMBfIKlSoYA899NBBqyJVilavXj27++673YFe1/+LFGxLmpaudVe7du24VZ868LRs2dKKFi3qpjd79uwsWX6I9s8//9jLL7/sSmhVohZZohKUxL/77rvuqjAK6Z988kmyfYi2Gw1Grv2EApxCO6NBZY2gBF2/e62z1q1buxO1PXv2uJLOI4880kqUKGFNmjRx6y+g9ar1895771ndunXdbz4IfSnVyOjkr3fv3m59ap+vz9W6D2gd635Qsq/ft16f2v5C85BSqd3+/futW7duVrNmTXfM0L5CpXxp2Z/kFgQ1pMvcuXPd2bUOviNGjHADHF500UVWpkwZW7hwod14442uymv9+vVx368Ddrt27eziiy924U3BQQdzZI52sLppB6idczxXXHGFbd682R1gFy9ebCeffLK1atXK/vrrr/BrfvzxR/e8BoyeOnWqK1XRgVrr8+OPP3ZB8L777nPrOjV63zXXXOPOxlUtGm8nrGqylStXuvD19ttvJ3tel4LTtqLqHH3es88+a3fddVeGlg/S55VXXrE6deq4A57Wo8J7bMjS73bo0KH23Xff2YknnphsGo899phb73qvgpy2sxkzZmTjt8i7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",
      "text/plain": [
       "<Figure size 700x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Overall Average Accuracy: 62.91%\n"
     ]
    }
   ],
   "source": [
    "\n",
    "res_df = pd.DataFrame(results, columns=[\"Cluster\",\"RMSE\",\"R2\",\"Accuracy\"])\n",
    "print(\"\\n Cluster-wise performance:\")\n",
    "print(res_df)\n",
    "\n",
    "plt.figure(figsize=(7,5))\n",
    "plt.bar(res_df[\"Cluster\"], res_df[\"Accuracy\"], color=[\"teal\",\"orange\",\"green\",\"purple\"])\n",
    "plt.title(\"Hybrid LSTM+RF Accuracy per Climate Cluster\")\n",
    "plt.ylabel(\"Accuracy (%)\")\n",
    "plt.ylim(0,100)\n",
    "plt.show()\n",
    "\n",
    "overall_acc = res_df[\"Accuracy\"].mean()\n",
    "print(f\"\\n Overall Average Accuracy: {overall_acc:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "412b45f6-b28c-42b0-a3fb-7400a391098a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " Model saved as best_drought_model.pth\n"
     ]
    }
   ],
   "source": [
    "torch.save(model.state_dict(), \"best_drought_model.pth\")\n",
    "print(\" Model saved as best_drought_model.pth\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5a942a1-f39a-44fb-976c-b8c1572ff7a4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

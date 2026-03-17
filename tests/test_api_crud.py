from datetime import datetime, timezone


def test_user_register_and_login(client):
    user_body = {"username": "alice", "email": "alice@example.com", "password": "secret123"}
    resp = client.post("/api/v1/users/register", json=user_body)
    assert resp.status_code == 200
    user = resp.json()
    assert user["user_id"] > 0
    assert user["username"] == "alice"

    login_resp = client.post("/api/v1/auth/login", json={"username": "alice", "password": "secret123"})
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert token_data["token_type"] == "bearer"
    assert isinstance(token_data["access_token"], str)
    assert token_data["access_token"].count(".") == 2


def test_strategy_model_signal_trade_alert_flow(client):
    user_body = {"username": "bob", "email": "bob@example.com", "password": "pw123456"}
    user_resp = client.post("/api/v1/users/register", json=user_body)
    assert user_resp.status_code == 200
    user_id = user_resp.json()["user_id"]

    strategy_body = {"name": "Mean Reversion", "description": "test", "parameters": {"lookback": 20}}
    strat_resp = client.post(f"/api/v1/strategies/?user_id={user_id}", json=strategy_body)
    assert strat_resp.status_code == 200
    strategy_id = strat_resp.json()["strategy_id"]

    model_body = {
        "name": "xgb_v1",
        "model_type": "xgboost",
        "version": "1.0",
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.8,
        "auc": 0.95,
        "metadata": {"features": ["rsi", "macd"]},
    }
    model_resp = client.post("/api/v1/models/", json=model_body)
    assert model_resp.status_code == 200
    model_id = model_resp.json()["model_id"]

    ts = datetime.now(tz=timezone.utc).isoformat()
    signal_body = {
        "strategy_id": strategy_id,
        "model_id": model_id,
        "symbol": "BTCUSDT",
        "signal_type": "BUY",
        "confidence": 0.75,
        "timestamp": ts,
        "metadata": {"why": "unit test"},
    }
    sig_resp = client.post("/api/v1/signals/", json=signal_body)
    assert sig_resp.status_code == 200
    signal_id = sig_resp.json()["signal_id"]

    trade_body = {
        "signal_id": signal_id,
        "symbol": "BTCUSDT",
        "trade_type": "BUY",
        "price": 42000.0,
        "quantity": 0.01,
        "is_paper_trade": True,
    }
    trade_resp = client.post("/api/v1/trades/", json=trade_body)
    assert trade_resp.status_code == 200
    assert trade_resp.json()["status"] in ("EXECUTED", "PENDING")

    alert_body = {"user_id": user_id, "alert_type": "SYSTEM", "message": "Test alert"}
    alert_resp = client.post("/api/v1/alerts/", json=alert_body)
    assert alert_resp.status_code == 200
    assert alert_resp.json()["user_id"] == user_id

    get_sigs = client.get("/api/v1/signals/BTCUSDT?limit=10")
    assert get_sigs.status_code == 200
    assert get_sigs.json()["symbol"] == "BTCUSDT"
    assert len(get_sigs.json()["signals"]) >= 1

    get_trades = client.get("/api/v1/trades/BTCUSDT?limit=10")
    assert get_trades.status_code == 200
    assert get_trades.json()["symbol"] == "BTCUSDT"
    assert len(get_trades.json()["trades"]) >= 1

    get_alerts = client.get(f"/api/v1/alerts/{user_id}?limit=10")
    assert get_alerts.status_code == 200
    assert get_alerts.json()["user_id"] == user_id
    assert len(get_alerts.json()["alerts"]) >= 1


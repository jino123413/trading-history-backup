import ccxt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta
import traceback
import logging

# 로깅 설정 개선
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    try:
        exchange = ccxt.bitget()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        raise

def preprocess_data(df):
    # 기술적 지표 추가
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
    
    # 특성과 타겟 준비
    features = df[['rsi', 'macd', 'macd_signal']].fillna(0)
    target = (df['close'].shift(-1) > df['close']).astype(int)
    
    return features, target

def train_model(features, target):
    model = XGBClassifier(random_state=42)
    model.fit(features[:-1], target[:-1])
    return model

def generate_signals(model, features):
    return model.predict(features)

def backtest_with_fibonacci_dca(data, signals, initial_balance=1000):
    current_prices = data['close'].values
    entry_prices = [None] * len(current_prices)
    exit_prices = [None] * len(current_prices)
    trades = []
    
    position = None
    balance = initial_balance
    leverage = 3
    
    for i in range(len(data)):
        current_price = current_prices[i]
        current_time = data['timestamp'].iloc[i]
        
        if position is None and signals[i] == 1:
            # 포지션 진입
            position_size = (balance * 0.1) * leverage / current_price
            position = {
                'entry_price': current_price,
                'size': position_size,
                'entry_time': current_time,  # 진입 시간 저장
                'timestamp': current_time
            }
            entry_prices[i] = current_price
            trades.append({
                'entry_time': current_time,  # 진입 시간
                'exit_time': None,  # 종료 시간은 나중에 업데이트
                'type': 'Long',
                'entry_price': current_price,
                'exit_price': None,
                'size': position_size,
                'pnl': None,
                'balance': balance
            })
            
        elif position is not None:
            pnl = ((current_price - position['entry_price']) / position['entry_price']) * leverage
            
            if pnl >= 0.02 or pnl <= -0.01:
                exit_prices[i] = current_price
                realized_pnl = pnl * (balance * 0.1)
                balance += realized_pnl
                
                # 마지막 거래 업데이트
                trades[-1].update({
                    'exit_time': current_time,  # 종료 시간 추가
                    'exit_price': current_price,
                    'pnl': pnl,
                    'balance': balance
                })
                position = None
    
    return current_prices, entry_prices, exit_prices, trades

def analyze_trades(trades):
    try:
        if not trades:
            return default_analysis_result()

        # NumPy float64를 일반 float로 변환하는 함수
        def convert_np_float(value):
            if hasattr(value, 'item'):  # np.float64 타입 체크
                return float(value.item())
            return float(value) if value is not None else 0.0

        # 거래 데이터 변환
        converted_trades = []
        for trade in trades:
            converted_trade = {
                'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if trade['exit_time'] else None,
                'type': trade['type'],
                'entry_price': convert_np_float(trade['entry_price']),
                'exit_price': convert_np_float(trade['exit_price']) if trade['exit_price'] else None,
                'size': convert_np_float(trade['size']),
                'pnl': convert_np_float(trade['pnl']) if trade.get('pnl') else None,
                'balance': convert_np_float(trade['balance'])
            }
            converted_trades.append(converted_trade)

        # 분석 결과 계산
        completed_trades = [t for t in converted_trades if t['exit_price'] is not None]
        winning_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) < 0]
        
        initial_balance = convert_np_float(trades[0]['balance'])
        final_balance = convert_np_float(trades[-1]['balance'])
        
        # 잔고 이력 변환
        balance_history = [convert_np_float(t['balance']) for t in trades]
        timestamps = [t['entry_time'].strftime('%Y-%m-%d %H:%M:%S') for t in trades]

        analysis_result = {
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round((len(winning_trades) / len(completed_trades) * 100) if completed_trades else 0.0, 2),
            'avg_profit': round(sum(t['pnl'] for t in winning_trades) / len(winning_trades) * 100 if winning_trades else 0.0, 2),
            'avg_loss': round(abs(sum(t['pnl'] for t in losing_trades)) / len(losing_trades) * 100 if losing_trades else 0.0, 2),
            'profit_factor': round(sum(t['pnl'] for t in winning_trades) / abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf'), 2),
            'risk_reward_ratio': round((sum(t['pnl'] for t in winning_trades) / len(winning_trades)) / abs(sum(t['pnl'] for t in losing_trades) / len(losing_trades)) if losing_trades and winning_trades else 0.0, 2),
            'initial_balance': round(initial_balance, 2),
            'final_balance': round(final_balance, 2),
            'total_profit_amount': round(final_balance - initial_balance, 2),
            'total_profit_percentage': round(((final_balance - initial_balance) / initial_balance) * 100, 2),
            'timestamps': timestamps,
            'balance_history': balance_history,
            'trades': converted_trades
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error in analyze_trades: {str(e)}")
        logger.error(traceback.format_exc())
        return default_analysis_result()

def default_analysis_result():
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'avg_profit': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
        'risk_reward_ratio': 0.0,
        'initial_balance': 1000.0,
        'final_balance': 1000.0,
        'total_profit_amount': 0.0,
        'total_profit_percentage': 0.0,
        'timestamps': [],
        'balance_history': [],
        'trades': []
    }

if __name__ == "__main__":
    # 테스트 실행
    data = fetch_data()
    features, target = preprocess_data(data)
    model = train_model(features, target)
    signals = generate_signals(model, features)
    current_prices, entry_prices, exit_prices, trades = backtest_with_fibonacci_dca(data, signals)
    analysis = analyze_trades(trades)
    
    print("\nTrade Analysis:")
    print(f"Total Trades: {analysis['total_trades']}")
    print(f"Win Rate: {analysis['win_rate']:.2f}%")
    print(f"Average Profit: {analysis['avg_profit']:.2f}%")
    print(f"Average Loss: {analysis['avg_loss']:.2f}%")
    print(f"Profit Factor: {analysis['profit_factor']:.2f}")
    print(f"Risk-Reward Ratio: {analysis['risk_reward_ratio']:.2f}")
    print(f"Initial Balance: {analysis['initial_balance']}")
    print(f"Final Balance: {analysis['final_balance']}")
    print(f"Total Profit Amount: {analysis['total_profit_amount']}")
    print(f"Total Profit Percentage: {analysis['total_profit_percentage']:.2f}%")
    print(f"Balance History: {analysis['balance_history']}")
    print(f"Trades: {analysis['trades']}")
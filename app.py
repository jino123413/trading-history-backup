<<<<<<< HEAD
from flask import Flask, render_template, jsonify, send_from_directory, request
=======
from flask import Flask, render_template, jsonify, send_from_directory
>>>>>>> f0c952e15b7c3b1bf15071a8efe2851a52f07d01
from trading_logic import fetch_data, preprocess_data, train_model, generate_signals, backtest_with_fibonacci_dca, analyze_trades
import logging
import traceback
import os

# 로깅 설정 개선
<<<<<<< HEAD
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# werkzeug 로거의 레벨을 WARNING으로 설정
logging.getLogger('werkzeug').setLevel(logging.WARNING)

=======
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

>>>>>>> f0c952e15b7c3b1bf15071a8efe2851a52f07d01
app = Flask(__name__)

# 정적 파일 (CSS, JS, 이미지 등) 제공을 위한 라우트
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 메인 페이지
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_trading_data')
def get_trading_data():
    try:
<<<<<<< HEAD
        timeframe = request.args.get('timeframe', '1h')  # URL 파라미터에서 타임프레임 가져오기
        
        # 1. 데이터 가져오기
        logger.info(f"Fetching data for timeframe: {timeframe}...")
        data = fetch_data(timeframe=timeframe)
=======
        # 1. 데이터 가져오기
        logger.info("Fetching data...")
        data = fetch_data(timeframe='1h')
>>>>>>> f0c952e15b7c3b1bf15071a8efe2851a52f07d01
        logger.info(f"Fetched data shape: {data.shape}")
        
        if data.empty:
            logger.error("Fetched data is empty")
            return jsonify({'error': 'No data available'}), 500
        
        # 2. 데이터 전처리
        logger.info("Preprocessing data...")
        features, target = preprocess_data(data)
        
        # 3. 모델 학습 및 신호 생성
        logger.info("Training model and generating signals...")
        model = train_model(features, target)
        signals = generate_signals(model, features)
        
        # 4. 백테스트 실행
        logger.info("Running backtest...")
        current_prices, entry_prices, exit_prices, trades = backtest_with_fibonacci_dca(data, signals)
        
        # 5. 거래 분석
        logger.info("Analyzing trades...")
        analysis = analyze_trades(trades)
        
        # 6. 응답 데이터 구성
        response_data = {
<<<<<<< HEAD
            timeframe: {  # 동적 타임프레임 사용
=======
            '1h': {
>>>>>>> f0c952e15b7c3b1bf15071a8efe2851a52f07d01
                'timestamps': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist() if hasattr(data.index, 'strftime') else data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'prices': current_prices.tolist() if hasattr(current_prices, 'tolist') else list(current_prices),
                'entries': [float(p) if p is not None else None for p in entry_prices],
                'exits': [float(p) if p is not None else None for p in exit_prices],
                'analysis': analysis
            }
        }
        
<<<<<<< HEAD
        # 응답 데이터 구조 검증
        if not all(key in response_data[timeframe] for key in ['timestamps', 'prices', 'entries', 'exits', 'analysis']):
            logger.error("Missing required fields in response data")
            return jsonify({'error': 'Invalid data structure'}), 500
            
=======
>>>>>>> f0c952e15b7c3b1bf15071a8efe2851a52f07d01
        logger.info("Successfully prepared response data")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in get_trading_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# 404 에러 핸들러
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
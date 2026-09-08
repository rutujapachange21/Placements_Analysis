from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained perceptron model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'perceptron.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Placement Predictor AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            color: #f8fafc;
            padding: 20px;
            overflow-x: hidden;
        }

        .background-blob {
            position: absolute;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.35) 0%, rgba(0, 0, 0, 0) 70%);
            border-radius: 50%;
            filter: blur(50px);
            z-index: 0;
            animation: pulse 8s infinite alternate ease-in-out;
        }

        @keyframes pulse {
            0% { transform: scale(0.9) translate(-20%, -20%); }
            100% { transform: scale(1.2) translate(20%, 20%); }
        }

        .card {
            position: relative;
            z-index: 1;
            width: 100%;
            max-width: 440px;
            padding: 36px 32px;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .header h1 {
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #c084fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-top: 6px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
        }

        .input-box {
            width: 100%;
            padding: 13px 16px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(15, 23, 42, 0.6);
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
        }

        .input-box:focus {
            border-color: #a855f7;
            box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25);
            background: rgba(15, 23, 42, 0.85);
        }

        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #9333ea 0%, #4f46e5 100%);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 10px 25px -5px rgba(124, 58, 237, 0.4);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px -5px rgba(124, 58, 237, 0.6);
        }

        .btn:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 24px;
            padding: 16px;
            border-radius: 14px;
            text-align: center;
            font-weight: 600;
            font-size: 0.95rem;
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-placed {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(52, 211, 153, 0.4);
            color: #34d399;
        }

        .result-not-placed {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(248, 113, 113, 0.4);
            color: #f87171;
        }
    </style>
</head>
<body>
    <div class="background-blob"></div>

    <div class="card">
        <div class="header">
            <h1>Chance Predictor</h1>
            <p>Perceptron-based candidate eligibility screening</p>
        </div>

        <form id="predictionForm">
            <div class="form-group">
                <label for="cgpa">CGPA (e.g. 0.0 – 10.0)</label>
                <input type="number" step="0.01" min="0" max="10" id="cgpa" class="input-box" placeholder="Enter your CGPA" required>
            </div>

            <div class="form-group">
                <label for="resume_score">Resume Score (e.g. 0.0 – 10.0)</label>
                <input type="number" step="0.01" min="0" max="10" id="resume_score" class="input-box" placeholder="Enter Resume Rating" required>
            </div>

            <button type="submit" class="btn" id="submitBtn">Analyze Profile</button>
        </form>

        <div id="resultBox" class="result-box"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const resultBox = document.getElementById('resultBox');

            btn.innerText = 'Calculating...';
            btn.style.opacity = '0.7';

            const payload = {
                cgpa: parseFloat(document.getElementById('cgpa').value),
                resume_score: parseFloat(document.getElementById('resume_score').value)
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                resultBox.style.display = 'block';
                if (data.prediction === 1) {
                    resultBox.className = 'result-box result-placed';
                    resultBox.innerHTML = '✨ High Likelihood of Selection!';
                } else {
                    resultBox.className = 'result-box result-not-placed';
                    resultBox.innerHTML = '⚠️ Profile needs improvement for selection.';
                }
            } catch (err) {
                resultBox.style.display = 'block';
                resultBox.className = 'result-box result-not-placed';
                resultBox.innerText = 'Error connecting to prediction server.';
            } finally {
                btn.innerText = 'Analyze Profile';
                btn.style.opacity = '1';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    cgpa = float(data.get('cgpa', 0.0))
    resume_score = float(data.get('resume_score', 0.0))

    # Features passed in exact order: ['cgpa', 'resume_score']
    features = np.array([[cgpa, resume_score]])
    prediction = int(model.predict(features)[0])

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

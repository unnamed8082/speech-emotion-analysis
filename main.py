from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
from datetime import datetime

# 使用非交互式后端
plt.switch_backend('Agg')

app = FastAPI(title="语音情绪分析")

def analyze_audio_simple(file_path: str):
    """简化版音频分析函数"""
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path, sr=22050, duration=10)  # 限制10秒防止超时
        
        # 计算基础特征
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # 计算情绪分数
        rms_var = np.std(rms) / (np.mean(rms) + 1e-8)
        zcr_mean = np.mean(zcr)
        
        # 情绪计算
        emotion_scores = {
            'calm': max(0, 100 - rms_var * 50),
            'tense': min(100, rms_var * 40 + zcr_mean * 20),
            'angry': min(100, rms_var * 60),
            'excited': min(100, (rms_var + zcr_mean) * 30)
        }
        
        # 归一化
        total = sum(emotion_scores.values())
        for key in emotion_scores:
            emotion_scores[key] = round(emotion_scores[key] / total * 100, 1)
        
        # 冲突风险
        conflict_risk = min(100, emotion_scores['tense'] * 0.4 + emotion_scores['angry'] * 0.6)
        
        return {
            'success': True,
            'emotions': emotion_scores,
            'risk': round(conflict_risk, 1),
            'duration': round(duration, 2),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def create_simple_chart(emotions):
    """创建简单的情绪图表"""
    plt.figure(figsize=(8, 5))
    labels = list(emotions.keys())
    values = list(emotions.values())
    colors = ['green', 'yellow', 'red', 'purple']
    
    plt.bar(labels, values, color=colors)
    plt.title('情绪分析结果')
    plt.ylabel('百分比 (%)')
    
    # 保存为base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str

@app.get("/")
async def home():
    """主页面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>语音情绪分析</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 8px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background: #fafafa;
            }
            button {
                background: #4CAF50;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .result {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
            }
            .risk-meter {
                text-align: center;
                margin: 20px 0;
            }
            .meter {
                width: 100%;
                height: 20px;
                background: #e0e0e0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .meter-fill {
                height: 100%;
                background: linear-gradient(90deg, green, yellow, red);
                width: 0%;
                transition: width 0.5s;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 语音情绪分析</h1>
            
            <div class="upload-area">
                <input type="file" id="audioFile" accept="audio/*">
                <p>选择音频文件 (WAV, MP3, M4A等)</p>
            </div>
            
            <button onclick="analyzeAudio()" id="analyzeBtn" disabled>开始分析</button>
            
            <div id="loading" style="display:none; text-align:center;">
                <p>分析中... 请稍候 (约5-10秒)</p>
            </div>
            
            <div class="result" id="result">
                <h3>分析结果</h3>
                <div id="resultContent"></div>
            </div>
        </div>

        <script>
            // 文件选择事件
            document.getElementById('audioFile').addEventListener('change', function(e) {
                document.getElementById('analyzeBtn').disabled = !e.target.files.length;
            });
            
            async function analyzeAudio() {
                const fileInput = document.getElementById('audioFile');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (!fileInput.files.length) return;
                
                analyzeBtn.disabled = true;
                loading.style.display = 'block';
                result.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        displayResults(data);
                    } else {
                        resultContent.innerHTML = '<p style="color:red;">分析失败: ' + data.error + '</p>';
                    }
                } catch (error) {
                    resultContent.innerHTML = '<p style="color:red;">请求失败: ' + error.message + '</p>';
                } finally {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    analyzeBtn.disabled = false;
                }
            }
            
            function displayResults(data) {
                const emotions = data.emotions;
                const risk = data.risk;
                
                let html = `
                    <div class="risk-meter">
                        <h4>冲突风险指数: ${risk}%</h4>
                        <div class="meter">
                            <div class="meter-fill" style="width: ${risk}%"></div>
                        </div>
                        <p>分析时间: ${data.timestamp} | 音频时长: ${data.duration}秒</p>
                    </div>
                    
                    <div style="text-align:center; margin:20px 0;">
                        
                    </div>
                    
                    <div>
                        <h4>情绪分布:</h4>
                        <p>• 平静: ${emotions.calm}% - 语调平稳，情绪稳定</p>
                        <p>• 紧张: ${emotions.tense}% - 语速可能较快</p>
                        <p>• 愤怒: ${emotions.angry}% - 音量变化较大</p>
                        <p>• 兴奋: ${emotions.excited}% - 能量集中</p>
                    </div>
                `;
                
                document.getElementById('resultContent').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """分析音频文件"""
    try:
        # 检查文件类型
        if not file.content_type.startswith('audio/'):
            return {"success": False, "error": "请上传音频文件"}
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # 分析音频
            result = analyze_audio_simple(tmp_file.name)
            
            # 如果分析成功，添加图表
            if result['success']:
                chart_img = create_simple_chart(result['emotions'])
                result['chart'] = chart_img
            
            # 删除临时文件
            os.unlink(tmp_file.name)
            
            return result
            
    except Exception as e:
        return {"success": False, "error": f"处理错误: {str(e)}"}

# Vercel需要这个
app = app

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
import librosa
import io
import base64
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="语音情绪分析系统")

# 配置matplotlib
plt.switch_backend('Agg')

# 存储分析结果的内存缓存
analysis_cache = {}

def analyze_audio(audio_data: bytes, filename: str):
    """分析音频文件并返回情绪分析结果"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # 如果音频不是WAV格式，先转换
            if not filename.lower().endswith('.wav'):
                audio = AudioSegment.from_file(io.BytesIO(audio_data))
                audio.export(tmp_file.name, format="wav")
                y, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
            else:
                tmp_file.write(audio_data)
                tmp_file.flush()
                y, sr = librosa.load(tmp_file.name, sr=22050)
                os.unlink(tmp_file.name)
        
        # 限制音频长度为30秒以避免超时
        max_length = 30 * sr
        if len(y) > max_length:
            y = y[:max_length]
        
        duration = len(y) / sr
        
        # 提取音频特征
        features = extract_audio_features(y, sr)
        
        # 生成情绪分析结果
        emotion_result = generate_emotion_analysis(features, duration)
        
        # 生成可视化图表
        chart_data = generate_charts(y, sr, features, emotion_result)
        
        return {
            "success": True,
            "emotion_result": emotion_result,
            "chart_data": chart_data,
            "duration": duration
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def extract_audio_features(y, sr):
    """提取音频特征"""
    features = {}
    
    # 基础特征
    features['duration'] = len(y) / sr
    features['rms'] = librosa.feature.rms(y=y)[0]  # 音量能量
    features['zcr'] = librosa.feature.zero_crossing_rate(y)[0]  # 过零率
    
    # 音高和频谱特征
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # MFCC特征（语调特征）
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfcc, axis=1)
    features['mfcc_std'] = np.std(mfcc, axis=1)
    
    return features

def generate_emotion_analysis(features, duration):
    """生成情感分析结果"""
    # 基于音频特征计算情绪指标
    rms_var = np.std(features['rms']) / (np.mean(features['rms']) + 1e-8)
    zcr_mean = np.mean(features['zcr'])
    spectral_centroid_mean = np.mean(features['spectral_centroid'])
    
    # 情绪计算（基于音频特征）
    emotion_scores = {
        'calm': max(0, 1 - rms_var * 2),  # 低音量变化
        'tense': min(1, rms_var * 1.5 + zcr_mean * 0.3),  # 高音量变化和过零率
        'angry': min(1, spectral_centroid_mean / 5000 + rms_var),  # 高频和高音量变化
        'excited': min(1, (rms_var + zcr_mean) * 0.8)  # 中等能量和变化
    }
    
    # 归一化
    total = sum(emotion_scores.values())
    for key in emotion_scores:
        emotion_scores[key] = round(emotion_scores[key] / total * 100, 1)
    
    # 冲突风险计算
    conflict_risk = min(100, emotion_scores['tense'] * 0.6 + emotion_scores['angry'] * 0.8 + emotion_scores['excited'] * 0.4)
    
    return {
        'emotion_scores': emotion_scores,
        'conflict_risk': round(conflict_risk, 1),
        'duration': round(duration, 2),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def generate_charts(y, sr, features, emotion_result):
    """生成可视化图表并返回base64编码的图像"""
    charts = {}
    
    try:
        # 1. 波形图
        plt.figure(figsize=(10, 4))
        time = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time, y, alpha=0.7)
        plt.title('音频波形')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['waveform'] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # 2. 情绪分布饼图
        plt.figure(figsize=(6, 6))
        emotions = emotion_result['emotion_scores']
        labels = ['平静', '紧张', '愤怒', '兴奋']
        sizes = [emotions['calm'], emotions['tense'], emotions['angry'], emotions['excited']]
        colors = ['#4CAF50', '#FFC107', '#F44336', '#9C27B0']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('情绪分布')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['emotion_pie'] = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # 3. 频谱图
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('频谱图')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['spectrogram'] = base64.b64encode(buf.read()).decode()
        plt.close()
        
    except Exception as e:
        print(f"图表生成错误: {e}")
    
    return charts

@app.get("/", response_class=HTMLResponse)
async def home():
    """主页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>语音情绪分析系统</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            header {
                text-align: center;
                margin-bottom: 30px;
            }
            h1 { color: #333; margin-bottom: 10px; }
            .upload-section, .results-section {
                margin-bottom: 30px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
            }
            .btn {
                padding: 12px 24px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                display: none;
            }
            .chart {
                margin: 20px 0;
                text-align: center;
            }
            .chart img {
                max-width: 100%;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .risk-meter {
                text-align: center;
                margin: 20px 0;
            }
            .meter {
                width: 200px;
                height: 20px;
                background: #f0f0f0;
                border-radius: 10px;
                margin: 10px auto;
                overflow: hidden;
            }
            .meter-fill {
                height: 100%;
                background: linear-gradient(90deg, green, yellow, red);
                width: 0%;
                transition: width 1s;
                border-radius: 10px;
            }
            .risk-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎤 语音情绪分析系统</h1>
                <p>上传音频文件，分析语音中的情绪状态和冲突风险</p>
            </header>
            
            <div class="upload-section">
                <h2>上传音频文件</h2>
                <div class="upload-area">
                    <input type="file" id="audioFile" accept="audio/*">
                    <p>支持格式: WAV, MP3, M4A, FLAC 等</p>
                    <p>建议时长: 5-30秒</p>
                </div>
                <button class="btn" id="analyzeBtn" disabled>开始分析</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>分析中，请稍候...</p>
            </div>
            
            <div class="results" id="results">
                <h2>分析结果</h2>
                <div id="resultsContent"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('audioFile').addEventListener('change', function(e) {
                const analyzeBtn = document.getElementById('analyzeBtn');
                analyzeBtn.disabled = !e.target.files.length;
            });
            
            document.getElementById('analyzeBtn').addEventListener('click', async function() {
                const fileInput = document.getElementById('audioFile');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                const resultsContent = document.getElementById('resultsContent');
                
                if (!fileInput.files.length) return;
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                loading.style.display = 'block';
                results.style.display = 'none';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        displayResults(data);
                    } else {
                        resultsContent.innerHTML = `<p style="color: red;">分析失败: ${data.error}</p>`;
                    }
                } catch (error) {
                    resultsContent.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
                } finally {
                    loading.style.display = 'none';
                    results.style.display = 'block';
                }
            });
            
            function displayResults(data) {
                const { emotion_result, chart_data } = data;
                const emotions = emotion_result.emotion_scores;
                
                let html = `
                    <div class="risk-meter">
                        <h3>冲突风险指数</h3>
                        <div class="risk-value">${emotion_result.conflict_risk}%</div>
                        <div class="meter">
                            <div class="meter-fill" style="width: ${emotion_result.conflict_risk}%"></div>
                        </div>
                        <p>分析时间: ${emotion_result.timestamp} | 时长: ${emotion_result.duration}秒</p>
                    </div>
                    
                    <div class="chart">
                        <h3>音频波形</h3>
                        
                    </div>
                    
                    <div class="chart">
                        <h3>情绪分布</h3>
                        
                    </div>
                    
                    <div class="chart">
                        <h3>频谱分析</h3>
                        
                    </div>
                    
                    <div class="emotion-details">
                        <h3>情绪分析详情</h3>
                        <p>平静: ${emotions.calm}% - 语调平稳，情绪稳定</p>
                        <p>紧张: ${emotions.tense}% - 语速较快，音调较高</p>
                        <p>愤怒: ${emotions.angry}% - 音量变化大，语调尖锐</p>
                        <p>兴奋: ${emotions.excited}% - 能量集中，节奏活跃</p>
                    </div>
                `;
                
                document.getElementById('resultsContent').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

@app.post("/analyze")
async def analyze_audio_file(file: UploadFile = File(...)):
    """分析上传的音频文件"""
    try:
        # 检查文件类型
        if not file.content_type.startswith('audio/'):
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "请上传音频文件"}
            )
        
        # 读取文件内容
        contents = await file.read()
        
        if len(contents) == 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "文件为空"}
            )
        
        # 分析音频
        result = analyze_audio(contents, file.filename)
        
        if result["success"]:
            return JSONResponse(content=result)
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["error"]}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"处理错误: {str(e)}"}
        )

# Vercel需要这个
handler = app

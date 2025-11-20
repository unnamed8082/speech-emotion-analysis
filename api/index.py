from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="语音情绪分析系统")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_audio_simple(file_path: str):
    """简化版音频分析函数"""
    try:
        # 加载音频文件（限制10秒）
        y, sr = librosa.load(file_path, sr=22050, duration=10)
        
        duration = len(y) / sr
        
        # 提取基础特征
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # 计算特征统计
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

def create_emotion_chart(emotions):
    """创建情绪图表"""
    plt.figure(figsize=(8, 5))
    labels = ['平静', '紧张', '愤怒', '兴奋']
    values = [emotions['calm'], emotions['tense'], emotions['angry'], emotions['excited']]
    colors = ['#4CAF50', '#FFC107', '#F44336', '#9C27B0']
    
    plt.bar(labels, values, color=colors)
    plt.title('情绪分析结果')
    plt.ylabel('百分比 (%)')
    plt.ylim(0, 100)
    
    # 添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 1, f'{v}%', ha='center')
    
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
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 { 
                color: #333; 
                text-align: center;
                margin-bottom: 10px;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px 20px;
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
                width: 100%;
                margin: 10px 0;
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
                border-radius: 10px;
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
                background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
                width: 0%;
                transition: width 1s;
                border-radius: 10px;
            }
            .emotion-bar {
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 语音情绪分析系统</h1>
            <p style="text-align: center; color: #666; margin-bottom: 20px;">
                上传音频文件，分析语音中的情绪状态
            </p>
            
            <div class="upload-area">
                <input type="file" id="audioFile" accept="audio/*" style="margin-bottom: 15px;">
                <p>支持格式: WAV, MP3, M4A等常见音频格式</p>
                <p>建议时长: 5-30秒，文件大小不超过10MB</p>
            </div>
            
            <button onclick="analyzeAudio()" id="analyzeBtn" disabled>开始分析</button>
            
            <div id="loading" style="display:none; text-align:center; padding: 20px;">
                <div style="border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 10px;"></div>
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
                        resultContent.innerHTML = '<p style="color:red; text-align:center;">分析失败: ' + data.error + '</p>';
                    }
                } catch (error) {
                    resultContent.innerHTML = '<p style="color:red; text-align:center;">请求失败: ' + error.message + '</p>';
                } finally {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    analyzeBtn.disabled = false;
                }
            }
            
            function displayResults(data) {
                const emotions = data.emotions;
                const risk = data.risk;
                
                let riskColor = '#4CAF50';
                let riskLevel = '低风险';
                if (risk > 60) {
                    riskColor = '#F44336';
                    riskLevel = '高风险';
                } else if (risk > 30) {
                    riskColor = '#FFC107';
                    riskLevel = '中等风险';
                }
                
                let html = `
                    <div class="risk-meter">
                        <h4>冲突风险指数: <span style="color: ${riskColor}">${risk}%</span></h4>
                        <p>风险级别: ${riskLevel}</p>
                        <div class="meter">
                            <div class="meter-fill" style="width: ${risk}%; background: ${riskColor}"></div>
                        </div>
                        <p>分析时间: ${data.timestamp} | 音频时长: ${data.duration}秒</p>
                    </div>
                    
                    <div style="text-align:center; margin:20px 0;">
                        
                    </div>
                    
                    <div>
                        <h4>情绪分布详情:</h4>
                        <div class="emotion-bar" style="border-color: #4CAF50">
                            <strong>平静:</strong> ${emotions.calm}% - 语调平稳，情绪稳定
                        </div>
                        <div class="emotion-bar" style="border-color: #FFC107">
                            <strong>紧张:</strong> ${emotions.tense}% - 语速可能较快，音调较高
                        </div>
                        <div class="emotion-bar" style="border-color: #F44336">
                            <strong>愤怒:</strong> ${emotions.angry}% - 音量变化较大，语调尖锐
                        </div>
                        <div class="emotion-bar" style="border-color: #9C27B0">
                            <strong>兴奋:</strong> ${emotions.excited}% - 能量集中，节奏活跃
                        </div>
                    </div>
                    
                    <div style="margin-top:20px; padding:15px; background:#e8f5e8; border-radius:5px;">
                        <h4>建议:</h4>
                        <p>${getAdvice(risk, emotions)}</p>
                    </div>
                `;
                
                document.getElementById('resultContent').innerHTML = html;
            }
            
            function getAdvice(risk, emotions) {
                if (risk < 30) {
                    return '对话氛围良好，继续保持当前沟通方式。';
                } else if (risk < 60) {
                    return '建议关注对话中的紧张情绪，适当调整语速和语调。';
                } else {
                    return '检测到较高冲突风险，建议暂停当前话题，先处理情绪再继续沟通。';
                }
            }
            
            // 加载动画样式
            const style = document.createElement('style');
            style.textContent = `
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
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
        
        # 检查文件大小（限制10MB）
        if file.size > 10 * 1024 * 1024:
            return {"success": False, "error": "文件大小不能超过10MB"}
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # 分析音频
            result = analyze_audio_simple(tmp_file.name)
            
            # 如果分析成功，添加图表
            if result['success']:
                chart_img = create_emotion_chart(result['emotions'])
                result['chart'] = chart_img
            
            # 删除临时文件
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            
            return result
            
    except Exception as e:
        return {"success": False, "error": f"处理错误: {str(e)}"}

# Vercel需要这个
handler = app
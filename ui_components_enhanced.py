"""
Enhanced UI Components for Pronunciation Scoring

增强的 UI 组件，用于展示详细的评分和扣分点信息。
"""

import streamlit as st
import html
import base64
from typing import List, Dict, Optional
from pathlib import Path


def render_chinese_character_details(
    char_scores: List[Dict],
    audio_path: str,
    show_modal: bool = True
) -> str:
    """
    渲染带有详细扣分点的中文字符评分。
    
    点击单个字可以：
    1. 播放该字的音频
    2. 显示详细的各维度得分
    3. 显示具体的错误类型和改进建议
    
    Args:
        char_scores: 字符评分列表
        audio_path: 用户音频路径
        show_modal: 是否显示弹窗详情
    
    Returns:
        HTML 字符串
    """
    
    # 读取音频文件
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
    except:
        audio_base64 = ""
    
    # 生成字符按钮 HTML
    char_buttons_html = ""
    
    for idx, char_data in enumerate(char_scores):
        char = char_data.get('char', '?')
        final_score = char_data.get('final_score', 0)
        start_time = char_data.get('start', 0)
        end_time = char_data.get('end', 0)
        
        # 各维度得分
        acoustic_score = char_data.get('acoustic_score', 0) * 100
        tone_score = char_data.get('tone_score', 0) * 100
        duration_score = char_data.get('duration_score', 0) * 100
        pause_score = char_data.get('pause_score', 0) * 100
        
        # 错误信息
        errors = char_data.get('errors', [])
        error_probs = char_data.get('error_probabilities', {})
        
        # 其他信息
        pinyin = char_data.get('pinyin', '')
        predicted_tone = char_data.get('predicted_tone', 0)
        expected_tone = char_data.get('expected_tone', 0)
        is_silence = char_data.get('is_silence', False)
        is_low_energy = char_data.get('is_low_energy', False)
        
        # 确定颜色和表情
        if is_silence:
            color = "#6c757d"
            emoji = "🔇"
            border_style = "dashed"
        elif is_low_energy:
            color = "#fd7e14"
            emoji = "🔉"
            border_style = "dashed"
        elif final_score >= 90:
            color = "#28a745"
            emoji = "✅"
            border_style = "solid"
        elif final_score >= 75:
            color = "#ffc107"
            emoji = "⚠️"
            border_style = "solid"
        else:
            color = "#dc3545"
            emoji = "❌"
            border_style = "solid"
        
        # 构建扣分详情
        deduction_details = []
        
        # 声学扣分
        if acoustic_score < 70:
            deduction_details.append(f"声母韵母: {acoustic_score:.0f}分 (扣{100-acoustic_score:.0f})")
        
        # 声调扣分
        if tone_score < 70:
            tone_info = f"声调: {tone_score:.0f}分"
            if predicted_tone != expected_tone and predicted_tone > 0:
                tone_info += f" (识别为{predicted_tone}声，应为{expected_tone}声)"
            deduction_details.append(tone_info)
        
        # 时长扣分
        if duration_score < 70:
            duration = char_data.get('duration', 0)
            if duration < 0.1:
                deduction_details.append(f"时长过短: {duration*1000:.0f}ms")
            elif duration > 0.6:
                deduction_details.append(f"时长过长: {duration*1000:.0f}ms")
        
        # 停顿扣分
        if pause_score < 70:
            pause_after = char_data.get('pause_after', 0)
            if pause_after > 0.3:
                deduction_details.append(f"停顿过长: {pause_after*1000:.0f}ms")
        
        # 错误类型
        error_list = []
        for err in errors:
            prob = error_probs.get(err, 0) * 100
            error_list.append(f"{err}")
        
        # 转义 HTML
        char_escaped = html.escape(char)
        pinyin_escaped = html.escape(pinyin)
        deduction_html = '<br>'.join(html.escape(d) for d in deduction_details) if deduction_details else '无明显问题'
        errors_html = ', '.join(html.escape(e) for e in error_list) if error_list else '无'
        
        # JSON 数据供 JavaScript 使用
        detail_data = {
            'char': char,
            'pinyin': pinyin,
            'final_score': final_score,
            'acoustic_score': round(acoustic_score, 1),
            'tone_score': round(tone_score, 1),
            'duration_score': round(duration_score, 1),
            'pause_score': round(pause_score, 1),
            'predicted_tone': predicted_tone,
            'expected_tone': expected_tone,
            'errors': errors,
            'deductions': deduction_details,
            'is_silence': is_silence,
            'is_low_energy': is_low_energy,
            'duration': char_data.get('duration', 0),
            'pause_after': char_data.get('pause_after', 0)
        }
        
        import json
        detail_json = html.escape(json.dumps(detail_data, ensure_ascii=False))
        
        char_buttons_html += f'''
        <button onclick="showCharDetail({idx}, {start_time}, {end_time}, '{detail_json}')" 
                class="char-btn"
                style="margin:4px; padding:10px 14px; border-radius:10px; 
                       border:2px {border_style} {color}; background:white; 
                       cursor:pointer; font-size:16px; min-width:60px;
                       transition: all 0.2s ease;">
            <span style="font-size:20px;">{char_escaped}</span><br>
            <small style="color:{color}; font-weight:bold;">{final_score}</small>
        </button>
        '''
    
    # 完整的 HTML（包含详情面板和 JavaScript）
    full_html = f'''
    <style>
        .char-btn:hover {{
            transform: scale(1.1);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .char-btn:active {{
            transform: scale(0.95);
        }}
        .detail-panel {{
            display: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
            color: white;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        .detail-panel.show {{
            display: block;
            animation: slideIn 0.3s ease;
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(-10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .score-bar {{
            height: 8px;
            border-radius: 4px;
            background: rgba(255,255,255,0.3);
            overflow: hidden;
            margin: 4px 0;
        }}
        .score-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        .error-tag {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 4px 10px;
            border-radius: 20px;
            margin: 2px;
            font-size: 12px;
        }}
        .deduction-item {{
            background: rgba(220,53,69,0.3);
            padding: 8px 12px;
            border-radius: 6px;
            margin: 4px 0;
            border-left: 3px solid #dc3545;
        }}
    </style>
    
    <audio id="user-audio" src="data:audio/wav;base64,{audio_base64}" style="display:none;"></audio>
    
    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;">
        {char_buttons_html}
    </div>
    
    <div id="detail-panel" class="detail-panel">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
            <div>
                <span id="detail-char" style="font-size:48px;"></span>
                <span id="detail-pinyin" style="font-size:18px; margin-left:8px; opacity:0.8;"></span>
            </div>
            <div style="text-align:right;">
                <div style="font-size:36px; font-weight:bold;" id="detail-score"></div>
                <div style="font-size:12px; opacity:0.8;">综合得分</div>
            </div>
        </div>
        
        <div style="display:grid; grid-template-columns:repeat(2, 1fr); gap:12px; margin-bottom:16px;">
            <div>
                <div style="display:flex; justify-content:space-between;">
                    <span>🎤 声母韵母</span>
                    <span id="score-acoustic"></span>
                </div>
                <div class="score-bar"><div id="bar-acoustic" class="score-fill" style="background:#4CAF50;"></div></div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between;">
                    <span>🎵 声调</span>
                    <span id="score-tone"></span>
                </div>
                <div class="score-bar"><div id="bar-tone" class="score-fill" style="background:#2196F3;"></div></div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between;">
                    <span>⏱️ 时长</span>
                    <span id="score-duration"></span>
                </div>
                <div class="score-bar"><div id="bar-duration" class="score-fill" style="background:#FF9800;"></div></div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between;">
                    <span>🌊 流畅度</span>
                    <span id="score-pause"></span>
                </div>
                <div class="score-bar"><div id="bar-pause" class="score-fill" style="background:#9C27B0;"></div></div>
            </div>
        </div>
        
        <div id="deduction-section" style="margin-bottom:16px;">
            <div style="font-weight:bold; margin-bottom:8px;">📉 扣分原因</div>
            <div id="deduction-list"></div>
        </div>
        
        <div id="error-section">
            <div style="font-weight:bold; margin-bottom:8px;">⚠️ 错误类型</div>
            <div id="error-list"></div>
        </div>
        
        <div style="margin-top:16px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.2);">
            <button onclick="replayChar()" 
                    style="background:white; color:#667eea; border:none; padding:10px 20px; 
                           border-radius:20px; cursor:pointer; font-weight:bold;">
                🔊 重新播放
            </button>
            <button onclick="hideDetail()" 
                    style="background:transparent; color:white; border:1px solid white; 
                           padding:10px 20px; border-radius:20px; cursor:pointer; margin-left:8px;">
                关闭
            </button>
        </div>
    </div>
    
    <script>
        let currentStart = 0;
        let currentEnd = 0;
        let animationId = null;
        
        function showCharDetail(idx, start, end, detailJson) {{
            currentStart = start;
            currentEnd = end;
            
            // 播放音频
            playAudioSegment(start, end);
            
            // 解析详情数据
            const detail = JSON.parse(detailJson);
            
            // 更新面板内容
            document.getElementById('detail-char').textContent = detail.char;
            document.getElementById('detail-pinyin').textContent = detail.pinyin;
            document.getElementById('detail-score').textContent = detail.final_score;
            
            // 更新各维度得分
            document.getElementById('score-acoustic').textContent = detail.acoustic_score;
            document.getElementById('score-tone').textContent = detail.tone_score;
            document.getElementById('score-duration').textContent = detail.duration_score;
            document.getElementById('score-pause').textContent = detail.pause_score;
            
            document.getElementById('bar-acoustic').style.width = detail.acoustic_score + '%';
            document.getElementById('bar-tone').style.width = detail.tone_score + '%';
            document.getElementById('bar-duration').style.width = detail.duration_score + '%';
            document.getElementById('bar-pause').style.width = detail.pause_score + '%';
            
            // 更新扣分原因
            const deductionList = document.getElementById('deduction-list');
            if (detail.deductions && detail.deductions.length > 0) {{
                deductionList.innerHTML = detail.deductions.map(d => 
                    `<div class="deduction-item">${{d}}</div>`
                ).join('');
                document.getElementById('deduction-section').style.display = 'block';
            }} else {{
                deductionList.innerHTML = '<div style="opacity:0.7;">👍 发音良好，无明显扣分</div>';
            }}
            
            // 更新错误类型
            const errorList = document.getElementById('error-list');
            if (detail.errors && detail.errors.length > 0) {{
                errorList.innerHTML = detail.errors.map(e => 
                    `<span class="error-tag">${{e}}</span>`
                ).join('');
                document.getElementById('error-section').style.display = 'block';
            }} else {{
                errorList.innerHTML = '<span style="opacity:0.7;">无错误标记</span>';
            }}
            
            // 显示面板
            document.getElementById('detail-panel').classList.add('show');
        }}
        
        function hideDetail() {{
            document.getElementById('detail-panel').classList.remove('show');
        }}
        
        function playAudioSegment(start, end) {{
            const audio = document.getElementById('user-audio');
            audio.pause();
            
            if (animationId) {{
                cancelAnimationFrame(animationId);
            }}
            
            audio.currentTime = start;
            audio.play();
            
            function checkTime() {{
                if (audio.currentTime >= end - 0.01) {{
                    audio.pause();
                }} else if (!audio.paused) {{
                    animationId = requestAnimationFrame(checkTime);
                }}
            }}
            
            animationId = requestAnimationFrame(checkTime);
        }}
        
        function replayChar() {{
            playAudioSegment(currentStart, currentEnd);
        }}
    </script>
    '''
    
    return full_html


def get_improvement_suggestions(errors: List[str]) -> List[str]:
    """
    根据错误类型生成改进建议。
    
    Args:
        errors: 错误类型列表
    
    Returns:
        改进建议列表
    """
    suggestions_map = {
        '声调错误': [
            '🎵 声调练习建议：',
            '- 一声：保持高平调，想象在山顶说话',
            '- 二声：从中音升到高音，像问问题的语气',
            '- 三声：先降后升，像感叹"哦~原来如此"',
            '- 四声：从高音快速降到低音，像生气地说"不！"'
        ],
        '发音模糊': [
            '🎤 清晰度练习：',
            '- 放慢语速，确保每个音节发完整',
            '- 嘴型要到位，尤其是圆唇音（u, o）',
            '- 可以对着镜子练习，观察嘴型变化'
        ],
        '声母轻': [
            '💪 声母加强练习：',
            '- 爆破音（b, p, d, t）需要有力的气流',
            '- 可以在手背前发音，感受气流强度',
            '- 塞擦音（zh, ch, z, c）注意舌位'
        ],
        '韵母不圆': [
            '👄 韵母圆唇练习：',
            '- u 音：嘴唇前突，呈圆形',
            '- o 音：嘴巴张圆，舌头后缩',
            '- ü 音：先发 i 音，保持舌位，嘴唇圆起'
        ],
        '发音过短': [
            '⏳ 时长控制：',
            '- 每个字要发完整，不要急促',
            '- 特别是三声，需要足够时间完成升降'
        ],
        '发音过长': [
            '⚡ 避免拖音：',
            '- 保持自然的说话节奏',
            '- 避免刻意拉长音节'
        ],
        '停顿过多': [
            '🌊 流畅度提升：',
            '- 字与字之间要连贯',
            '- 可以先慢速连读，再逐渐加快',
            '- 多听标准发音，模仿节奏'
        ]
    }
    
    suggestions = []
    for error in errors:
        if error in suggestions_map:
            suggestions.extend(suggestions_map[error])
    
    return suggestions


def render_scoring_summary(
    overall_metrics: Dict,
    char_scores: List[Dict]
) -> None:
    """
    渲染评分总结。
    
    Args:
        overall_metrics: 整体评分指标
        char_scores: 各字符评分
    """
    st.markdown("### 📊 评分详情")
    
    # 总分
    overall_score = overall_metrics.get('overall_score', 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 综合得分",
            f"{overall_score}/100",
            delta=None
        )
    
    with col2:
        acoustic = overall_metrics.get('avg_acoustic_score', 0)
        st.metric("🎤 声母韵母", f"{acoustic}")
    
    with col3:
        tone = overall_metrics.get('avg_tone_score', 0)
        st.metric("🎵 声调", f"{tone}")
    
    with col4:
        fluency = overall_metrics.get('avg_pause_score', 0)
        st.metric("🌊 流畅度", f"{fluency}")
    
    # 统计问题
    all_errors = []
    for char_data in char_scores:
        errors = char_data.get('errors', [])
        all_errors.extend(errors)
    
    if all_errors:
        st.markdown("### ⚠️ 发现的问题")
        
        # 统计错误频率
        from collections import Counter
        error_counts = Counter(all_errors)
        
        for error, count in error_counts.most_common():
            if count > 1:
                st.warning(f"**{error}**: 出现 {count} 次")
            else:
                st.info(f"**{error}**")
        
        # 改进建议
        st.markdown("### 💡 改进建议")
        unique_errors = list(error_counts.keys())
        suggestions = get_improvement_suggestions(unique_errors)
        
        for suggestion in suggestions:
            st.markdown(suggestion)
    else:
        st.success("🎉 太棒了！没有发现明显的发音问题！")


def create_comparison_view(
    char_scores: List[Dict],
    show_reference: bool = True
) -> str:
    """
    创建用户发音与标准发音的对比视图。
    
    Args:
        char_scores: 字符评分数据
        show_reference: 是否显示参考信息
    
    Returns:
        HTML 字符串
    """
    html_content = '''
    <style>
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }
        .comparison-table th, .comparison-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #eee;
        }
        .comparison-table th {
            background: #f8f9fa;
            font-weight: bold;
        }
        .score-cell {
            font-weight: bold;
        }
        .score-high { color: #28a745; }
        .score-mid { color: #ffc107; }
        .score-low { color: #dc3545; }
    </style>
    
    <table class="comparison-table">
        <thead>
            <tr>
                <th>字符</th>
                <th>拼音</th>
                <th>声学</th>
                <th>声调</th>
                <th>时长</th>
                <th>流畅</th>
                <th>总分</th>
            </tr>
        </thead>
        <tbody>
    '''
    
    for char_data in char_scores:
        char = char_data.get('char', '?')
        pinyin = char_data.get('pinyin', '')
        acoustic = char_data.get('acoustic_score', 0) * 100
        tone = char_data.get('tone_score', 0) * 100
        duration = char_data.get('duration_score', 0) * 100
        pause = char_data.get('pause_score', 0) * 100
        final = char_data.get('final_score', 0)
        
        def get_score_class(score):
            if score >= 80:
                return 'score-high'
            elif score >= 60:
                return 'score-mid'
            else:
                return 'score-low'
        
        html_content += f'''
        <tr>
            <td style="font-size:24px;">{html.escape(char)}</td>
            <td>{html.escape(pinyin)}</td>
            <td class="score-cell {get_score_class(acoustic)}">{acoustic:.0f}</td>
            <td class="score-cell {get_score_class(tone)}">{tone:.0f}</td>
            <td class="score-cell {get_score_class(duration)}">{duration:.0f}</td>
            <td class="score-cell {get_score_class(pause)}">{pause:.0f}</td>
            <td class="score-cell {get_score_class(final)}">{final}</td>
        </tr>
        '''
    
    html_content += '''
        </tbody>
    </table>
    '''
    
    return html_content

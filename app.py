# flask-api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Mengizinkan akses dari Laravel

# ============================================
# KONFIGURASI PATH MODEL
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ============================================
# LOAD MODEL DAN ENCODERS
# ============================================
print("=" * 50)
print("🚀 Memulai Flask API Server...")
print("=" * 50)

try:
    # Load model Decision Tree
    model = joblib.load(os.path.join(MODELS_DIR, 'model_harian.pkl'))
    print("✅ Model Decision Tree berhasil diload")
    
    # Load encoder untuk mood (input)
    encoder_mood = joblib.load(os.path.join(MODELS_DIR, 'encoder_mood.pkl'))
    print(f"✅ Encoder mood berhasil diload (classes: {list(encoder_mood.classes_)})")
    
    # Load encoder untuk label (output)
    encoder_label = joblib.load(os.path.join(MODELS_DIR, 'encoder_label.pkl'))
    print(f"✅ Encoder label berhasil diload (classes: {list(encoder_label.classes_)})")
    
    model_loaded = True
    
except FileNotFoundError as e:
    print(f"❌ ERROR: File model tidak ditemukan - {e}")
    print("Pastikan folder 'models' berisi:")
    print("  - model_harian.pkl")
    print("  - encoder_mood.pkl")
    print("  - encoder_label.pkl")
    model_loaded = False
    model = None
    encoder_mood = None
    encoder_label = None

print("=" * 50)

# ============================================
# ENDPOINT HEALTH CHECK
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Cek status server dan model"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'mood_classes': list(encoder_mood.classes_) if encoder_mood else [],
        'label_classes': list(encoder_label.classes_) if encoder_label else []
    })

# ============================================
# ENDPOINT INFO MODEL
# ============================================
@app.route('/info', methods=['GET'])
def info():
    """Informasi detail tentang model"""
    if not model_loaded:
        return jsonify({'error': 'Model tidak tersedia'}), 503
    
    return jsonify({
        'model_type': type(model).__name__,
        'model_parameters': model.get_params() if hasattr(model, 'get_params') else {},
        'features': ['mood (encoded)', 'durasi_belajar (menit)', 'durasi_tidur (jam)'],
        'mood_classes': list(encoder_mood.classes_),
        'label_classes': list(encoder_label.classes_),
        'description': 'Decision Tree Classifier untuk rekomendasi durasi belajar'
    })

# ============================================
# ENDPOINT PREDIKSI HARIAN (HARI 1-7)
# ============================================
@app.route('/predict', methods=['POST'])
def predict_daily():
    """
    Prediksi berdasarkan input HARIAN
    Digunakan untuk 7 hari pertama
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model tidak tersedia'
        }), 503
    
    try:
        data = request.json
        
        # ========================================
        # VALIDASI INPUT
        # ========================================
        required_fields = ['mood', 'durasi_belajar', 'durasi_tidur']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Field "{field}" diperlukan'
                }), 400
        
        mood = data['mood']
        durasi_belajar = int(data['durasi_belajar'])
        durasi_tidur = int(data['durasi_tidur'])
        
        # ========================================
        # ENCODE MOOD
        # ========================================
        try:
            mood_encoded = encoder_mood.transform([mood])[0]
        except ValueError as e:
            return jsonify({
                'success': False,
                'error': f'Mood "{mood}" tidak valid. Mood yang tersedia: {list(encoder_mood.classes_)}'
            }), 400
        
        # ========================================
        # PREDIKSI DENGAN DECISION TREE
        # ========================================
        features = pd.DataFrame([[
        mood_encoded,
        durasi_belajar,
        durasi_tidur
        ]], columns=['mood', 'durasi_belajar', 'durasi_tidur'])
        
        prediction = model.predict(features)
        label = encoder_label.inverse_transform(prediction)[0]
        
        # Hitung confidence (probabilitas)
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))
        
        # Probabilitas per kelas
        proba_dict = {}
        for i, prob in enumerate(proba):
            class_name = encoder_label.inverse_transform([i])[0]
            proba_dict[class_name] = float(prob)
        
        # ========================================
        # RESPONSE
        # ========================================
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': confidence,
            'probabilities': proba_dict,
            'input_received': {
                'mood': mood,
                'durasi_belajar': durasi_belajar,
                'durasi_tidur': durasi_tidur
            },
            'model_used': 'Decision Tree (daily)'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ENDPOINT PREDIKSI POLA (HARI KE-8 dst)
# ============================================
@app.route('/predict/pattern', methods=['POST'])
def predict_pattern():
    """
    Prediksi berdasarkan POLA 7 HARI TERAKHIR
    Tetap menggunakan Decision Tree!
    Digunakan untuk hari ke-8 dan seterusnya
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model tidak tersedia'
        }), 503
    
    try:
        data = request.json
        
        # ========================================
        # VALIDASI INPUT
        # ========================================
        required_fields = ['avg_duration', 'most_frequent_mood', 'avg_sleep']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Field "{field}" diperlukan'
                }), 400
        
        avg_duration = float(data['avg_duration'])
        most_frequent_mood = data['most_frequent_mood']
        avg_sleep = float(data['avg_sleep'])
        trend = data.get('trend', 'stabil')
        consistency = data.get('consistency', 0)
        
        # ========================================
        # ENCODE MOOD
        # ========================================
        try:
            mood_encoded = encoder_mood.transform([most_frequent_mood])[0]
        except ValueError as e:
            # Fallback: coba cari mood terdekat
            available_moods = list(encoder_mood.classes_)
            print(f"⚠️ Mood '{most_frequent_mood}' tidak dikenal. Menggunakan 'Biasa Saja' sebagai fallback.")
            mood_encoded = encoder_mood.transform(['Biasa Saja'])[0]
        
        # ========================================
        # PREDIKSI DENGAN DECISION TREE
        # Fitur yang digunakan: [mood_encoded, avg_duration, avg_sleep]
        # ========================================
        features = np.array([[
            mood_encoded,
            int(avg_duration),
            int(avg_sleep)
        ]])
        
        prediction = model.predict(features)
        label = encoder_label.inverse_transform(prediction)[0]
        
        # Hitung confidence (probabilitas)
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))
        
        # ========================================
        # ANALISIS TAMBAHAN (untuk frontend)
        # ========================================
        
        # Rekomendasi berdasarkan trend
        trend_recommendation = ""
        if trend == 'meningkat':
            trend_recommendation = "Bagus! Durasi belajarmu terus meningkat. Pertahankan momentum ini!"
        elif trend == 'menurun':
            trend_recommendation = "Durasi belajarmu menurun. Coba buat jadwal belajar yang lebih konsisten."
        else:
            trend_recommendation = "Konsistensimu stabil. Tingkatkan sedikit demi sedikit untuk hasil lebih baik."
        
        # Rekomendasi berdasarkan konsistensi
        consistency_recommendation = ""
        if consistency >= 6:
            consistency_recommendation = "Luar biasa! Kamu sangat konsisten. Terus pertahankan!"
        elif consistency >= 4:
            consistency_recommendation = "Cukup baik, tapi masih ada hari yang terlewat. Ayo lebih konsisten lagi!"
        else:
            consistency_recommendation = "Masih banyak hari yang terlewat. Yuk, mulai rutin mencatat aktivitas!"
        
        # Rekomendasi berdasarkan mood
        mood_recommendation = ""
        if most_frequent_mood in ['Bagus', 'Lumayan']:
            mood_recommendation = "Mood positifmu mendukung belajar efektif!"
        else:
            mood_recommendation = "Coba cari aktivitas yang menyenangkan sebelum belajar agar mood lebih baik."
        
        # Rekomendasi berdasarkan tidur
        sleep_recommendation = ""
        if avg_sleep >= 7:
            sleep_recommendation = "Tidur cukup! Ini bagus untuk konsentrasi."
        elif avg_sleep >= 5:
            sleep_recommendation = "Tidur kurang ideal. Coba tidur lebih awal."
        else:
            sleep_recommendation = "Kurang tidur akan mempengaruhi konsentrasi belajar. Prioritaskan istirahat!"
        
        # ========================================
        # RESPONSE
        # ========================================
        return jsonify({
            'success': True,
            'prediction': label,
            'confidence': confidence,
            'pattern_analysis': {
                'avg_duration': round(avg_duration, 1),
                'most_frequent_mood': most_frequent_mood,
                'avg_sleep': round(avg_sleep, 1),
                'trend': trend,
                'consistency': f"{consistency}/7 hari",
                'trend_recommendation': trend_recommendation,
                'consistency_recommendation': consistency_recommendation,
                'mood_recommendation': mood_recommendation,
                'sleep_recommendation': sleep_recommendation
            },
            'input_received': {
                'avg_duration': avg_duration,
                'most_frequent_mood': most_frequent_mood,
                'avg_sleep': avg_sleep,
                'trend': trend,
                'consistency': consistency
            },
            'model_used': 'Decision Tree (pattern-based)',
            'note': 'Prediksi berdasarkan pola 7 hari terakhir menggunakan Decision Tree'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ENDPOINT BATCH PREDICTION (Opsional)
# ============================================
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prediksi untuk multiple data (batch)
    """
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model tidak tersedia'
        }), 503
    
    try:
        data = request.json
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({
                'success': False,
                'error': 'Field "samples" diperlukan'
            }), 400
        
        results = []
        for sample in samples:
            try:
                mood_encoded = encoder_mood.transform([sample['mood']])[0]
                features = np.array([[
                    mood_encoded,
                    int(sample['durasi_belajar']),
                    int(sample['durasi_tidur'])
                ]])
                
                prediction = model.predict(features)
                label = encoder_label.inverse_transform(prediction)[0]
                
                results.append({
                    'input': sample,
                    'prediction': label
                })
            except Exception as e:
                results.append({
                    'input': sample,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# RUN SERVER
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Menjalankan Flask API Server...")
    print("=" * 50)
    print(f"📍 Endpoint yang tersedia:")
    print(f"   GET  /health  - Cek status server")
    print(f"   GET  /info    - Informasi model")
    print(f"   POST /predict - Prediksi harian (hari 1-7)")
    print(f"   POST /predict/pattern - Prediksi pola (hari 8+)")
    print(f"   POST /predict/batch - Prediksi batch")
    print("=" * 50)
    print("\n🔥 Server berjalan di http://127.0.0.1:5000")
    print("   Tekan Ctrl+C untuk menghentikan server")
    print("=" * 50)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

'''
	Contoh Deloyment untuk Domain Data Science (DS)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path='/static')
model = pickle.load(open('model_gini.pkl', 'rb'))

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]


@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]


@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Nilai default untuk variabel input atau features (X) ke model
    input_harapan_lama_sekolah = 14.50
    input_pengeluaran_perkapita = 8000
    input_rerata_lama_sekolah = 8.50
    input_usia_harapan_hidup = 65.0

    if request.method == 'POST':
        # Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
        input_harapan_lama_sekolah = float(
            request.form['harapan_lama_sekolah'])
        input_pengeluaran_perkapita = float(
            request.form['pengeluaran_perkapita'])
        input_rerata_lama_sekolah = float(request.form['rerata_lama_sekolah'])
        input_usia_harapan_hidup = float(request.form['usia_harapan_hidup'])

        # Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
        df_test = pd.DataFrame(data={
            "Harapan_Lama_Sekolah": [input_harapan_lama_sekolah],
            "Pengeluaran_Perkapita": [input_pengeluaran_perkapita],
            "Rerata_Lama_Sekolah": [input_rerata_lama_sekolah],
            "Usia_Harapan_Hidup": [input_usia_harapan_hidup]
        })

        hasil_prediksi = model.predict(df_test[0:1])[0]

        # Set Path untuk gambar hasil prediksi
        if hasil_prediksi == 'Very-High':
            gambar_prediksi = '/static/images/ipm_veryhigh.jpg'
        elif hasil_prediksi == 'High':
            gambar_prediksi = '/static/images/ipm_high.jpg'
        elif hasil_prediksi == 'Normal':
            gambar_prediksi = '/static/images/ipm_normal.jpg'
        else:
            gambar_prediksi = '/static/images/ipm_low.jpg'

        # Return hasil prediksi dengan format JSON
        # return jsonify({
        #     "prediksi": hasil_prediksi,
        #     "gambar_prediksi": "http://localhost:5000"+gambar_prediksi
        # })
        return render_template("index.html",gambar_prediksi=gambar_prediksi,prediksi=hasil_prediksi)

# =[Main]========================================


if __name__ == '__main__':

    # Load model yang telah ditraining
    # model = load('modeldtreeipm.moodelGini')

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)

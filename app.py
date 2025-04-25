from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from model.predictive_model import run_forecast
from model.pso_clustering import run_pso_clustering  # <-- NEW

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Run LSTM Forecasting
            metrics, fig_path = run_forecast(filepath)

            # Run PSO-Enhanced Clustering
            best_k, cluster_score = run_pso_clustering(filepath)

            return render_template(
                'index.html',
                metrics=metrics,
                fig_path=fig_path,
                best_k=best_k,
                cluster_score=cluster_score
            )

    return render_template('index.html', metrics=None)

if __name__ == '__main__':
    app.run(debug=True)

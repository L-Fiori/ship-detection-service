# app/routes.py
from flask import render_template, request, redirect, url_for, flash
from app import app, db
from datetime import datetime

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    # Logica para login
    return render_template('login.html')

@app.route('/collect_inputs', methods=['GET', 'POST'])
def collect_inputs():
    """
    Contem a logica para coleta de inputs do usuario
    que tenham a ver com a regiao desejada das imagens
    de satelite coletadas, periodo de tempo, taxa de 
    cobertura de nuvens e outros parametros que podem
    aparecer.
    Tais entradas sao utilizadas na API para obtencao das
    imagens de satelite. Nesse bloco as imagens sao obtidas
    e salvas no banco de dados. 
    """
    if request.method == 'POST':
        region = request.form.get('region')
        date_range = request.form.get('date_range')
        cloud_percentage = request.form.get('cloud_percentage')

        # A seguir utilizar as entradas para chamar a API

        # Apos, salvamos as imagens no banco de dados de forma
        # estruturada e enviamos para a pagina que vai realizar
        # a inferencia nas imagens e dispo-las na tela.

        # Por enquanto, simulamos a obtencao das imagens
        # como se fossem urls em uma lista.
        images = [
            'url1',
            'url2',
            'url3'
        ]

        return render_template('result.html', images=images)

    return render_template('collect_inputs.html')

@app.route('/process_results', methods=['POST'])
def process_results():
    if request.method == 'POST':
        image_urls = request.form.getlist('image_url')

        # Aqui ocorre o processamento das imagens coletadas
        # pelo modelo treinado e posterior extracao das
        # caracteristicas convenientes

        # A titulo de estrutura, fingimos que os dados foram
        # processados e estao em uma lista de dicionarios
        processed_data = [
            {'url': 'url1', 'characteristics': 'characteristics1'},
            {'url': 'url2', 'characteristics': 'characteristics2'},
            {'url': 'url3', 'characteristics': 'characteristics3'},
        ]

        # Ocorre o armazenamento desses dados no banco de dados
        for data in processed_data:
            pass

        #db.session.commit()

        flash('Results processed and stored successfully!', 'success')
        return render_template('result.html', results=processed_data)

    # Redirect pra home caso acesse a url sem emitir o form
    return redirect(url_for('index'))


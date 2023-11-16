from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
from time import sleep
from app.utils import get_products

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

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
        location = request.json.get('loc')
        start_date = request.json.get('startDate')
        end_date = request.json.get('endDate')
        cloud = request.json.get('cloud')

        print(location, start_date, end_date, cloud)

        products = get_products(location, start_date, end_date, cloud)
        
        return jsonify(request.json)

    return jsonify({'message': 'Invalid request'}), 400

@app.route('/processing', methods=['GET'])
def processing():
    # Retrieve user_input from URL parameter
    user_input = request.args.get('user_input')

    # A seguir utilizar as entradas para chamar a API
    
    # Apos, salvamos as imagens no banco de dados de forma
    # estruturada e enviamos para a pagina que vai realizar
    # a inferencia nas imagens e dispo-las na tela.

    # Por enquanto, simulamos a obtencao das imagens
    # como se fossem urls em uma lista.

    print(user_input)

    # Simulando o tempo de processamento
    sleep(8)

    return redirect(url_for('process_results'))

@app.route('/result_page', methods=['POST'])
def process_results():
    if request.method == 'POST':
        # image_urls = request.form.getlist('image_url')

        # Aqui ocorre o processamento das imagens coletadas
        # pelo modelo treinado e posterior extracao das
        # caracteristicas convenientes

        # A titulo de estrutura, fingimos que os dados foram
        # processados e estao em uma lista de dicionarios
        #processed_data = [
        #    {'url': 'url1', 'characteristics': 'characteristics1'},
        #    {'url': 'url2', 'characteristics': 'characteristics2'},
        #    {'url': 'url3', 'characteristics': 'characteristics3'},
        #]
        processed_data = []

        # Ocorre o armazenamento desses dados no banco de dados
        #for data in processed_data:
        #    pass

        #db.session.commit()

        #flash('Results processed and stored successfully!', 'success')
        return render_template('index.html')

    # Redirect pra home caso acesse a url sem emitir o form
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

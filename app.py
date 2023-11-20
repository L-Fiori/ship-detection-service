from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
from datetime import datetime
from flask_cors import CORS
from time import sleep
from io import BytesIO
from PIL import Image
import numpy as np
import base64
from app.utils import get_products, download_images, run
from app.models import db, images, ships

app = Flask(__name__, template_folder='frontend/public')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.sqlite3'
app.config['SQLAMCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
migrate = Migrate(app, db)
db.init_app(app)
bcrypt = Bcrypt(app)

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
        ## --- Add to 'image' table in db ---
        # img_name = "000155de5.jpg"
        # pil_img = Image.open("C:/Users/ticto/Downloads/"+img_name)

        # buff = BytesIO()
        # pil_img.save(buff, format="JPEG")
        # pil_img_64 = base64.b64encode(buff.getvalue())

        # datetime_str = '09/19/22 13:55:26'
        # datetime_object = datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')

        # name = img_name
        # image = pil_img_64
        # ship_count = 2
        # date = datetime_object
        # product = 'CBERS4A_WPM20214320220618'
        # if images.query.filter_by(name = img_name).first() is None:
        #     imgs = images(name, image, ship_count, product, date)
        #     db.session.add(imgs)
        #     db.session.commit()

        ## --- Add to 'ships' table in db ---
        # number = 0
        # classification = 'tank'
        # size = 3
        # image_id = 2

        # ship = ships(number, classification, size, image_id)
        # db.session.add(ship)
        # db.session.commit()

        ## --- Delete from db ---
        #image = images.query.filter_by(_id = 1).delete()
        #db.session.commit()

        ## --- Normal run ---
        location = request.json.get('loc')
        start_date = request.json.get('startDate')
        end_date = request.json.get('endDate')
        cloud = request.json.get('cloud')

        print(location, start_date, end_date, cloud)

        products = get_products(location, start_date, end_date, cloud)
        #download_images(products)
        run(products, location)

        return jsonify({})


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
    sleep(2)

    return redirect(url_for('result_page'))

@app.route('/result_page', methods=['GET', 'POST'])
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
    
    if request.method == 'GET':
        images_table = images.query.all()
        ships_table = ships.query.all()
        products_array = []
        images_array = []
        names_array = []
        ships_array = []
        properties_array = []
        for image_entry in images_table:
            if not image_entry.product in products_array:
                products_array.append(image_entry.product)
                names_array.append([image_entry.name])
                images_array.append([image_entry.image.decode("utf-8")])
                ships_array.append([image_entry.ship_count])
                properties_array.append([[]])
                for ship in ships.query.filter_by(image_id = image_entry._id):
                    properties_array[products_array.index(image_entry.product)][0].append([ship.number, ship.classification, ship.size])
            else:
                id_product = products_array.index(image_entry.product)
                names_array[id_product].append(image_entry.name)
                images_array[id_product].append(image_entry.image.decode("utf-8"))
                ships_array[id_product].append(image_entry.ship_count)
                properties_array[id_product].append([])
                for ship in ships.query.filter_by(image_id = image_entry._id):
                    properties_array[id_product][names_array[id_product].index(image_entry.name)].append([ship.number, ship.classification, ship.size])

        response = {'images': images_array,
                    'names': names_array,
                    'products': products_array,
                    'ships': ships_array,
                    'properties': properties_array}

        return jsonify(response)

    # Redirect pra home caso acesse a url sem emitir o form
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.app_context().push()
    db.create_all()
    app.run(debug=True, port=8080)

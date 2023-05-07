from flask import Flask, request
from flask_cors import CORS, cross_origin
import os
import torch
import torchvision
import PIL.Image as Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename
from bson.binary import Binary
import io
import zlib
import pymongo



app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

#app.config['MONGO_URI'] = 'mongodb://database/pythonmongodb'

#mongo = PyMongo(app)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mi_base_de_datos"]
collection_entrenar = db["mis_pajaros_entrenar"]
collection_subidas = db["mis_pajaros_subidas"]
collection_clases = db["mis_clases_de_pajaros"]
collection_test = db["mis_pajaros_test"]
#mongo = PyMongo(app)

#client = MongoClient('mongodb://localhost:27017/') # dirección de conexión a la base de datos
#db = client['nombre-de-tu-base-de-datos'] # nombre de la base de datos
@app.route('/populate', methods=['POST'])
def create_user():
     # Receiving Data
     print(request.json)
     ruta_padre = "./pajaros2/birds/birds"
     for tipo_ave in os.listdir(ruta_padre):
         ruta_tipo_ave = os.path.join(ruta_padre, tipo_ave)
         for imagen_nombre in os.listdir(ruta_tipo_ave):
             ruta_imagen = os.path.join(ruta_tipo_ave, imagen_nombre)

             # Preparar los datos y las imágenes
             datos_imagen = {"tipo": tipo_ave, "test": False}
             imagen = Image.open(ruta_imagen)

             # Convertir la imagen a bytes para almacenarla en MongoDB
             imagen_bytes = Binary(imagen.tobytes())

             # Insertar los datos y la imagen en la colección
             collection_entrenar.insert_one({"datos": datos_imagen, "imagen": imagen_bytes})
     return {'message': 'recieved'}

@app.route('/populatetest', methods=['POST'])
def create_user2():
     # Receiving Data
     print(request.json)
     ruta_padre = "./pajaros2/birds/birdsTest"
     for tipo_ave in os.listdir(ruta_padre):
         ruta_tipo_ave = os.path.join(ruta_padre, tipo_ave)
         for imagen_nombre in os.listdir(ruta_tipo_ave):
             ruta_imagen = os.path.join(ruta_tipo_ave, imagen_nombre)

             # Preparar los datos y las imágenes
             datos_imagen = {"tipo": tipo_ave, "test": True}
             imagen = Image.open(ruta_imagen)

             # Convertir la imagen a bytes para almacenarla en MongoDB
             imagen_bytes = Binary(imagen.tobytes())

             # Insertar los datos y la imagen en la colección
             collection_test.insert_one({"datos": datos_imagen, "imagen": imagen_bytes})
     return {'message': 'recieved'}

# aquí se carga el modelo solo hay que darle el path
model_path = './best_model.pth'
model = torch.load(model_path)
# se cargan las clases con el número de clases para utilizarlo más tarde
# se puede cambiar para coger la carpeta birds

#clases_path = './pajaros2/birds/birds'
#classes = os.listdir(clases_path)

@app.route('/postclases', methods=['POST'])
def post_clases():
     clases_path = './pajaros2/birds/birds'
     classes = os.listdir(clases_path)
     for tipo_ave in classes:
         collection_clases.insert_one({"clase": tipo_ave})
     return {'message': 'recieved'}

documentos = collection_clases.find()
clases = []
# iterar sobre los documentos e imprimirlos
for documento in documentos:
    clases.append(documento['clase'])

# se crea un transformador para las imágenes de test
mean = [0.4704, 0.4669, 0.3898]
std = [0.2037, 0.2002, 0.2051]

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# se crea una pequeña función para clasificar todas las imágenes de los tests


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return classes[predicted.item()]


# se crea una lista de las imágenes de los tests que utilizamos
# cambiar el path si es necesario utilizar otros tests
#tests_path = './pajaros2/submission_test/submission_test2'
#a = os.listdir(tests_path)
#a.sort(key=len)

# se crea una lista para almacenar los resultados de la clasificación
resultados = []


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Script para archivo
    file = request.files['image']
    # La ruta donde se encuentra el archivo actual
    basepath = os.path.dirname(__file__)
    # Nombre original del archivo
    filename = secure_filename(file.filename)
    upload_path = os.path.join(basepath, 'static/archivos',filename)
    file.save(upload_path)

    # se guarda la imagen en el directorio 'uploads'

    # se clasifica la imagen con la función 'classify' y se almacena el resultado
    resultado = classify(model, image_transforms, os.path.join('static/archivos', filename), clases)
    datos_imagen = {"tipo": resultado, "test": False}
    imagen = Image.open(os.path.join('static/archivos', filename))
    
    buf = io.BytesIO()
    imagen.save(buf, format='JPEG', quality=90)
    imagen_comprimida = buf.getvalue()

# Opcional: Comprimir imagen con zlib
    imagen_comprimida = zlib.compress(imagen_comprimida)
            # Convertir la imagen a bytes para almacenarla en MongoDB
    imagen_bytes = Binary(imagen.tobytes())
    collection_subidas.insert_one({"datos": datos_imagen, "imagen": imagen_comprimida})
    # se añade el resultado a la lista de resultados
    resultados.append(resultado)
    os.remove(os.path.join('static/archivos', filename))
    # se devuelve el resultado
    return resultado

if __name__ == '__main__':
    app.run(debug=True)
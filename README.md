



[![Youtube Demo](./misc/Captura%20desde%202024-10-06%2008-40-02.png)](https://youtu.be/BdJ3xwUYGvU)

## Dependencias

- python 3.11
- requirements.txt

## Configuración del entorno de ejecución

Se asume que los datos del resto se encuentran, descargados, descomprimidos y localizados en la raíz del repositorio

## Orden de ejecución

- **00_download_images.py**: descarga todas las imágenes del campo cr_imágenes asociado a cada claim.Las imágenes se guardan en la carpeta `cr_images`, donde a cada imagen se le asigna un uuid que es referenciado en el registro original de los datos proporcionados.**WARNING: TARDA EN EJECUTARSE**

- **01_transform_csv.py**: genera un nuevo archivo csv, derivado del original, que contiene un subconjunto de campos y ciertos campos renombrados.

- **02_image_descriptions.ipynb**: genera un nuevo archivo csv, que contiene una columna extra, en la cual se almacena una descripción generada a partir de un LLM.**WARNING: TARDA EN EJECUTARSE**

- **03_embeddings milvus.ipnyb**: se encarga de inicializar la BD Vectorial (MilvusDB), generar los embeddings de texto, los embeddings de imagen e insertar dicha información en la BD (junto con algunos metadatos extra).**WARNING: TARDA >1h EN EJECUTARSE**

- **04_start_image_server.sh**: levanta un simple servidor http, que permite servir las imágenes descargadas, de cara a ser consumidas posteriormente por la UI desarrollada.

- **ui.pi**: interfaz de usuario que permite ejecutar el RAG, esta realizada con streamlit y se ejecuta con el comando `streamlit run ui.py`. La interfaz tiene algunas funcionalidades como:
    - generar múltiples queries a partir de la query original del usuario. Estas queries son las que realmente se usan para consultar el almacén vectorial. Los resultados de cara query son combinados y re-ordenados para construir el resultado final.
    - proporcionar los resultados de búsqueda obtenidos a un LLM para que use como contexto, junto con las imágenes asociadas
    - proporcionar ciertas explicabilidad sobre la decisión tomada por el modelo
    - capacidades de logging para controlar las operaciones de fondo realizadas


## Demos

### Elizabeth II

#### Imagen

![Imagen Elizabeth II](misc/Queen_Elizabeth_II_in_March_2015_3.jpg)

Expected Search Results: 
- Claim 1 - Newschecker (India)
- Claim 2 - Boom Live (India)

##### Texto 

    la reina elizabeth II ha alimentado a niños como si se tratasen de animales?

Expected Search Results:
- Claim 1 - Alt News (India)
- Claim 2 - Alt News (India)
- Claim 3 - ICIR Nigeria (Unknown)

##### COMBINADO

Aparecen los 5 claims y la clasificación final tiene más certeza. Ahora aparece como False y precviamente como partially False

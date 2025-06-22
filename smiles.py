import urllib.parse
import requests
import xml.etree.ElementTree as ET
import os
import time
import compound_parser
import compound_ds_writer
from typing import Dict
from datetime import datetime
from pathlib import Path
import similarity_calc

# Algunos tipos de similitud que vamos a usar, el request que hacemos a PubChem
# para buscar similitudes soporta varios tipos de similitud. En este caso, solo usamos subestructura.
# Para similitudes 3D, usamos un request especial.
SIMILARITY_TYPES = ["substructure"]
SIMILARITY_TYPES_3D = "3D"

PUB_CHEM_XML_NAMESPACE = "http://pubchem.ncbi.nlm.nih.gov/pug_rest"
URL_STRUCTURE_3D_SESARCH = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_3d/smiles/cids/TXT?smiles={smiles}"
URL_STRUCTURE_SEARCH = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{similarity_type}/smiles/XML?smiles={smiles}"
URL_STRUCTURE_SEARCH_CIDS = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listKey}/cids/TXT"
URL_COMPOUND_INFO_JSON = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/JSON"
SIMILAR_CIDS_FOLDER = Path("similar_cids")
SIMILAR_CIDS_FILE = "{smiles}_similarity_{similarity}_related_cids.txt"

TOTAL_COMPOUNDS = 0
TOTAL_COMPOUNDS_LEFT = 0
I = 0
TIMES = []

def print_progress(current, total, bar_length=60):
    percent = int((current / total) * 100)
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)
    print(f'\r0%[{bar}]100% ({percent}%)')

def download_pubchem_structure_result_cids(listKey, similarity_type, smiles, similarity_type_3d=False):
    """
    Genera un request a PubChem para obtener CIDs similares a un SMILES dado.
    Si similarity_type_3d es True, se usa un request especial para similitudes 3D.
    Si similarity_type_3d es False, se usa el request de subestructura normal.
    Esta función guarda los CIDs encontrados en un archivo de texto para consultarlos posteriormente.

    Args:
        listKey (str): El ID de la lista de PubChem para buscar CIDs.
        similarity_type (str): El tipo de similitud a usar (subestructura o 3D).
        smiles (str): El SMILES del compuesto de referencia para buscar CIDs similares.
    """
    global TOTAL_COMPOUNDS
    global TOTAL_COMPOUNDS_LEFT

    if similarity_type_3d:
        url = URL_STRUCTURE_3D_SESARCH.format(smiles=urllib.parse.quote(smiles.strip()))
        listKey = "No key needed"
        similarity_type = SIMILARITY_TYPES_3D
        print(f"URL de búsqueda de subestructura 3D: {url}")
    else:
        url = URL_STRUCTURE_SEARCH_CIDS.format(listKey=listKey)
        print(f"URL de búsqueda de CIDs: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        cids_text = response.text

        # Guardamos una linea por cada CID encontrado
        new_cids = [line.strip() for line in cids_text.splitlines() if line.strip()]

        total_new_cids = len(new_cids)
        TOTAL_COMPOUNDS += total_new_cids
        TOTAL_COMPOUNDS_LEFT += total_new_cids

        print(f"Encontré {total_new_cids} CIDs para el request con ID ListKey {listKey} y smiles {smiles}.")

        # Guardamos los CIDs en un archivo de texto.
        # El nombre del archivo es el SMILES y el tipo de similitud.
        # Si el tipo de similitud es 3D, usamos un nombre especial.
        file_name = SIMILAR_CIDS_FILE.format(smiles=smiles, similarity=similarity_type)
        file = SIMILAR_CIDS_FOLDER / file_name
        with open(file, "w", encoding="utf-8") as f:
            for cid in new_cids:
                f.write(cid + "\n")

        print(f"Archivo sobrescrito con {total_new_cids} CIDs en {file_name}.")

    except requests.exceptions.RequestException as e:
        print(f"Error during the request: {e}")

def extract_pubchem_request_id(xml_string):
    """
    Extrae el ListKey de una respuesta XML de PubChem.
    Args:
        xml_string (str): Una cadena XML de respuesta de PubChem.
    Returns:
        str: El ListKey extraído del XML, o un mensaje de error si no se encuentra.
    Raises:
        ET.ParseError: Si hay un error al analizar el XML.
    """
    try:
        root = ET.fromstring(xml_string)
        ns = {'ns': PUB_CHEM_XML_NAMESPACE}
        listkey = root.find('ns:ListKey', ns)
        if listkey is not None:
            return listkey.text
        else:
            return "No ListKey found in the XML."
    except ET.ParseError as e:
        return f"Error parsing XML: {e}"

def generate_substructure_search_process(_smiles, _similarity_type):
    """
    Genera un request a PubChem para buscar subestructuras similares a un SMILES dado.
    Esta función construye la URL de búsqueda, realiza la solicitud y extrae el ListKey
    de la respuesta XML de PubChem. El ListKey se utiliza para buscar los CIDs
    relacionados con el SMILES proporcionado.
    Args:
        _smiles (str): El SMILES del compuesto de referencia para buscar subestructuras similares.
        _similarity_type (str): El tipo de similitud a usar (subestructura o 3D).
    Returns:
        str: El ListKey extraído de la respuesta XML de PubChem, o un mensaje de error si ocurre un problema.
    Raises:
        requests.exceptions.RequestException: Si hay un error durante la solicitud HTTP.
        ET.ParseError: Si hay un error al analizar la respuesta XML.
    """
    escaped_smiles = urllib.parse.quote(_smiles.strip())
    print(f"SMILES original {_smiles}, escapado: {escaped_smiles}")
    url = URL_STRUCTURE_SEARCH.format(similarity_type=_similarity_type, smiles=escaped_smiles)
    print(f"URL de búsqueda de subestructura: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        listkey = extract_pubchem_request_id(response.text)
        return listkey
    except requests.exceptions.RequestException as e:
        return f"Error during the request: {e}"

def build_similar_substructures_list_file(smiles_list):
    """
    Consulta la base de datos de PubChem para obtener CIDs similares a los SMILES proporcionados.
    Genera un archivo de texto con los CIDs encontrados para cada SMILES y tipo de similitud.
    Esta función maneja tanto la búsqueda de subestructuras como la búsqueda de similitudes 3D.
    Esta función es la primera parte del proceso de búsqueda de subestructuras similares.

    Args:
        smiles_list (list): Lista de SMILES para buscar compuestos similares.
    """
    _process_key_to_smiles_map: Dict[str, str] = {}

    for smiles in smiles_list:
        for similarity_type in SIMILARITY_TYPES:
            print(f"Generando ListKey para SMILES: {smiles} con tipo de similitud: {similarity_type}")
            list_key = generate_substructure_search_process(smiles, similarity_type)
            print(f"ListKey generada para busqueda de subestructura. ListKey: {list_key}, SimilarityType: {similarity_type} SMILES: {smiles}")
            map_key = f"{list_key}_{similarity_type}"
            _process_key_to_smiles_map[map_key] = smiles
        
        # Para la búsqueda de similitudes 3D, usamos un tipo de request especial.
        download_pubchem_structure_result_cids(None, None, smiles, similarity_type_3d=True)

    print("Esperando 10 segundos para evitar problemas de límite de solicitudes...")
    time.sleep(10)

    for map_key, smiles in _process_key_to_smiles_map.items():
        list_key, similarity_type = map_key.split('_')
        print(f"Descargando CIDs para ListKey: {list_key} y SMILES: {smiles}")
        download_pubchem_structure_result_cids(list_key, similarity_type, smiles)

def write_csv_dataset(file, _ref_smile, similarity):
    if not os.path.exists(file):
        print(f"El archivo {file} no existe.")
        return
    with open(file, "r", encoding="utf-8") as f:
        cids = {line.strip() for line in f if line.strip()}
    if not cids:
        print(f"No se encontraron CIDs en {file}.")
        return
    print(f"Se encontraron {len(cids)} CIDs en {file}.")
    
    global TOTAL_COMPOUNDS_LEFT
    global I

    for cid in cids:
        url = URL_COMPOUND_INFO_JSON.format(cid=cid)
        print(f"Consultando CID {cid} en PubChem: {url} para smiles de referencia {_ref_smile}.")
        print_progress(I, TOTAL_COMPOUNDS)
        start = time.time()
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.text
            compound_properties = compound_parser.parse_compound_json(data)
            
            #WRITE TO CSV
            compound_ds_writer.write_to_csv(_ref_smile, compound_properties, similarity)

            I += 1
        except requests.exceptions.RequestException as e:
            print(f"Error al consultar el CID {cid}: {e}")
        finally:
            TOTAL_COMPOUNDS_LEFT -= 1
            print(f"Compuestos procesados: {I}/{TOTAL_COMPOUNDS}. Restantes: {TOTAL_COMPOUNDS_LEFT}.")
    
    print(f"Proceso finalizado. DataSet completo para SMILES: {_ref_smile}.")

def build_data_set(smiles):
    """
    Lee el archivo de CIDs similares para cada SMILES y tipo de similitud,
    y construye un DataSet en formato CSV con las propiedades de cada compuesto.
    Esta función recorre cada SMILES y tipo de similitud, generando un archivo CSV
    con las propiedades de los compuestos encontrados en PubChem.
    Cada archivo CSV contendrá las propiedades de los compuestos similares al SMILES dado,
    incluyendo información como CID, nombre IUPAC, SMILES canónico, fórmula molecular,
    aceptores de enlaces de hidrógeno, donadores de enlaces de hidrógeno,
    enlaces rotables, log P, peso molecular, número de átomos pesados,
    área de superficie polar, coeficiente de superficie atómica, IC50 contra Leishmania,
    y si cumple con las reglas de Lipinski.
    Esta función es la segunda parte del proceso de búsqueda de subestructuras similares.
    Args:
        smiles (list): Lista de SMILES para los cuales se generarán los DataSets.
    """
    for smile in smiles:
        for _similarity in SIMILARITY_TYPES:
            file_name = SIMILAR_CIDS_FILE.format(smiles=smile, similarity=_similarity)
            file = SIMILAR_CIDS_FOLDER / file_name
            write_csv_dataset(file, smile, _similarity)

        # 3D similarity search is a special case, handled separately
        print(f"Generando DataSet para SMILES: {smile} con tipo de similitud 3D.")
        file_name = SIMILAR_CIDS_FILE.format(smiles=smile, similarity=SIMILARITY_TYPES_3D)
        file = SIMILAR_CIDS_FOLDER / file_name
        write_csv_dataset(file, smile, SIMILARITY_TYPES_3D)

if __name__ == "__main__":
    start_time = datetime.now()
    inputs = []
    SIMILAR_CIDS_FOLDER.mkdir(exist_ok=True)  # Creamos el directorio si no existe

    print("Introduce los SMILEs que quieres evaluar (escribe 'ACEPTAR' para iniciar el proceso):")

    while True:
        user_input = input("> ").strip()
        
        if user_input.upper() == "ACEPTAR":
            break
        
        if user_input and user_input not in inputs:
            inputs.append(user_input)

    print("\nSMILES introducidos:")
    for item in inputs:
        print(item)

    # Hacemos la búsqueda inicial por similitud 3D y por subestructura en PubChem para tener un punto de partida.
    build_similar_substructures_list_file(inputs)
    print("\nGenerando todos los DataSet de CIDs para cada tipo de similaridad...")
    print(f"Total de compuestos a procesar: {TOTAL_COMPOUNDS}")

    # Ahora que tenemos los CIDs, construimos el DataSet.
    build_data_set(inputs)

    # Ahora calculamos las similitudes entre los compuestos.
    similarity_calc.calculate_similarity()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nProceso iniciado a las {start_time.strftime('%Y-%m-%d %H:%M:%S')}.")
    print(f"Proceso finalizado a las {end_time.strftime('%Y-%m-%d %H:%M:%S')}.")
    print(f"\nProceso completado en {duration}.")
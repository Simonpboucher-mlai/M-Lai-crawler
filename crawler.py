import os
import json
import logging
import re
import time
import io
import string
import unicodedata
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pdfplumber
from readability import Document
import openai
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Fonction pour configurer le logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Fonction pour charger la configuration depuis le fichier JSON
def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Utilitaires
def sanitize_filename(filename):
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in filename if c in valid_chars)
    filename = filename.rstrip('.')
    return filename

def normalize_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Gestion des PDF
def extract_text_from_pdf(pdf_content):
    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = ""
            for page in pdf.pages:
                # Extraire le texte brut
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                # Extraire les tableaux et les formater en texte
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        # Joindre les cellules avec un séparateur tabulaire
                        text += "\t".join([cell if cell else "" for cell in row]) + "\n"
                    text += "\n"  # Séparer les tableaux
        return text
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du contenu du PDF avec pdfplumber : {e}")
        return ""

def save_pdf(pdf_content, pdf_path):
    try:
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
        logging.info(f"PDF sauvegardé : {pdf_path}")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du PDF {pdf_path} : {e}")

# Parser les hyperliens
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hyperlinks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

def get_hyperlinks(url, user_agent, retries=3, backoff_factor=0.3):
    headers = {
        'User-Agent': user_agent
    }
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, timeout=30) as response:
                response.raise_for_status()
                if not response.headers.get('Content-Type', '').startswith("text/html"):
                    return []
                html = response.text
            break  # Succès, sortir de la boucle
        except requests.RequestException as e:
            logging.error(f"Erreur lors de la récupération de {url} (tentative {attempt + 1}/{retries}) : {e}")
            if attempt < retries - 1:
                time.sleep(backoff_factor * (2 ** attempt))  # Exponentiel backoff
                continue
            else:
                return []
    parser = HyperlinkParser()
    parser.feed(html)
    return parser.hyperlinks

def get_domain_hyperlinks(local_domain, url, user_agent):
    HTTP_URL_PATTERN = r'^http[s]*://.+'
    clean_links = []
    for link in set(get_hyperlinks(url, user_agent)):
        clean_link = None
        if re.search(HTTP_URL_PATTERN, link):
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = f"https://{local_domain}/{link}"

        if clean_link:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]

            # Ignorer les URL de postulation en ligne
            if "postulez-en-ligne" not in clean_link:
                clean_links.append(clean_link)

    return list(set(clean_links))

# Extraction de texte HTML
def extract_text_from_html(html_content):
    try:
        doc = Document(html_content)
        readable_html = doc.summary()
        soup = BeautifulSoup(readable_html, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()

        text = ""
        for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text += element.get_text() + "\n"

        return clean_text(text)
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction du contenu HTML avec readability : {e}")
        return ""

def extract_text_alternative(html_content):
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        return clean_text(soup.get_text(separator=' ', strip=True))
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction alternative du contenu HTML : {e}")
        return ""

# Interaction avec l'API OpenAI
def get_page_info(text, openai_api_key, model, max_tokens, temperature):
    try:
        openai.api_key = openai_api_key
        prompt = (
            "Vous êtes un assistant qui aide à extraire des informations spécifiques. "
            "À partir du texte fourni ci-dessous, veuillez fournir les informations suivantes de manière structurée :\n"
            "1. Mots clés en anglais.\n"
            "2. Mots clés en français.\n"
            "3. Résumé en anglais (deux phrases).\n"
            "4. Résumé en français (deux phrases).\n"
            "5. Numéro de produit principal (si disponible).\n"
            "Si un numéro de produit n'est pas disponible, indiquez 'no'.\n\n"
            "Texte:\n"
            f"{text}\n\n"
            "Format de réponse:\n"
            "Keywords (EN): [vos mots clés en anglais]\n"
            "Keywords (FR): [vos mots clés en français]\n"
            "Summary (EN): [votre résumé en anglais]\n"
            "Summary (FR): [votre résumé en français]\n"
            "Product Number: [numéro de produit ou 'no']"
        )

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Vous êtes un assistant compétent en extraction d'informations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response['choices'][0]['message']['content'].strip()

        # Parser la réponse
        info = {
            "keywords_en": "",
            "keywords_fr": "",
            "summary_en": "",
            "summary_fr": "",
            "product_number": "no"
        }

        for line in content.split('\n'):
            if line.startswith("Keywords (EN):"):
                info["keywords_en"] = line.replace("Keywords (EN):", "").strip()
            elif line.startswith("Keywords (FR):"):
                info["keywords_fr"] = line.replace("Keywords (FR):", "").strip()
            elif line.startswith("Summary (EN):"):
                info["summary_en"] = line.replace("Summary (EN):", "").strip()
            elif line.startswith("Summary (FR):"):
                info["summary_fr"] = line.replace("Summary (FR):", "").strip()
            elif line.startswith("Product Number:"):
                prod_num = line.replace("Product Number:", "").strip()
                info["product_number"] = prod_num if prod_num.lower() != 'no' else "no"

        return info

    except Exception as e:
        logging.error(f"Erreur lors de l'extraction des informations de la page via OpenAI : {e}")
        return {
            "keywords_en": "",
            "keywords_fr": "",
            "summary_en": "",
            "summary_fr": "",
            "product_number": "no"
        }

# Traitement d'une URL
def process_url(url, config, local_domain, seen, queue, lock):
    try:
        normalized_url = normalize_url(url)
        with lock:
            if normalized_url in seen:
                return
            seen.add(normalized_url)

        logging.info(f"Crawling: {url}")
        print(f"Crawling: {url}")

        headers = {
            'User-Agent': config['user_agent']
        }
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        final_url = response.url

        if response.status_code == 404:
            logging.warning(f"Page non trouvée : {url}")
            print(f"Page non trouvée : {url}")
            return

        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        extracted_text = ""

        if 'application/pdf' in content_type or final_url.lower().endswith('.pdf'):
            # Sauvegarder le PDF
            pdf_filename = sanitize_filename(unquote(final_url.split('/')[-1]))
            pdf_path = os.path.join(config['pdf_directory'], f"{pdf_filename}.pdf")
            save_pdf(response.content, pdf_path)

            # Extraire le texte du PDF
            extracted_text = extract_text_from_pdf(response.content)
            if not extracted_text.strip():
                logging.warning(f"Contenu PDF vide : {final_url}")
                print(f"Contenu PDF vide : {final_url}")
                return
        elif 'text/html' in content_type:
            extracted_text = extract_text_from_html(response.content)
            if not extracted_text.strip():
                extracted_text = extract_text_alternative(response.content)
            if not extracted_text.strip():
                logging.warning(f"Contenu HTML vide : {final_url}")
                print(f"Contenu HTML vide : {final_url}")
                return
        else:
            logging.info(f"Type de contenu non supporté pour : {final_url}")
            return

        # Extraire les informations de la page via OpenAI
        page_info = get_page_info(
            extracted_text,
            config['openai_api_key'],
            config['openai_model'],
            config['openai_max_tokens'],
            config['openai_temperature']
        )

        # Préparer l'en-tête avec les informations extraites
        header = (
            f"lien: {final_url}\n"
            f"mot clé anglais: {page_info['keywords_en']}\n"
            f"mot clé français: {page_info['keywords_fr']}\n"
            f"résumé (EN): {page_info['summary_en']}\n"
            f"résumé (FR): {page_info['summary_fr']}\n\n"
        )

        # Ajouter le numéro de produit
        if page_info["product_number"].lower() != "no":
            header += f"#pro : {page_info['product_number']}\n"
        else:
            header += "#pro: no\n"

        header += "\n------\n\n"

        # Générer un nom de fichier sécurisé
        filename = sanitize_filename(unquote(final_url.split('/')[-1] or "index"))
        txt_path = os.path.join(config['output_directory'], local_domain, f"{filename}.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write(header + extracted_text)
        logging.info(f"Contenu sauvegardé : {final_url}")
        print(f"Contenu sauvegardé : {final_url}")

        # Ajouter les nouveaux liens à la queue
        new_links = get_domain_hyperlinks(local_domain, final_url, config['user_agent'])
        with lock:
            for link in new_links:
                normalized_link = normalize_url(link)
                if normalized_link not in seen:
                    queue.append(link)
                    logging.info(f"Ajouté à la queue : {link}")
                    print(f"Ajouté à la queue : {link}")

    except requests.RequestException as e:
        logging.error(f"Erreur lors du crawl de {url} : {e}")
        print(f"Erreur lors du crawl de {url} : {e}")
    except Exception as e:
        logging.error(f"Erreur inattendue lors du crawl de {url} : {e}")
        print(f"Erreur inattendue lors du crawl de {url} : {e}")

# Fonction principale de crawling
def crawl(config):
    print(f"Démarrage du crawl sur : {config['start_url']}")
    logging.info(f"Démarrage du crawl sur : {config['start_url']}")
    local_domain = urlparse(config['start_url']).netloc
    queue = deque([config['start_url']])
    seen = set()
    processed_count = 0
    max_pages = config['max_pages']
    num_workers = config.get('num_workers', 5)
    lock = threading.Lock()

    # Créer les dossiers nécessaires
    if not os.path.exists(config['output_directory']):
        os.makedirs(config['output_directory'])
        logging.info(f"Création du dossier {config['output_directory']}/")
    domain_output = os.path.join(config['output_directory'], local_domain)
    if not os.path.exists(domain_output):
        os.makedirs(domain_output)
        logging.info(f"Création du dossier {domain_output}/")

    if not os.path.exists(config['pdf_directory']):
        os.makedirs(config['pdf_directory'])
        logging.info(f"Création du dossier {config['pdf_directory']}/")

    with ThreadPoolExecutor(max_workers=num_workers) as executor, tqdm(total=max_pages, desc="Pages crawled") as pbar:
        futures = []
        while queue and processed_count < max_pages:
            with lock:
                if not queue:
                    break
                current_url = queue.popleft()

            future = executor.submit(process_url, current_url, config, local_domain, seen, queue, lock)
            futures.append(future)
            processed_count += 1
            pbar.update(1)

            # Limiter le nombre de futures en attente
            if len(futures) >= num_workers * 2:
                done, futures = wait_futures(futures, num_workers)

        # Attendre que toutes les futures soient terminées
        for future in as_completed(futures):
            pass

    logging.info("Crawling et extraction de texte terminés.")
    print("Crawling et extraction de texte terminés.")

def wait_futures(futures, num_workers):
    done = []
    for future in as_completed(futures, timeout=None):
        done.append(future)
        if len(done) >= num_workers:
            break
    remaining = [f for f in futures if f not in done]
    return done, remaining

# Fonction principale
def main():
    # Charger la configuration depuis le fichier JSON
    config = load_config('config.json')

    # Configurer le logging
    setup_logging(config.get('log_file', 'crawler_log.txt'))
    logging.info("Configuration du logging terminée.")

    # Vérifier la clé API OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        logging.error("La clé API OpenAI n'est pas définie. Veuillez vérifier votre fichier .env.")
        print("Erreur : La clé API OpenAI n'est pas définie. Veuillez vérifier votre fichier .env.")
        exit(1)
    else:
        logging.info("Clé API OpenAI chargée avec succès.")
        print("Clé API OpenAI chargée avec succès.")

    # Ajouter les configurations OpenAI au dictionnaire de configuration
    config['openai_api_key'] = openai_api_key

    # Exécuter le crawler
    crawl(config)

if __name__ == "__main__":
    main()

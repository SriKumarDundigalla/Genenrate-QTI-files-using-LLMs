import os
import PyPDF2
import openai
import tiktoken
from dotenv import load_dotenv
import logging
import nbformat
import jsonpatch
# class Learning_Outcomes(BaseModel):
#     learning_outcomes : List[str] = Field(description="list of learning outcomes")
import matplotlib.pyplot as plt
import networkx as nx
# Load environment variables from the .env file
import community as community_louvain
load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import warnings
import shutil
import json
from langchain_text_splitters import TokenTextSplitter
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import nbformat
import PyPDF2
import ast
warnings.filterwarnings("ignore")
# Configure basic logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_directory(directory):
    """
    Analyzes the content of the given directory and returns details of specific file types.

    :param directory: The path of the directory to analyze.
    :return: A list of dictionaries, each containing 'name', 'type', 'size', and 'path' of the file.
    """
    logging.info(f"Analyzing directory: {directory}")
    supported_extensions = {'.md', '.ipynb','.pdf'} #'.py'
    file_details = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            extension = os.path.splitext(file_path)[1]

            if extension in supported_extensions:
                file_info = {
                    'name': file,
                    'type': extension,
                    'size': os.path.getsize(file_path),
                    'path': file_path
                }
                file_details.append(file_info)
                logging.info(f"File added for processing: {file_path}")

    return file_details


def clean_content(content):
    """
    Performs cleaning of the file content, including trimming whitespace and removing non-printable characters.
    :param content: Raw content string to be cleaned.
    :return: Cleaned content string.
    """
    content = content.strip()  # Remove leading and trailing whitespace
    content = content.replace('\x00', '')  # Remove null bytes if present
    # Normalize line breaks and whitespace
    content = content.replace('\n', ' ')  # Replace new lines with spaces
    content = re.sub(r'\s+', ' ', content)  # Replace multiple spaces with a single space

    # Remove non-printable characters
    content = ''.join(char for char in content if char.isprintable() or char in ('\n', '\t', ' '))
    # Remove non-printable characters, including the replacement character
    content = re.sub(r'[^\x20-\x7E]+', '', content)
    return content

def read_file_content(file_info):
    """
    Reads the content of a file based on its type and returns the cleaned content as a string.
    :param file_info: Dictionary containing the file's details.
    :return: Cleaned content of the file as a string.
    """
    file_path = file_info['path']
    file_type = file_info['type']
    content = ''

    try:
        if file_type == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() if page.extract_text() else ''
                    content += clean_content(page_text)
        elif file_type == '.ipynb':
            with open(file_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                for cell in nb.cells:
                    cell_content = cell.source + '\n\n'  # Add cell content
                    content += clean_content(cell_content)
        else:  # Assuming '.py' or other plaintext files
            with open(file_path, 'r', encoding='utf-8') as f:
                content = clean_content(f.read())

        logging.info(f"Successfully read and cleaned content from: {file_path}")
    except Exception as e:
        logging.exception(f"Error reading {file_path}: {e}")

    return content


def get_file_contents(file_details):
    """
    Retrieves the contents of each file based on the provided file details.

    :param file_details: List of dictionaries containing file details.
    :return: A list of dictionaries, each containing 'path' and 'content' of the file.
    """
    content_details = []
    for file_info in file_details:
        file_content = read_file_content(file_info)
        if file_content:
            content_details.append({
                'path': file_info['path'],
                'content': file_content
            })

    return content_details
def process_and_insert_contents(file_contents, persist_directory):
    """
    Processes the contents of each file, splits them, embeds, and inserts into a database.

    :param file_contents: List of dictionaries containing file paths and their contents.
    :param persist_directory: The directory to persist any necessary data for database insertion.
    """
    # Initialize the text splitter and embedding tools
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150,length_function=len)
    embedding = OpenAIEmbeddings()
    all_page_contents = []  # Collect all page contents for further processing or analysis
    # Extract page_content from each Document
    for content_detail in file_contents:
        # Split the content
        documents  = text_splitter.create_documents([content_detail['content']])
        for document in documents:
            page_content = document.page_content  # Accessing the page_content attribute
            all_page_contents.append(page_content)
        # Here, you would generate embeddings and insert them into your database
        # This is a placeholder to illustrate the process
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        # Logging or any other operation after insertion
        logging.info(f"Processed and inserted content from: {content_detail['path']}")
    return vectordb
def summarize_files(file_details):
    """
    Processes the content of files whose content exceeds a specified global token size,
    by splitting the content into chunks. Each chunk's size is determined to ensure it 
    doesn't exceed the global token size limit. The function returns a list of dictionaries 
    with the filename/path, chunked content, and the token size of each chunk.

    :param file_details: List of dictionaries with file details.
    :return: A list of dictionaries with filename/path and chunked content.
    """
    global_token_size = int(os.getenv('GLOBAL_TOKEN_SIZE'))
    Overlap = 500  # Example overlap size, adjust as needed
    summarized_files = []

    for file in file_details:
        original_token_count = len(tiktoken.encoding_for_model("gpt-4").encode(file['content']))

        if original_token_count > global_token_size:
            # Calculate the number of chunks needed
            N = 1 + (original_token_count - global_token_size) // (global_token_size - Overlap)

            # Initialize the splitter with calculated chunk size and overlap
            splitter = RecursiveCharacterTextSplitter( chunk_size = original_token_count // N ,  chunk_overlap = Overlap, length_function=len, is_separator_regex=False)
            # Split the content into documents/chunks
            documents = splitter.create_documents([file['content']])
            for document in documents:
                summarized_files.append({
                        'content': document.page_content,
                        'token_size': len(tiktoken.encoding_for_model("gpt-4").encode(document.page_content))
                    })   
        else:
            # If the content does not exceed global token size, add it directly
            summarized_files.append({
                'path': file['path'],
                'content': file['content'],
                'token_size': original_token_count
            })

    return summarized_files

def create_chunks_from_content_greedy(file_contents, context_window_size):
    """
    Creates content chunks from a list of file content dictionaries using a Greedy approach, 
    ensuring that each chunk does not exceed a specified context window size in terms of tokens.

    Parameters:
    - file_contents (list of dict): A list of dictionaries, where each dictionary contains 
      'content' (str) and 'token_size' (int) keys. 'content' is the text of the file, and 
      'token_size' is the number of tokens that text consists of.
    - context_window_size (int): The maximum number of tokens that a single chunk can contain. 
      It defines the upper limit for the size of each content chunk.

    Returns:
    - list of str: A list of content chunks, where each chunk is a string composed of file contents 
      that together do not exceed the context window size. Each content is enclosed in triple backticks.
    """
    all_chunks = []  # Initialize the list to hold all content chunks
    current_chunk = ""  # Initialize the current chunk as an empty string
    current_token_count = 0  # Initialize the current token count to 0

    # Sort file_contents by 'token_size' in descending order
    sorted_file_contents = sorted(file_contents, key=lambda x: x['token_size'], reverse=True)

    for content in sorted_file_contents:
        # If adding this content exceeds the context window size, start a new chunk
        if current_token_count + content['token_size'] > context_window_size:
            if current_chunk:  # Ensure the current chunk is not empty
                all_chunks.append(current_chunk)  # Add the current chunk to all_chunks
                current_chunk = ""  # Reset the current chunk
                current_token_count = 0  # Reset the token count for the new chunk

        # Add the content to the current chunk if it fits, enclosed in triple backticks
        if current_token_count + content['token_size'] <= context_window_size:
            current_chunk += "\n" +"```" + content['content'] + "```" + "\n"  # Enclose content in backticks and append with a newline for readability
            current_token_count += content['token_size']
    
    # Add the last chunk if it contains any content, ensuring it's also enclosed in backticks
    if current_chunk:
        all_chunks.append(current_chunk)

    return all_chunks

    
def extract_key_topic(outcome):
    ignore_words = {"understand", "develop", "analyze", "the", "of", "using", "with", "for", "to", "and", "basic", "fundamentals"}
    words = re.sub(r'[^\w\s]', '', outcome.lower()).split()
    significant_words = [word for word in words if word not in ignore_words and len(word) > 4]
    
    # If after filtering there are multiple words, attempt to form a phrase that makes sense
    if len(significant_words) > 1:
        # Prioritize the last two words, often more specific in educational content
        return " ".join(significant_words[-2:])
    elif significant_words:
        return significant_words[0]
    else:
        return "General"

def draw_similarity_graph(indices, cosine_sim, title, threshold, extracted_topics):
    G = nx.Graph()

    # Add nodes with document labels
    for i in indices:
        G.add_node(i, label=f"Doc {i+1}")

    # Add edges between nodes if similarity exceeds the threshold
    for i in indices:
        for j in indices:
            if i != j and cosine_sim[i][j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i][j])
    
    # Use the Louvain method to find communities for color-coding
    partition = community_louvain.best_partition(G)
    # Generate a color map from the partition
    topic_to_color = {topic: i for i, topic in enumerate(set(extracted_topics))}
    color_map = [topic_to_color[extracted_topics[node]] for node in G]

    pos = nx.spring_layout(G)  # Positions for all nodes

    plt.figure(figsize=(12, 10))
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.axis('off')

    # Drawing nodes with community color coding
    nx.draw_networkx_nodes(G, pos, node_size=500, cmap=plt.cm.viridis, node_color=color_map, alpha=0.8)

    # Drawing edges
    edges = G.edges(data=True)
    weights = [d['weight']*10 for u, v, d in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)

    # Drawing labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

    # Create a legend for the topics
    legend_labels = {value: key for key, value in topic_to_color.items()}
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                      markerfacecolor=plt.cm.viridis(i / len(legend_labels)), markersize=10) for i in legend_labels]
    
    # Adjusting legend position to avoid overlap with the graph
    plt.legend(handles=legend_patches, title="Topics", loc='upper left', bbox_to_anchor=(1,1), fontsize='small')
    
    # Adjust subplot parameters to give the legend some room
    plt.subplots_adjust(right=0.7)
    
    plt.show()

 # Define your desired data structure for learning outcomes.
class LearningOutcomes(BaseModel):
    outcomes: List[str] = Field(description="List of learning outcomes")
 # Set up a parser to enforce the output structure.
parser = PydanticOutputParser(pydantic_object=LearningOutcomes)

def generate_learning_outcomes_for_chunks(documents):
    api_key = os.getenv('OPENAI_API_KEY')
    delimiter = "###"
    chunk_LOs = {}  # Dictionary to hold learning outcomes for each chunk

    # Initialize OpenAI client with your API key
    client = openai.OpenAI(api_key=api_key)

    # The number of outcomes to generate per chunk, adjust as needed or dynamically set
    number_of_outcomes = int(os.getenv('LOs_PER_CHUNK', 5))

    
    system_message = f"""
      As a Professor with expertise in curriculum development and crafting learning outcomes, 
      your task is to extract and enumerate {number_of_outcomes} distinct learning outcomes from the 
      provided course content. This content includes programming code, with each topic or code example 
      distinctly separated by triple backticks ```. Your challenge is to recognize and interpret these 
      segmented topics, especially those conveyed through programming code, to determine their thematic 
      and practical contributions to the course. These outcomes should address the comprehensive skills 
      and knowledge base essential to the subject matter, with a special emphasis on the interpretation 
      and application of programming concepts as demonstrated within the code segments. 
      The learning outcomes should be formatted as a Python list, precisely containing {number_of_outcomes} 
      entries. Each entry must represent a unique learning outcome that students are expected to achieve by 
      the end of the course, directly informed by the theoretical content and the 
      practical programming code examples provided.
    """
    all_out_comes=[]
    # Generate learning outcomes for each chunk
    for index, chunk in enumerate(documents, start=1):
        logging.info(chunk)
        user_message = f"Generate Lerning outcomes for the following content enclosed by triple hashtag{delimiter}{chunk}{delimiter}."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0
        )
        
        summary = response.choices[0].message.content
        start = summary.find("[")
        end = summary.rfind("]") + 1
        outcome_list = eval(summary[start:end])
        
        all_out_comes.append(outcome_list)

    
    # Flatten each list of outcomes into a single string per list to simplify the example
    documents = [item for outcome_list in all_out_comes for item in outcome_list]

    print(documents)
    return
    extracted_topics = [extract_key_topic(outcome) for outcome in documents]

    # Vectorize the documents
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between documents
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Determine similarity threshold
    threshold = 0.2  # Example threshold, adjust based on your needs

    # Filter documents based on similarity (naive approach)
    filtered_indices = []
    for i in range(len(cosine_sim)):
        if not any(cosine_sim[i][j] > threshold and j != i for j in filtered_indices):
            filtered_indices.append(i)

    # Extract the filtered documents based on filtered_indices
    filtered_documents = [documents[i] for i in filtered_indices]

    logging.info(f"Original number of documents: {len(documents)}")
    logging.info(f"Filtered number of documents: {len(filtered_documents)}")

    # Visualization before and after filtering
    # Assuming draw_similarity_graph is a function defined to create graphs
    all_indices = range(len(cosine_sim))
    # draw_similarity_graph(list(range(len(cosine_sim))), cosine_sim, "Before Filtering Similarity Graph", threshold, extracted_topics)
    # filtered_extracted_topics = [extracted_topics[i] for i in filtered_indices]
    # draw_similarity_graph(filtered_indices, cosine_sim, "After Filtering Similarity Graph", threshold, filtered_extracted_topics)
    print(filtered_documents)
    return filtered_documents

def find_most_relevant_learning_outcome_document(vectordb, learning_outcomes):
    """
    Uses vectordb to find the most relevant learning outcome document from the database for each topic.

    :param vectordb: The vectordb instance configured for retrieval.
    :param learning_outcomes: A list of lists, where each sublist represents learning outcomes related to a specific topic.
    :return: A list of tuples, each containing the most relevant document's content and its relevance score for each list of learning outcomes.
    """
    # Initialize the vectordb retriever with 'k' set to 1 to retrieve the most relevant document
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 1})

    documents=[]
    for LOS in learning_outcomes:
        outcome_page_mapping={}
        for i in LOS:
            docs = retriever.get_relevant_documents(i)
            outcome_page_mapping[i]=docs[0].page_content
        documents.append(outcome_page_mapping)
    return documents

def format_learning_outcomes_with_identifiers(learning_outcomes):
    formatted_strings = []
    outcome_counter = 1  # Initialize counters for learning outcomes
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    for outcome in learning_outcomes:
        formatted_string=""
        for key, value in outcome.items():
            # Format string with counters
            formatted_string = f"learning_outcome_{outcome_counter}: {key}\nrelated_content_{outcome_counter}: {value}\n"
            outcome_counter += 1  # Increment the learning outcome counter
        formatted_strings.append(formatted_string)
    system_message=f"""
                    You are a professor tasked with developing a concise series of multiple-choice quiz questions for your students, each aligned with a distinct learning outcome and directly related to specific content. Your objective is to ensure that each question not only integrates with the learning material but also that its correct answer is unequivocally found within the provided content. To accomplish this, follow the enhanced approach detailed below, which includes steps for identifying a similar main heading to unify the theme of your quiz.
                    **Enhanced Steps to Follow:**

                    1. **Synthesize the Core Theme:**
                    - Identify and define a central theme that encapsulates all six learning outcomes and their associated content. This theme will be the focal point of your quiz, guiding the formulation of questions and answers.

                    2. **Broaden the Theme with a Supplementary Heading:**
                    - Expand the quiz's scope by incorporating a supplementary theme that complements the core theme. This additional perspective should still connect directly to the six learning outcomes, enriching the quiz's thematic depth.

                    3. **Analyze Learning Outcomes and Related Content:**
                    - Thoroughly review each of the six learning outcomes and their corresponding content. Aim to extract key insights and knowledge that are essential to the core and supplementary themes, ensuring a comprehensive understanding that will be reflected in the quiz questions.

                    4. **Craft Themed Multiple-Choice Questions:**
                    - Create six multiple-choice questions, one for each learning outcome, that highlight critical aspects of the related content. Each question should align with both the core and supplementary themes, maintaining thematic consistency throughout the quiz.

                    5. **Select Correct Answers and Design Distractors:**
                    - Choose the correct answer for each question based on the related content. Then, develop three distractors for each question. These distractors should be relevant and plausible but incorrect, according to the content, ensuring the quiz accurately assesses the learner's understanding of the themes.

                    6. **Ensure Verification of Output:**
                    - Verify that the output (the quiz) includes six multiple-choice questions that adhere to the guidelines above. Each question should be clearly linked to one of the learning outcomes and accurately reflect both the core and supplementary themes.

                    This structured approach ensures each quiz question is directly tied to a learning outcome and related content, with a clear thematic link and verified structure for assessing learners' understanding.
                     
                    **Implementation Directive:**
                        If you are working with 6 learning outcomes and their related content, this process will result in 6 multiple-choice questions. Each question is tailored to its corresponding learning outcome, maintaining a strict one-to-one ratio between learning outcomes and questions, thereby ensuring a focused and effective evaluation of student understanding within the context of the similar main heading.

                    **Example Output Format:**

                        "Artificial Intelligence Essentials": "**1. What is Artificial Intelligence (AI)?**\nA) The simulation of human intelligence in machines\nB) A new internet technology\nC) A database management system\nD) A type of computer hardware\n**Answer: A) The simulation of human intelligence in machines**\n\n**2. Which of the following is a primary application of AI?**\nA) Data storage\nB) Speech recognition\nC) Website hosting\nD) Network cabling\n**Answer: B) Speech recognition**\n\n**3. What is a Neural Network?**\nA) A social media platform\nB) A computer system modeled on the human brain\nC) A new programming language\nD) A cybersecurity protocol\n**Answer: B) A computer system modeled on the human brain**\n\n**4. What does 'Natural Language Processing' (NLP) enable computers to do?**\nA) Increase processing power\nB) Understand and interpret human language\nC) Cool down servers\nD) Connect to the internet\n**Answer: B) Understand and interpret human language**\n\n**5. What is 'machine vision'?**\nA) A new type of display technology\nB) The ability of computers to 'see' and process visual data\nC) A marketing term for high-resolution screens\nD) A feature in video games\n**Answer: B) The ability of computers to 'see' and process visual data** "\n\n**6. How does AI impact the field of Robotics?**\nA) By reducing the cost of computer hardware\nB) By enabling robots to learn from their environment and improve their tasks\nC) By increasing the weight of robots\nD) By decreasing the need for internet connectivity\n**Answer: B) By enabling robots to learn from their environment and improve their tasks**"

                        "Data Science Introduction": "**1. What is the primary purpose of data analysis?**\nA) To store large amounts of data\nB) To transform data into meaningful insights\nC) To create visually appealing data presentations\nD) To increase data storage capacity\n**Answer: B) To transform data into meaningful insights**\n\n**2. Which programming language is most commonly used in data science?**\nA) Java\nB) Python\nC) C++\nD) JavaScript\n**Answer: B) Python**\n\n**3. What is a DataFrame in the context of data science?**\nA) A method to secure data\nB) A 3D representation of data\nC) A data structure for storing data in tables\nD) A type of database\n**Answer: C) A data structure for storing data in tables**\n\n**4. What does 'machine learning' refer to?**\nA) The process of programming machines to perform tasks\nB) The ability of a machine to improve its performance based on previous results\nC) The science of making machines that require energy\nD) The study of computer algorithms that improve automatically through experience\n**Answer: D) The study of computer algorithms that improve automatically through experience**\n\n**5. What is 'big data'?**\nA) Data that is too large to be processed by traditional databases\nB) A large amount of small datasets\nC) Data about big things\nD) A type of data visualization\n**Answer: A) Data that is too large to be processed by traditional databases**"\n\n**6. What role does data visualization play in data science?**\nA) To make databases run faster\nB) To improve data storage efficiency\nC) To represent data in graphical format for easier interpretation\nD) To encrypt data for security\n**Answer: C) To represent data in graphical format for easier interpretation**"
  
                        "Web Development Fundamentals": "**1. Which language is primarily used for structuring web pages?**\nA) CSS\nB) JavaScript\nC) HTML\nD) Python\n**Answer: C) HTML**\n\n**2. What does CSS stand for?**\nA) Cascading Style Scripts\nB) Cascading Style Sheets\nC) Computer Style Sheets\nD) Creative Style Sheets\n**Answer: B) Cascading Style Sheets**\n\n**3. What is the purpose of JavaScript in web development?**\nA) To add interactivity to web pages\nB) To structure web pages\nC) To style web pages\nD) To send data to a server\n**Answer: A) To add interactivity to web pages**\n\n**4. Which HTML element is used to link a CSS stylesheet?**\nA) <script>\nB) <link>\nC) <css>\nD) <style>\n**Answer: B) <link>**\n\n**5. What does AJAX stand for?**\nA) Asynchronous JavaScript and XML\nB) Automatic JavaScript and XML\nC) Asynchronous Java and XML\nD) Automatic Java and XML\n**Answer: A) Asynchronous JavaScript and XML**"\n\n**6. What is responsive web design?**\nA) Designing websites to respond to user behavior and environment based on screen size, platform, and orientation\nB) A design technique to make web pages load faster\nC) Creating web pages that respond to voice commands\nD) Developing websites that automatically update content\n**Answer: A) Designing websites to respond to user behavior and environment based on screen size, platform, and orientation**"\n\n**6. What is the difference between a virus and a worm?**\nA) A virus requires human action to spread, whereas a worm can propagate itself\nB) A worm is a type of antivirus software\nC) A virus can only affect data, not hardware\nD) Worms are beneficial software that improve system performance\n**Answer: A) A virus requires human action to spread, whereas a worm can propagate itself**"

                        "Cybersecurity Basics": "**1. What is the primary goal of cybersecurity?**\nA) To create new software\nB) To protect systems and networks from digital attacks\nC) To improve computer speed\nD) To promote open-source software\n**Answer: B) To protect systems and networks from digital attacks**\n\n**2. What is phishing?**\nA) A technique to fish information from the internet\nB) A cyberattack that uses disguised email as a weapon\nC) A firewall technology\nD) A data analysis method\n**Answer: B) A cyberattack that uses disguised email as a weapon**\n\n**3. What does 'encryption' refer to in the context of cybersecurity?**\nA) Converting data into a coded format to prevent unauthorized access\nB) Deleting data permanently\nC) Copying data to a secure location\nD) Monitoring data access\n**Answer: A) Converting data into a coded format to prevent unauthorized access**\n\n**4. What is a VPN used for?**\nA) Increasing internet speed\nB) Protecting online privacy and securing internet connections\nC) Creating websites\nD) Developing software\n**Answer: B) Protecting online privacy and securing internet connections**\n\n**5. What is malware?**\nA) Software used to perform malicious actions\nB) A new programming language\nC) A data analysis tool\nD) A type of computer hardware\n**Answer: A) Software used to perform malicious actions**"\n\n**6. What is the difference between a virus and a worm?**\nA) A virus requires human action to spread, whereas a worm can propagate itself\nB) A worm is a type of antivirus software\nC) A virus can only affect data, not hardware\nD) Worms are beneficial software that improve system performance\n**Answer: A) A virus requires human action to spread, whereas a worm can propagate itself**"

                        "Cloud Computing Fundamentals": "**1. What is cloud computing?**\nA) Storing and accessing data over the internet\nB) Predicting weather patterns\nC) Computing at high altitudes\nD) A new web development framework\n**Answer: A) Storing and accessing data over the internet**\n\n**2. Which of the following is a benefit of cloud computing?**\nA) Reduced IT costs\nB) Increased data loss\nC) Slower internet speeds\nD) More hardware requirements\n**Answer: A) Reduced IT costs**\n\n**3. What is SaaS?**\nA) Software as a Service\nB) Storage as a System\nC) Security as a Software\nD) Servers as a Service\n**Answer: A) Software as a Service**\n\n**4. What does 'scalability' mean in the context of cloud computing?**\nA) Decreasing the size of databases\nB) The ability to increase or decrease IT resources as needed\nC) The process of moving to a smaller server\nD) Reducing the number of cloud services\n**Answer: B) The ability to increase or decrease IT resources as needed**\n\n**5. What is a 'public cloud'?**\nA) A cloud service that is open for anyone to use\nB) A weather phenomenon\nC) A private network\nD) A type of VPN\n**Answer: A) A cloud service that is open for anyone to use**"\n\n**6. What is IaaS?**\nA) Internet as a Service\nB) Infrastructure as a Service\nC) Information as a System\nD) Integration as a Service\n**Answer: B) Infrastructure as a Service**"

                    **Proceed with this format for all questions, and is answerable based on the provided content. This comprehensive approach ensures a focused, educational, and thematic quiz that effectively assesses students' understanding and engagement with the material.** 
                """
    Quetions=[]
    for i in formatted_strings:
        user_message = f"Create multiple-choice questions based on the specified learning outcomes and their associated content within triple hashtags . Content details are provided below: ###{i}###."
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message.strip()},
                {"role": "user", "content": user_message.strip()}
            ],
            temperature=0
        )
        
        summary = response.choices[0].message.content
        Quetions.append(summary)
    return Quetions
        

def remove_old_database_files(directory_path='./docs/chroma'):
    """
    Removes the specified directory and all its contents.

    :param directory_path: Path to the directory to be removed.
    """
    try:
        # Check if the directory exists
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Remove the directory and all its contents
            shutil.rmtree(directory_path)
            logging.info(f"Successfully removed directory: {directory_path}")
        else:
            logging.info(f"Directory does not exist, no need to remove: {directory_path}")
    except Exception as e:
        logging.exception(f"Error removing directory {directory_path}: {e}")

def generate_markdown_file(Quetions):
    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)
    Content = "\n".join(Quetions)
    system_message = f"""

    You are tasked with creating a markdown document that efficiently organizes and presents given input content. Utilize your expertise in markdown formatting to structure this document according to the specified guidelines. Perform the following actions to achieve the desired template format:

        1. **Generate a Summary Heading**: Examine the input content to identify a all learning outcomes from core topic heading . Use this insight to create a dynamic heading for the summary section. This heading should reflect the core heading of the summary.

        2. **Write a Summary**: Under the dynamic heading you've created, craft a concise summary about 100 words long, capturing the essence of the input content.

        3. **List Learning Outcomes**: Under the '## Learning Outcomes' section, enumerate the learning outcomes provided, each with a one-line explanation derived from the core themes of the input content. You can find learning outcome from Core Theme heading in content

        4. **Map Learning Outcomes to Questions**: In the '## Mapping of LO's to Questions' section, create a clear mapping of Learning Outcomes to their corresponding question numbers, following the provided template structure.

        5. **Organize Multiple Choice Questions and Answers**: List out the MCQs under the '## Multiple Choice Questions and Answers' section as they appear in the content, including each question followed by its options (A, B, C, D) and the correct answer as per Map Learning Outcomes to Questions.

    Ensure all details are accurately extracted and formatted according to the guidelines. The number and specifics of questions and learning outcomes might vary, but the arrangement should adhere to the template structure.

    Given content for transformation should be analyzed for a summary heading, followed by sections for learning outcomes, mapping these outcomes to questions, and a series of multiple-choice questions with answers, according to the outline template below:

            # <Dynamic Summary Heading Extracted from Content>

            <Your 100-word summary here>

            ## Learning Outcomes

            1. **Learning Outcome 1**: <One-line explanation>
            2. **Learning Outcome 2**: <One-line explanation>
            N. **Learning Outcome N**: <One-line explanation>

            ## Mapping of LO's to Questions

            | Learning Outcome | Corresponding Question Numbers |
            |------------------|--------------------------------|
            | Learning Outcome 1 | 1, 2, 3 |
            | Learning Outcome 2 | 4, 5, 6 |
            ...
            | Learning Outcome N | N1, N2, N3 |

            ## Multiple Choice Questions and Answers

            **1. Question?**
            A) Option 1
            B) Option 2
            C) Option 3
            D) Option 4
            **Answer: B)**

            **2. Question?**
            A) Option 1
            B) Option 2
            C) Option 3
            D) Option 4
            **Answer: A)**

            **N. Question?**
            A) Option 1
            B) Option 2
            C) Option 3
            D) Option 4
            **Answer: D)**

        Example output:  
                        # Summary of MongoDB

                        MongoDB is a NoSQL database that provides high performance, high availability, and easy scalability. It works on the concept of collections and documents. MongoDB offers a rich set of features such as full index support, replication, sharding, and flexible data processing and aggregation. It's designed to handle large volumes of data and offers a robust solution for storing and retrieving data in a format that is both convenient and efficient.

                        ## Learning Outcomes

                        1. **Understanding MongoDB and NoSQL Databases:** Introduction to MongoDB and the basic concepts of NoSQL databases.

                        2. **CRUD Operations:** Understanding the Create, Read, Update, and Delete operations in MongoDB.

                        3. **Data Modeling and Indexing:** Learn about data modeling concepts, schema design, and indexing in MongoDB.

                        4. **Advanced Features:** Explore advanced MongoDB features like aggregation, replication, and sharding.

                        5. **Security and Administration:** Understanding MongoDB security features and basic database administration.

                        6. **Performance Tuning and Optimization:** Learn how to optimize and tune MongoDB performance.

                        7. **MongoDB and Big Data:** Understanding how MongoDB can be used for big data applications.

                        ## Mapping of LO's to questions

                        | Learning Outcome | Corresponding Question Numbers |
                        |------------------|--------------------------------|
                        | Understanding MongoDB and NoSQL Databases | 1, 2, 3 |
                        | CRUD Operations | 4, 5, 6, 7 |
                        | Data Modeling and Indexing | 8, 9, 10 |
                        | Advanced Features | 11, 12, 13 |
                        | Security and Administration | 14, 15, 16 |
                        | Performance Tuning and Optimization | 17, 18, 19 |
                        | MongoDB and Big Data | 20, 21, 22 |

                        ## Multiple Choice Questions and Answers

                        **1. What type of database is MongoDB?**
                        A) Relational
                        B) NoSQL
                        C) Graph
                        D) SQL
                        **Answer: B) NoSQL**

                        **2. What is a 'Document' in MongoDB?**
                        A) A text file
                        B) A table
                        C) A record in a collection
                        D) A query
                        **Answer: C) A record in a collection**

                        **3. Which format does MongoDB use to store data?**
                        A) XML
                        B) JSON
                        C) CSV
                        D) HTML
                        **Answer: B) JSON**

                        **4. How do you create a new collection in MongoDB?**
                        A) Using the 'newCollection' command
                        B) It is created automatically when you insert the first document
                        C) Through the MongoDB user interface
                        D) With the 'createCollection' method
                        **Answer: B) It is created automatically when you insert the first document**

                        **5. In MongoDB, how do you find a document with a specific field value?**
                        A) find()
                        B) select * from documents where field = value
                        C) getDocument(field, value)
                        D) fieldValue(field = value)
                        **Answer: A) find()**

                        **6. Which command is used to update a document in MongoDB?**
                        A) updateDocument()
                        B) modify()
                        C) save()
                        D) update()
                        **Answer: D) update()**

                        **7. How can you delete a document in MongoDB?**
                        A) remove()
                        B) deleteDocument()
                        C) erase()
                        D) delete()
                        **Answer: A) remove()**

                        **8. What is 'Indexing' in MongoDB?**
                        A) A way to organize documents in alphabetical order
                        B) Creating unique identifiers for documents
                        C) Enhancing the performance of database operations
                        D) Encrypting data for security purposes
                        **Answer: C) Enhancing the performance of database operations**

                        **9. In MongoDB, what is 'Sharding'?**
                        A) Fragmenting data across multiple servers
                        B) Encrypting data for security
                        C) Merging multiple collections into one
                        D) Backing up the database
                        **Answer: A) Fragmenting data across multiple servers**

                        **10. What is the purpose of replication in MongoDB?**
                        A) To improve the performance of queries
                        B) To ensure data redundancy and high availability
                        C) To reduce data storage requirements
                        D) To index the database faster
                        **Answer: B) To ensure data redundancy and high availability**

                        **11. How does MongoDB ensure data security?**
                        A) By using firewalls
                        B) Through encryption and access control
                        C) By frequent data backups
                        D) By limiting the database size
                        **Answer: B) Through encryption and access control**

                        **12. What is 'MongoDB Atlas'?**
                        A) A MongoDB IDE
                        B) A MongoDB GUI
                        C) MongoDB's cloud database service
                        D) A data visualization tool for MongoDB
                        **Answer: C) MongoDB's cloud database service**

                        **13. What is a 'Collection' in MongoDB?**
                        A) A type of index
                        B) A group of databases
                        C) A set of MongoDB commands
                        D) A group of documents
                        **Answer: D) A group of documents**

                        **14. What command in MongoDB is used to show all databases?**
                        A) showDatabases()
                        B) showAll()
                        C) db.show()
                        D) show dbs
                        **Answer: D) show dbs**

                        **15. Which feature in MongoDB helps to avoid JavaScript injection attacks?**
                        A) Script scanning
                        B) Field validation
                        C) Query parameterization
                        D) Data type enforcement
                        **Answer: C) Query parameterization**

                        **16. How can you improve query performance in MongoDB?**
                        A) By using larger servers
                        B) By increasing the network bandwidth
                        C) By indexing relevant fields
                        D) By writing shorter queries
                        **Answer: C) By indexing relevant fields**

                        **17. What is a 'Replica Set' in MongoDB?**
                        A) A copy of data for backup
                        B) A group of MongoDB instances that maintain the same data
                        C) A set of replicated queries
                        D) A tool for data replication
                        **Answer: B) A group of MongoDB instances that maintain the same data**

                        **18. What is 'Aggregation' in MongoDB?**
                        A) Combining multiple documents into a single document
                        B) Summarizing data and computing group values
                        C) Increasing the number of documents in a collection
                        D) Distributing data across collections
                        **Answer: B) Summarizing data and computing group values**

                        **19. How is 'Big Data' handled in MongoDB?**
                        A) Through traditional relational database methods
                        B) By limiting the size of collections
                        C) Using features like sharding and replication
                        D) By compressing the data
                        **Answer: C) Using features like sharding and replication**

                        **20. What does 'CRUD' stand for in the context of MongoDB?**
                        A) Create, Read, Update, Delete
                        B) Connect, Retrieve, Utilize, Disconnect
                        C) Copy, Record, Upload, Download
                        D) Compute, Report, Unify, Deploy
                        **Answer: A) Create, Read, Update, Delete**

                        **21. In MongoDB, what is the purpose of the 'findAndModify' method?**
                        A) To search for a document and delete it
                        B) To find a document and update it in a single operation
                        C) To locate and index a document
                        D) To discover and replicate a document
                        **Answer: B) To find a document and update it in a single operation**

                        **22. Which of the following is a valid BSON type in MongoDB?**
                        A) Double
                        B) SmallInt
                        C) VarChar
                        D) Blob
                        **Answer: A) Double**


    Check the mapping table and ensure the number of questions matches before giving the output.
        
        """
    user_message = f"create a markdown file using the provided content. The sections marked by triple hashtags (###), as these indicate the content that needs to be organized.###{Content}###"
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_message.strip()},
            {"role": "user", "content": user_message.strip()}
        ],
        temperature=0
    )
    
    summary = response.choices[0].message.content
    logging.info(50*"#ouput#")
    logging.info(summary)
    file_path="quiz.md"
     # Writing the final string to the quiz.md file
    with open(file_path, 'w') as file:
        file.write(summary)


# Main execution
if __name__ == "__main__":
    remove_old_database_files()
    # Load environment variables from the .env file
    load_dotenv()

    # Define the path of the directory to analyze
    directory_path = r"C:\Users\dsksr\Documents\BIG DATA\2024\Independent Study\QIT-LC\Test"

    # Retrieve the OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
     # Retrieve the OpenAI API key from environment variables
    context_window_size =int(os.getenv('context_window_size'))
    encoding = tiktoken.encoding_for_model("gpt-4")
    persist_directory = 'docs/chroma/'
    try:
        # Analyze the directory and get details of the files present
        file_details = analyze_directory(directory_path)
        # Retrieve the contents of each file from the analyzed directory
        file_contents = get_file_contents(file_details)
        # Process and insert the file contents into the database
        vectordb = process_and_insert_contents(file_contents, persist_directory)
        # Summarize the content of the files using the OpenAI API
        summarized_contents = summarize_files(file_contents)
        chunked_contents = create_chunks_from_content_greedy(summarized_contents,context_window_size)
        learning_outcomes_by_chunk = generate_learning_outcomes_for_chunks(chunked_contents)
        print(learning_outcomes_by_chunk)
        # learning_outcomes=find_most_relevant_learning_outcome_document(vectordb,learning_outcomes_by_chunk)
        # Quetions = format_learning_outcomes_with_identifiers(learning_outcomes)
        # mark_down = generate_markdown_file(Quetions)
        

    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")
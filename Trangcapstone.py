import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.reduction import ReductionSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import time
from tqdm import tqdm

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MAX_ARTICLES = 100  # Adjust based on your needs
SUMMARY_LENGTH = 2500  # Target summary length in words

def fetch_recent_diabetes_articles(days=7):
    """
    Fetch recent diabetes articles from PubMed
    Returns list of articles with title, abstract, and metadata
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for PubMed query
    start_date_str = start_date.strftime("%Y/%m/%d")
    end_date_str = end_date.strftime("%Y/%m/%d")
    
    # Search query for diabetes articles
    query = f"diabetes[Title/Abstract] AND ({start_date_str}[Date - Publication] : {end_date_str}[Date - Publication])"
    
    # Step 1: Search PubMed and get article IDs
    search_url = f"{PUBMED_BASE_URL}esearch.fcgi?db=pubmed&term={query}&retmax={MAX_ARTICLES}&retmode=json"
    response = requests.get(search_url)
    data = response.json()
    
    article_ids = data.get('esearchresult', {}).get('idlist', [])
    if not article_ids:
        print("No articles found for the given query and date range.")
        return []
    
    # Step 2: Fetch details for each article
    fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi?db=pubmed&id={','.join(article_ids)}&retmode=xml"
    response = requests.get(fetch_url)
    root = ET.fromstring(response.content)
    
    articles = []
    for article in root.findall('.//PubmedArticle'):
        try:
            # Extract article metadata
            pmid = article.find('.//PMID').text
            article_title = article.find('.//ArticleTitle').text
            
            # Handle cases where Abstract might be missing or structured differently
            abstract_text = ""
            abstract = article.find('.//AbstractText')
            if abstract is not None:
                if abstract.text:
                    abstract_text = abstract.text
                else:
                    # Handle structured abstracts
                    abstract_sections = article.findall('.//AbstractText')
                    abstract_text = " ".join([section.text for section in abstract_sections if section.text])
            
            # Get authors
            authors = []
            author_list = article.findall('.//Author')
            for author in author_list:
                last_name = author.find('LastName').text if author.find('LastName') is not None else ""
                fore_name = author.find('ForeName').text if author.find('ForeName') is not None else ""
                authors.append(f"{fore_name} {last_name}".strip())
            
            # Get journal info
            journal = article.find('.//Journal/Title').text if article.find('.//Journal/Title') is not None else ""
            pub_date = article.find('.//PubDate')
            year = pub_date.find('Year').text if pub_date is not None and pub_date.find('Year') is not None else ""
            
            articles.append({
                'pmid': pmid,
                'title': article_title,
                'abstract': abstract_text,
                'authors': authors,
                'journal': journal,
                'year': year,
                'citation': generate_citation(authors, year, article_title, journal)
            })
        except Exception as e:
            print(f"Error processing article: {e}")
            continue
    
    return articles

def generate_citation(authors, year, title, journal):
    """
    Generate APA-style citation for an article
    """
    if not authors:
        author_part = "Anonymous"
    elif len(authors) <= 3:
        author_part = ", ".join(authors)
    else:
        author_part = f"{authors[0]} et al."
    
    return f"{author_part} ({year}). {title}. {journal}."

def preprocess_text(text):
    """
    Clean and preprocess text for summarization
    """
    if not text:
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def generate_summary(articles, target_length=2500):
    """
    Generate a comprehensive summary from multiple articles
    """
    # Combine all article texts with their citations
    combined_texts = []
    for article in articles:
        text = f"{article['title']}. {article['abstract']}"
        combined_texts.append((text, article['citation']))
    
    # Initialize summarizers
    stemmer = Stemmer('english')
    summarizers = {
        'lsa': LsaSummarizer(stemmer),
        'lex_rank': LexRankSummarizer(stemmer),
        'luhn': LuhnSummarizer(stemmer),
        'text_rank': TextRankSummarizer(stemmer),
        'kl': KLSummarizer(stemmer),
        'reduction': ReductionSummarizer(stemmer)
    }
    
    # Configure summarizers
    for name, summarizer in summarizers.items():
        summarizer.stop_words = get_stop_words('english')
    
    # Generate summaries with different algorithms
    all_summaries = []
    for text, citation in combined_texts:
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        
        # Generate summary with each algorithm
        summaries = []
        for name, summarizer in summarizers.items():
            summary_sentences = summarizer(parser.document, 3)  # Get top 3 sentences
            summary = ' '.join([str(sentence) for sentence in summary_sentences])
            summaries.append(summary)
        
        # Combine summaries and select the most important parts
        combined_summary = ' '.join(summaries)
        sentences = sent_tokenize(combined_summary)
        unique_sentences = list(dict.fromkeys(sentences))  # Remove duplicates
        article_summary = ' '.join(unique_sentences[:5])  # Take top 5 unique sentences
        
        all_summaries.append((article_summary, citation))
    
    # Combine all article summaries
    full_summary = ""
    current_length = 0
    citations_used = []
    
    for summary, citation in all_summaries:
        if current_length >= target_length:
            break
        
        # Add citation marker
        citation_id = len(citations_used) + 1
        summary_with_citation = f"{summary} [^{citation_id}]"
        
        # Check if adding this would exceed our target
        new_length = current_length + len(word_tokenize(summary_with_citation))
        if new_length > target_length:
            # Trim the summary to fit
            remaining_words = target_length - current_length - 5  # Leave room for citation
            words = word_tokenize(summary)
            trimmed_summary = ' '.join(words[:remaining_words]) + f"... [^{citation_id}]"
            full_summary += trimmed_summary + " "
            citations_used.append(citation)
            break
        else:
            full_summary += summary_with_citation + " "
            citations_used.append(citation)
            current_length = new_length
    
    return full_summary.strip(), citations_used

def generate_word_cloud(articles, filename='wordcloud.png'):
    """
    Generate a word cloud from article texts
    """
    all_text = ' '.join([article['title'] + ' ' + article['abstract'] for article in articles])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_summary_to_file(summary_text, citations, filename='diabetes_research_summary.md'):
    """
    Save the summary to a Markdown file with proper citations
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Weekly Diabetes Research Summary\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Number of articles summarized:** {len(citations)}\n\n")
        
        f.write("## Summary\n\n")
        f.write(summary_text + "\n\n")
        
        f.write("## References\n\n")
        for i, citation in enumerate(citations, 1):
            f.write(f"[^{i}]: {citation}\n")
        
        f.write("\n---\n")
        f.write("This summary was automatically generated using natural language processing techniques.")

def main():
    print("Starting diabetes research summarization project...")
    
    # Step 1: Fetch recent diabetes articles
    print("Fetching recent diabetes articles from PubMed...")
    articles = fetch_recent_diabetes_articles(days=7)
    
    if not articles:
        print("No articles found. Exiting.")
        return
    
    print(f"Found {len(articles)} articles about diabetes from the past week.")
    print(articles)

    for article in articles:
        print(article['title'])
    
    # Step 2: Generate word cloud visualization
    print("Generating word cloud...")
    generate_word_cloud(articles)
    
    # Step 3: Generate comprehensive summary
    print("Generating research summary...")
    summary_text, citations = generate_summary(articles, target_length=SUMMARY_LENGTH)
    
    # Step 4: Save results
    print("Saving results...")
    save_summary_to_file(summary_text, citations)
    
    print("\nSummary generation complete!")
    print(f"Final summary contains approximately {len(word_tokenize(summary_text))} words.")
    print(f"Summary saved to 'diabetes_research_summary.md'")
    print(f"Word cloud visualization saved to 'wordcloud.png'")

if __name__ == "__main__":
    main()
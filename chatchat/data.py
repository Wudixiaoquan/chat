import os
import time
import requests
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import re
import logging
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import random


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

 #API configuration
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

# def __init__(self, max_pages: int = 200, max_depth: int = 3):
#     self.session = requests.Session()
#     self.session.headers.update({
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#         'Accept-Language': 'en-US,en;q=0.5',
#         'Accept-Encoding': 'gzip, deflate, br',
#         'Connection': 'keep-alive',
#         'Upgrade-Insecure-Requests': '1',
#     })
#     self.max_pages = max_pages
#     self.visited_urls: Set[str] = set()
#     self.base_domain = "dbs.com.sg"
#     self.max_depth = max_depth
#     self.to_crawl = deque()
class RealDBSBankCrawler:
    def __init__(self, max_pages: int = 200):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.base_domain = "dbs.com.sg"

    def get_with_proxy(self, url: str) -> Optional[requests.Response]:
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response

            time.sleep(2)
            return None

        except Exception as e:
            logger.warning(f"Request failed {url}: {e}")
            return None

    def extract_real_content(self, soup: BeautifulSoup) -> str:

        try:

            body = soup.find('body')
            if body:
                text = body.get_text()
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 500:  
                    return text

            
            visible_texts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'li', 'td']):
                text = element.get_text().strip()
                if text and len(text) > 10:
                    visible_texts.append(text)

            content = " ".join(visible_texts)
            if len(content) > 300:
                return content

            
            full_text = soup.get_text()
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            return full_text

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return ""

    # def crawl_real_pages(self) -> List[Document]:
    #     seed_urls = ["https://www.dbs.com.sg", "https://www.dbs.com.sg/personal"]
    #     for url in seed_urls:
    #         self.to_crawl.append((url, 0))  # (url, depth)
    #
    #     documents = []
    #
    #     while self.to_crawl and len(documents) < self.max_pages:
    #         current_url, depth = self.to_crawl.popleft()
    #
    #         if depth < self.max_depth:
    #             new_links = self.extract_links(soup, current_url)
    #             for link in new_links:
    #                 if link not in self.visited_urls:
    #                     self.to_crawl.append((link, depth + 1))

    def crawl_real_pages(self) -> List[Document]:

        logger.info("Starting to crawl DBS real content pages...")

        
        content_urls = [
            
            "https://www.dbs.com.sg/personal/cards/credit-cards/dbs-visa-debit-card",
            "https://www.dbs.com.sg/personal/cards/credit-cards/dbs-yuu-card",
            "https://www.dbs.com.sg/personal/deposits/savings-accounts/multi-currency-autosave",
            "https://www.dbs.com.sg/personal/loans/personal-loans/personal-loan",
            "https://www.dbs.com.sg/personal/insurance/life-insurance/term-life",
            "https://www.dbs.com.sg/personal/investments/unit-trusts/equity-funds",

            
            "https://www.dbs.com.sg/support/faq.html",
            "https://www.dbs.com.sg/support/contact-us.html",
            "https://www.dbs.com.sg/support/digital-banking.html",

            
            "https://www.dbs.com.sg/rates/personal-banking-rates.html",
            "https://www.dbs.com.sg/fees/personal-banking-fees.html",

            
            "https://www.dbs.com.sg/wealth/insights",
            "https://www.dbs.com.sg/personal/articles",
            
            "https://www.dbs.com.sg",
            "https://www.dbs.com.sg/personal",
            "https://www.dbs.com.sg/business",
            "https://www.dbs.com.sg/wealth",
            "https://www.dbs.com.sg/corporate",

            
            "https://www.dbs.com.sg/personal/cards",
            "https://www.dbs.com.sg/personal/cards/credit-cards",
            "https://www.dbs.com.sg/personal/cards/debit-cards",
            "https://www.dbs.com.sg/personal/cards/prepaid-cards",

            
            "https://www.dbs.com.sg/personal/deposits",
            "https://www.dbs.com.sg/personal/deposits/savings-accounts",
            "https://www.dbs.com.sg/personal/deposits/current-accounts",
            "https://www.dbs.com.sg/personal/deposits/fixed-deposits",
        ]

        documents = []

        for url in content_urls:
            if len(documents) >= self.max_pages:
                break

            logger.info(f"Attempting to crawl: {url}")

            try:
                response = self.get_with_proxy(url)
                if not response:
                    continue

                soup = BeautifulSoup(response.content, 'html.parser')

                
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "DBS Content"

                
                content = self.extract_real_content(soup)

                if content and len(content) > 200:  
                    full_content = f"Title: {title_text}\nURL: {url}\nContent:\n{content}"

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            'source_type': 'dbs_website',
                            'source_file': url,
                            'title': title_text,
                            'url': url,
                            'page_type': self.classify_page_type(url, title_text, content),
                            'content_length': len(content),
                            'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    )

                    documents.append(doc)
                    logger.info(f"✅ Successfully obtained: {title_text} ({len(content)} characters)")
                else:
                    logger.warning(f"Insufficient content: {url} - {len(content) if content else 0} characters")

                time.sleep(1)  

            except Exception as e:
                logger.error(f"Crawling failed {url}: {e}")
                continue

        
        if len(documents) < 10:
            logger.info("Attempting to crawl main entry pages...")
            fallback_urls = [
                "https://www.dbs.com.sg",
                "https://www.dbs.com.sg/personal",
                "https://www.dbs.com.sg/business",
                "https://www.dbs.com.sg/wealth",
            ]

            for url in fallback_urls:
                if len(documents) >= self.max_pages:
                    break

                try:
                    response = self.session.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "DBS Main Page"

                    content = self.extract_real_content(soup)

                    if content and len(content) > 100:
                        full_content = f"Title: {title_text}\nURL: {url}\nContent:\n{content}"

                        doc = Document(
                            page_content=full_content,
                            metadata={
                                'source_type': 'dbs_website',
                                'source_file': url,
                                'title': title_text,
                                'url': url,
                                'page_type': 'Main Page',
                                'content_length': len(content),
                                'crawl_time': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        )

                        documents.append(doc)
                        logger.info(f"✅ Obtained main page: {title_text}")

                except Exception as e:
                    logger.error(f"Failed to get main page {url}: {e}")

        logger.info(f"✅ Crawling completed! Obtained {len(documents)} real documents")
        return documents

    # def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
    #     links = []
    #     for link in soup.find_all('a', href=True):
    #         href = link['href']
    #         if href and not href.startswith(('javascript:', 'mailto:')):
    #             full_url = urljoin(current_url, href)
    #             if self.base_domain in full_url:
    #                 links.append(full_url)
    #     return links
    #
    # def is_valid_content_page(self, url: str, content: str) -> bool:
    #     invalid_patterns = [r'\.pdf$', r'/login', r'/search']
    #     for pattern in invalid_patterns:
    #         if re.search(pattern, url):
    #             return False
    #     return len(content) > 200

    def classify_page_type(self, url: str, title: str, content: str) -> str:

        combined_text = url.lower() + " " + title.lower() + " " + content.lower()

        if any(word in combined_text for word in ['card', 'credit', 'visa', 'mastercard']):
            return 'Credit Card'
        elif any(word in combined_text for word in ['deposit', 'saving', 'account', 'current']):
            return 'Deposit Account'
        elif any(word in combined_text for word in ['loan', 'borrow', 'mortgage']):
            return 'Loan'
        elif any(word in combined_text for word in ['insurance', 'protection']):
            return 'Insurance'
        elif any(word in combined_text for word in ['investment', 'invest', 'wealth', 'fund']):
            return 'Investment'
        elif any(word in combined_text for word in ['digibank', 'digital', 'online']):
            return 'Digital Banking'
        elif any(word in combined_text for word in ['promotion', 'offer', 'deal']):
            return 'Promotion'
        else:
            return 'Other'


class HighVolumeQAGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=1200,
            timeout=120,
            max_retries=5,
        )

    def chunk_content_aggressive(self, content: str, chunk_size: int = 1500) -> List[str]:
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        sentences = re.split(r'[.!?。！？]', content)

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def generate_high_volume_qa(self, document: Document) -> List[Dict[str, str]]:

        try:
            content = document.page_content
            metadata = document.metadata
            page_type = metadata.get('page_type', 'Other')

            
            content_chunks = self.chunk_content_aggressive(content, 800)
            all_qa_pairs = []

            logger.info(f"Processing document: {metadata.get('title', 'Unknown')} - {len(content_chunks)} content chunks")

            for i, chunk in enumerate(content_chunks):
                
                chunk_length = len(chunk)
                if chunk_length > 800:
                    num_pairs = 10
                elif chunk_length > 500:
                    num_pairs = 8
                else:
                    num_pairs = 4

                prompt = f"""
Based on the following DBS Bank {page_type} related content, generate {num_pairs} detailed and practical QA pairs.

Page Information:
Title: {metadata.get('title', 'Unknown')}
Type: {page_type}

Requirements:
1. Generate {num_pairs} QA pairs
2. Question types include:
   - Product features and usage methods
   - Application conditions and requirements
   - Fees, interest rates, and charges
   - Operation steps and processes
   - Advantages and limitations
3. Answers should be detailed and specific, based on provided content
4. Each answer 150-300 words
5. Format:
Q1: Question
A1: Answer

Q2: Question  
A2: Answer

Content:
{chunk}

Please generate {num_pairs} high-quality QA pairs:
"""

                try:
                    response = self.llm.invoke(prompt)
                    qa_text = response.content

                    chunk_qa_pairs = self.parse_qa_pairs(qa_text, metadata)
                    all_qa_pairs.extend(chunk_qa_pairs)

                    logger.info(f"Content chunk {i + 1} generated {len(chunk_qa_pairs)} QA pairs")

                    
                    if i < len(content_chunks) - 1:
                        time.sleep(5)

                except Exception as e:
                    logger.error(f"Failed to generate QA pairs for chunk {i + 1}: {e}")
                    continue

            logger.info(f"Document total generated {len(all_qa_pairs)} QA pairs")
            return all_qa_pairs

        except Exception as e:
            logger.error(f"Failed to generate QA pairs: {e}")
            return []

    def parse_qa_pairs(self, qa_text: str, metadata: Dict) -> List[Dict[str, str]]:
        qa_pairs = []

        
        patterns = [
            r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)',
            r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)',
            r'(\d+)\.\s*Q:\s*(.*?)\s*A:\s*(.*?)(?=\d+\.|$)',
            r'Question\s*(\d+):\s*(.*?)\s*Answer\s*\1:\s*(.*?)(?=Question\d+:|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, qa_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    question = match[1] if len(match) > 2 else match[0]
                    answer = match[2] if len(match) > 2 else match[1]

                    question = question.strip()
                    answer = answer.strip()

                    
                    if (len(question) > 5 and len(answer) > 20 and
                            'Cannot answer' not in answer and 'Insufficient information' not in answer):
                        qa_pairs.append({
                            'question': question,
                            'answer': answer,
                            'source': metadata.get('url', 'Unknown'),
                            'title': metadata.get('title', 'Unknown'),
                            'page_type': metadata.get('page_type', 'Unknown'),
                            'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'content_length': len(answer)
                        })

        
        if not qa_pairs:
            lines = qa_text.split('\n')
            current_q = None

            for line in lines:
                line = line.strip()
                if re.match(r'^(Q\d*|Question\d*)[:：]', line):
                    if current_q and current_q.get('answer') and len(current_q['answer']) > 20:
                        qa_pairs.append(current_q)

                    question = re.sub(r'^(Q\d*|Question\d*)[:：]\s*', '', line)
                    current_q = {
                        'question': question,
                        'answer': '',
                        'source': metadata.get('url', 'Unknown'),
                        'title': metadata.get('title', 'Unknown'),
                        'page_type': metadata.get('page_type', 'Unknown'),
                        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'content_length': 0
                    }
                elif re.match(r'^(A\d*|Answer\d*)[:：]', line) and current_q:
                    answer = re.sub(r'^(A\d*|Answer\d*)[:：]\s*', '', line)
                    current_q['answer'] = answer
                    current_q['content_length'] = len(answer)
                elif current_q and current_q['answer'] and line:
                    current_q['answer'] += ' ' + line
                    current_q['content_length'] = len(current_q['answer'])

            if current_q and current_q.get('answer') and len(current_q['answer']) > 20:
                qa_pairs.append(current_q)

        return qa_pairs

    def generate_massive_dataset(self, documents: List[Document]) -> List[Dict[str, str]]:

        all_qa_pairs = []

        logger.info(f"Starting to generate massive QA pairs for {len(documents)} documents")

        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}: {doc.metadata.get('title', 'Unknown')}")

            try:
                qa_pairs = self.generate_high_volume_qa(doc)

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    logger.info(f"  ✅ Successfully generated {len(qa_pairs)} QA pairs")
                else:
                    logger.warning(f"  ⚠️ No QA pairs generated")

            except Exception as e:
                logger.error(f"  ❌ Failed to process document: {e}")

            
            if i < len(documents):
                time.sleep(8)

        logger.info(f"✅ Generation completed! Total {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs


def save_real_qa_data(qa_pairs: List[Dict[str, str]]):

    output_dir = "D:/chatchat/real_dbs_qa"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    
    full_path = os.path.join(output_dir, f"dbs_qa_real_{timestamp}.json")
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    
    stats = {
        'total_qa_pairs': len(qa_pairs),
        'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'average_answer_length': sum(len(qa.get('answer', '')) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
        'sources': list(set(qa['source'] for qa in qa_pairs))
    }

    stats_path = os.path.join(output_dir, f"stats_real_{timestamp}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Real data saved to: {output_dir}")
    logger.info(f"📊 Dataset statistics: {len(qa_pairs)} QA pairs")

    return full_path


def main():
    """Main function - Only use real crawler data"""
    print("=== DBS Bank Real Data QA Pair Generator ===")
    print("Target: Generate QA pairs based on real website data")
    print("=" * 50)

    try:
        
        print("🚀 Phase 1: Crawling DBS real content...")
        # Exploratory crawling
        # crawler = ExploratoryDBSBankCrawler(max_pages=100)
        # documents = crawler.explore_from_seed("https://www.dbs.com.sg")
        crawler = RealDBSBankCrawler(max_pages=100)
        documents = crawler.crawl_real_pages()
        # documents = documents[:2]
        if not documents:
            print("❌ Unable to obtain any real data, please check network or website access")
            return

        print(f"✅ Crawling completed, obtained {len(documents)} real documents")

        
        print("📊 Document details:")
        for doc in documents:
            print(f"  - {doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('content_length', 0)} characters)")

        
        print(f"\n🚀 Phase 2: Generating massive QA pairs...")
        qa_generator = HighVolumeQAGenerator()
        qa_pairs = qa_generator.generate_massive_dataset(documents)

        if not qa_pairs:
            print("❌ QA pair generation failed")
            return

        print(f"✅ Generation completed, obtained {len(qa_pairs)} real QA pairs")

        
        print("\n🚀 Phase 3: Saving real data...")
        save_path = save_real_qa_data(qa_pairs)

        print(f"✅ All tasks completed!")
        print(f"📁 Real data saved at: {save_path}")
        print(f"📊 Final statistics: {len(qa_pairs)} QA pairs based on real website")

        
        sources = set(qa['source'] for qa in qa_pairs)
        print(f"🌐 Data sources: {len(sources)} different pages")

    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
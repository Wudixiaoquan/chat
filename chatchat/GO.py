import os
import time
import requests
import json
import re
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''


class RealDBSBankCrawler:
    """Real DBS Bank Crawler - Focused on obtaining real content"""

    def __init__(self, max_pages: int = 200):
        self.session = requests.Session()
        # Use more realistic headers
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
        """Get page using proxy"""
        try:
            # Try direct access
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response

            # If restricted, try different methods
            time.sleep(2)
            return None

        except Exception as e:
            logger.warning(f"Request failed {url}: {e}")
            return None

    def extract_real_content(self, soup: BeautifulSoup) -> str:
        """Completely rewritten content extraction method"""
        try:
            # Method 1: Get all body text directly
            body = soup.find('body')
            if body:
                text = body.get_text()
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 500:  # Significantly lower requirement
                    return text

            # Method 2: Get all visible text
            visible_texts = []
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'li', 'td']):
                text = element.get_text().strip()
                if text and len(text) > 10:
                    visible_texts.append(text)

            content = " ".join(visible_texts)
            if len(content) > 300:
                return content

            # Method 3: Last attempt - get entire page text
            full_text = soup.get_text()
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            return full_text

        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return ""



    def crawl_real_pages(self) -> List[Document]:
        """Crawl real pages - using known content pages"""
        logger.info("Starting to crawl DBS real content pages...")

        # Use known URLs containing real content
        content_urls = [
            # Product detail pages - more likely to contain real content
            "https://www.dbs.com.sg/personal/cards/credit-cards/dbs-visa-debit-card",
            "https://www.dbs.com.sg/personal/cards/credit-cards/dbs-yuu-card",
            "https://www.dbs.com.sg/personal/deposits/savings-accounts/multi-currency-autosave",
            "https://www.dbs.com.sg/personal/loans/personal-loans/personal-loan",
            "https://www.dbs.com.sg/personal/insurance/life-insurance/term-life",
            "https://www.dbs.com.sg/personal/investments/unit-trusts/equity-funds",

            # Help and support pages
            "https://www.dbs.com.sg/support/faq.html",
            "https://www.dbs.com.sg/support/contact-us.html",
            "https://www.dbs.com.sg/support/digital-banking.html",

            # Interest rates and fees pages
            "https://www.dbs.com.sg/rates/personal-banking-rates.html",
            "https://www.dbs.com.sg/fees/personal-banking-fees.html",

            # Blog and article pages
            "https://www.dbs.com.sg/wealth/insights",
            "https://www.dbs.com.sg/personal/articles",
            # Main entry pages
            "https://www.dbs.com.sg",
            "https://www.dbs.com.sg/personal",
            "https://www.dbs.com.sg/business",
            "https://www.dbs.com.sg/wealth",
            "https://www.dbs.com.sg/corporate",

            # Credit card related - add more variants
            "https://www.dbs.com.sg/personal/cards",
            "https://www.dbs.com.sg/personal/cards/credit-cards",
            "https://www.dbs.com.sg/personal/cards/debit-cards",
            "https://www.dbs.com.sg/personal/cards/prepaid-cards",

            # Deposit accounts - add more types
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

                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "DBS Content"

                # Extract content
                content = self.extract_real_content(soup)

                if content and len(content) > 200:  # Lower length requirement
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

                time.sleep(1)  # Polite delay

            except Exception as e:
                logger.error(f"Crawling failed {url}: {e}")
                continue

        # If the above URLs don't work, try crawling sitemap or main pages
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

    def classify_page_type(self, url: str, title: str, content: str) -> str:
        """Page type classification"""
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
    """High Volume QA Pair Generator - Restoring original generation capability"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=1200,
            timeout=120,
            max_retries=5,
        )

    def chunk_content_aggressive(self, content: str, chunk_size: int = 1500) -> List[str]:
        """Aggressive content chunking - Ensure generating more QA pairs"""
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
        """Generate high volume QA pairs"""
        try:
            content = document.page_content
            metadata = document.metadata
            page_type = metadata.get('page_type', 'Other')

            # Aggressive chunking, generate more content chunks
            content_chunks = self.chunk_content_aggressive(content, 800)
            all_qa_pairs = []

            logger.info(
                f"Processing document: {metadata.get('title', 'Unknown')} - {len(content_chunks)} content chunks")

            for i, chunk in enumerate(content_chunks):
                # Determine generation quantity based on content length
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

                    # Delay between chunks
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
        """Parse QA pairs - Lenient parsing"""
        qa_pairs = []

        # Multiple format support
        patterns = [
            r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)',
            r'Q:\s*(.*?)\s*A:\s*(.*?)(?=Q:|$)',
            r'(\d+)\.\s*Q:\s*(.*?)\s*A:\s*(json?)(?=\d+\.|$)',
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

                    # Lenient quality check
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

        # Alternative parsing
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
        """Generate massive dataset"""
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

            # Delay between documents
            if i < len(documents):
                time.sleep(8)

        logger.info(f"✅ Generation completed! Total {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs


class EnhancedQASearcher:
    """Enhanced QA Searcher - More precise matching"""

    def __init__(self, qa_pairs: List[Dict[str, str]]):
        self.qa_pairs = qa_pairs

        # More detailed keyword weights
        self.bank_keywords = {
            # Credit card related
            'credit card': 3.0, 'annual fee': 2.0, 'credit limit': 2.0, 'points': 1.8,
            'promotion': 1.8, 'application conditions': 2.5, 'application requirements': 2.5,
            'eligibility': 2.0, 'income requirement': 2.0,
            'age requirement': 2.0, 'application process': 2.0, 'apply': 2.0, 'activate': 1.8,

            # Account related
            'deposit account': 2.5, 'savings account': 2.5, 'open account': 2.0, 'account': 1.8,
            'current account': 2.0,

            # Loan related
            'loan': 2.5, 'interest rate': 2.0, 'mortgage': 2.2, 'car loan': 2.2,
            'personal loan': 2.0, 'application conditions': 2.0,

            # Investment
            'investment': 2.0, 'wealth management': 2.0, 'fund': 1.8, 'stock': 1.8, 'investment product': 2.0,

            # Banking services
            'online banking': 1.8, 'digibank': 1.8, 'transfer': 1.5, 'payment': 1.5,

            # Brands
            'DBS': 1.5, 'POSB': 1.3
        }

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search related QA pairs - Enhanced version"""
        scores = []
        query_lower = query.lower()

        for i, qa in enumerate(self.qa_pairs):
            score = self.calculate_enhanced_similarity(query_lower, qa)
            if score > 0.5:  # Increase threshold, only return highly relevant results
                scores.append((i, score, qa))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return results
        results = []
        for idx, score, qa in scores[:top_k]:
            results.append({
                'index': idx,
                'score': score,
                'question': qa['question'],
                'answer': qa['answer'],
                'page_type': qa.get('page_type', 'Unknown'),
                'source': qa.get('source', 'Unknown')
            })

        return results

    def calculate_enhanced_similarity(self, query: str, qa: Dict[str, str]) -> float:
        """Enhanced similarity calculation"""
        score = 0.0
        question = qa['question'].lower()
        answer = qa['answer'].lower()
        content = question + " " + answer

        # 1. Exact question match (highest weight)
        if query in question:
            score += 5.0

        # 2. Answer contains query keywords
        if query in answer:
            score += 3.0

        # 3. Keyword matching
        for keyword, weight in self.bank_keywords.items():
            keyword_lower = keyword.lower()
            if keyword_lower in query and keyword_lower in content:
                # Both query and content contain keyword, highest weight
                score += weight * 2
            elif keyword_lower in content:
                # Only content contains keyword
                score += weight * 0.5

        # 4. Specific question type matching
        if self.is_application_question(query) and self.contains_application_info(content):
            score += 3.0

        if self.is_fee_question(query) and self.contains_fee_info(content):
            score += 2.5

        if self.is_requirement_question(query) and self.contains_requirement_info(content):
            score += 3.0

        # 5. Answer quality scoring
        if len(qa['answer']) > 100:  # Longer answer, possibly more detailed
            score += 1.0

        if any(marker in qa['answer'] for marker in ['condition', 'requirement', 'need', 'must', 'step', 'process']):
            score += 1.5

        return score

    def is_application_question(self, query: str) -> bool:
        """Determine if it's an application question"""
        application_words = ['apply', 'application', 'how to', 'process', 'condition', 'requirement', 'need']
        return any(word in query for word in application_words)

    def is_fee_question(self, query: str) -> bool:
        """Determine if it's a fee question"""
        fee_words = ['fee', 'annual fee', 'service charge', 'interest rate', 'charge']
        return any(word in query for word in fee_words)

    def is_requirement_question(self, query: str) -> bool:
        """Determine if it's a requirement question"""
        requirement_words = ['condition', 'requirement', 'eligibility', 'what do I need', 'what is required']
        return any(word in query for word in requirement_words)

    def contains_application_info(self, content: str) -> bool:
        """Whether content contains application information"""
        application_indicators = ['apply', 'application', 'step', 'process', 'submit', 'material', 'document']
        return any(indicator in content for indicator in application_indicators)

    def contains_fee_info(self, content: str) -> bool:
        """Whether content contains fee information"""
        fee_indicators = ['fee', 'annual fee', 'free', 'charge', 'interest rate', '%']
        return any(indicator in content for indicator in fee_indicators)

    def contains_requirement_info(self, content: str) -> bool:
        """Whether content contains requirement information"""
        requirement_indicators = ['condition', 'requirement', 'must', 'need', 'age', 'income', 'document']
        return any(indicator in content for indicator in requirement_indicators)


class SmartDBSBankChatBot:
    """Smart DBS Bank ChatBot"""

    def __init__(self, qa_data_dir: str = "D:/chatchat/real_dbs_qa"):
        self.qa_data_dir = qa_data_dir
        self.qa_pairs = self.load_or_generate_qa_data()

        if not self.qa_pairs:
            raise ValueError("Unable to load or generate QA pair data")

        self.searcher = EnhancedQASearcher(self.qa_pairs)
        self.llm = self.setup_llm()

        logger.info(f"✅ Chatbot initialization completed, loaded {len(self.qa_pairs)} QA pairs")

    def load_or_generate_qa_data(self) -> List[Dict[str, str]]:
        """Load existing QA data or generate new ones"""
        # Try to find existing QA files
        existing_files = self.find_existing_qa_files()

        if existing_files:
            logger.info(f"Found existing QA files: {existing_files}")
            # Load the most recent file
            latest_file = max(existing_files, key=os.path.getctime)
            return self.load_qa_data(latest_file)
        else:
            logger.info("No existing QA files found, generating new QA pairs...")
            return self.generate_new_qa_data()

    def find_existing_qa_files(self) -> List[str]:
        """Find existing QA data files"""
        if not os.path.exists(self.qa_data_dir):
            return []

        qa_files = []
        for file in os.listdir(self.qa_data_dir):
            if file.startswith("dbs_qa_real_") and file.endswith(".json"):
                qa_files.append(os.path.join(self.qa_data_dir, file))

        return qa_files

    def load_qa_data(self, file_path: str) -> List[Dict[str, str]]:
        """Load QA pair data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            logger.info(f"Successfully loaded {len(qa_pairs)} QA pairs from {file_path}")
            return qa_pairs
        except Exception as e:
            logger.error(f"Failed to load QA pair data from {file_path}: {e}")
            return []

    def generate_new_qa_data(self) -> List[Dict[str, str]]:
        """Generate new QA data by crawling and processing"""
        # Create output directory if it doesn't exist
        os.makedirs(self.qa_data_dir, exist_ok=True)

        # 1. Crawl real pages
        logger.info("Starting to crawl DBS real content...")
        crawler = RealDBSBankCrawler(max_pages=50)
        documents = crawler.crawl_real_pages()

        if not documents:
            logger.error("Unable to obtain any real data")
            return []

        logger.info(f"Crawling completed, obtained {len(documents)} real documents")

        # 2. Generate massive QA pairs
        logger.info("Generating massive QA pairs...")
        qa_generator = HighVolumeQAGenerator()
        qa_pairs = qa_generator.generate_massive_dataset(documents)

        if not qa_pairs:
            logger.error("QA pair generation failed")
            return []

        # 3. Save the generated data
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.qa_data_dir, f"dbs_qa_real_{timestamp}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ New QA data saved to: {output_path}")
        logger.info(f"📊 Generated {len(qa_pairs)} QA pairs")

        return qa_pairs

    def setup_llm(self) -> ChatOpenAI:
        """Setup LLM"""
        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=1000,
            timeout=30,
            max_retries=1,
        )

    def chat(self, question: str) -> Dict[str, Any]:
        """Process user question - Enhanced version"""
        try:
            logger.info(f"User question: {question}")

            # Search related QA pairs
            search_results = self.searcher.search(question, top_k=5)

            # If no relevant results found, return friendly prompt directly
            if not search_results:
                return self.get_fallback_response(question)

            # Build more detailed context
            context = self.build_detailed_context(search_results, question)

            # Build enhanced prompt
            prompt = self.build_enhanced_prompt(question, context, search_results)

            # Call LLM
            response = self.llm.invoke(prompt)
            answer = response.content

            # Post-process answer
            answer = self.post_process_answer(answer, question)

            # Build response
            response_data = {
                "answer": answer,
                "source_documents": [
                    {
                        "question": result["question"],
                        "answer_preview": result["answer"][:100] + "...",
                        "page_type": result["page_type"],
                        "similarity_score": round(result["score"], 2)
                    }
                    for result in search_results[:3]
                ],
                "confidence": self.calculate_enhanced_confidence(question, answer, search_results)
            }

            logger.info(f"Generated answer, confidence: {response_data['confidence']:.2f}")
            return response_data

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return self.get_error_response()

    def build_detailed_context(self, search_results: List[Dict], question: str) -> str:
        """Build detailed context"""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            context_parts.append(f"【Reference Information {i}】")
            context_parts.append(f"Question: {result['question']}")
            context_parts.append(f"Answer: {result['answer']}")
            context_parts.append(f"Type: {result['page_type']}")
            context_parts.append(f"Relevance: {result['score']:.2f}")
            context_parts.append("")

        return "\n".join(context_parts)

    def build_enhanced_prompt(self, question: str, context: str, search_results: List[Dict]) -> str:
        """Build enhanced prompt"""
        question_type = self.analyze_question_type(question)

        prompt = f"""You are a professional DBS Bank customer service expert. Please provide accurate, detailed, and practical answers based on the following reference information.

User Question: {question}
Question Type: {question_type}

Reference Information:
{context}

Please answer according to the following requirements:
1. 🎯 **Accuracy**: Strictly based on reference information, do not fabricate non-existent information
2. 📝 **Detail**: If there are specific conditions, steps, numbers in the reference information, please list them in detail
3. 💡 **Practicality**: Provide actionable advice, don't just say "please contact customer service"
4. 🗣️ **Friendliness**: Professional but friendly tone, answer in English

If the reference information is detailed enough:
- Please directly provide specific conditions, steps, fees and other information
- Can appropriately summarize and organize to make information clearer

If the reference information is not detailed enough:
- Honestly inform which information is incomplete
- Give best suggestions based on existing information
- Can suggest how users can get more detailed information

Now please answer the user's question:"""

        return prompt

    def analyze_question_type(self, question: str) -> str:
        """Analyze question type"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['apply', 'application', 'how to', 'process']):
            return "Application Type"
        elif any(word in question_lower for word in ['condition', 'requirement', 'eligibility', 'what do I need']):
            return "Requirement Type"
        elif any(word in question_lower for word in ['fee', 'annual fee', 'service charge', 'interest rate']):
            return "Fee and Rate Type"
        elif any(word in question_lower for word in ['promotion', 'offer', 'deal']):
            return "Promotion Type"
        elif any(word in question_lower for word in ['step', 'process', 'operation']):
            return "Operation Process Type"
        else:
            return "General Inquiry Type"

    def post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer"""
        lines = answer.split('\n')
        unique_lines = []
        seen_lines = set()

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped not in seen_lines:
                unique_lines.append(line)
                seen_lines.add(line_stripped)

        answer = '\n'.join(unique_lines)

        if not any(phrase in answer for phrase in ['Wish you', 'Thank you', 'Hope', 'If you have']):
            answer += "\n\nIf you have other questions, feel free to ask anytime!"

        return answer

    def calculate_enhanced_confidence(self, question: str, answer: str, search_results: List[Dict]) -> float:
        """Enhanced confidence calculation"""
        if not search_results:
            return 0.0

        confidence = 0.3

        max_score = max([result['score'] for result in search_results])
        confidence += min(max_score * 0.15, 0.45)

        if len(answer) > 150:
            confidence += 0.2

        specific_indicators = ['age', 'income', 'document', 'material', 'step', 'process', 'fee', 'interest rate', '%']
        if any(indicator in answer for indicator in specific_indicators):
            confidence += 0.2

        if any(phrase in answer for phrase in
               ['cannot answer', 'no information', 'insufficient information', 'please contact customer service']):
            confidence -= 0.3

        return min(max(confidence, 0.0), 1.0)

    def get_fallback_response(self, question: str) -> Dict[str, Any]:
        """Get fallback response"""
        fallback_answers = {
            'application': "Regarding specific conditions for applying for DBS credit cards, they usually include age requirements, income proof, credit history, etc. We recommend visiting the DBS official website to check the latest application requirements or calling customer service for consultation.",
            'fee': "Credit card fee information varies by card type, including annual fees, interest, cash withdrawal fees, etc. Please check the specific product description on the DBS official website for accurate fee information.",
            'general': "Thank you for your inquiry! Since the related information is quite specific, we recommend visiting the DBS official website or contacting customer service for the most accurate information."
        }

        question_lower = question.lower()
        if any(word in question_lower for word in ['apply', 'application', 'condition']):
            answer = fallback_answers['application']
        elif any(word in question_lower for word in ['fee', 'annual fee', 'service charge']):
            answer = fallback_answers['fee']
        else:
            answer = fallback_answers['general']

        return {
            "answer": answer,
            "source_documents": [],
            "confidence": 0.3
        }

    def get_error_response(self) -> Dict[str, Any]:
        """Get error response"""
        return {
            "answer": "Sorry, the system is temporarily unable to process your request. Please try again later or contact DBS official customer service for assistance.",
            "source_documents": [],
            "confidence": 0.0
        }


class InteractiveChat:
    """Interactive chat interface"""

    def __init__(self, chatbot: SmartDBSBankChatBot):
        self.chatbot = chatbot
        self.chat_history = []

    def start_chat(self):
        """Start interactive chat"""
        print("=== DBS Bank Smart Customer Service ===")
        print("=" * 50)
        print("Welcome to DBS Bank Smart Customer Service!")
        print("I can help you answer the following questions:")
        print("  💳 Credit card application conditions, fees, promotions")
        print("  🏦 Account opening, deposit products")
        print("  📈 Loans, investment and wealth management")
        print("  🌐 Online banking services")
        print("\nPlease enter your question (enter 'exit' to end conversation)")
        print("=" * 50)

        while True:
            try:
                question = input("\n👤 You: ").strip()

                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nThank you for using DBS Bank Smart Customer Service, goodbye!")
                    break

                if not question:
                    print("Please enter a valid question")
                    continue

                print("🤖 Thinking...")
                start_time = time.time()
                response = self.chatbot.chat(question)
                response_time = time.time() - start_time

                self.chat_history.append({
                    "question": question,
                    "answer": response["answer"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                print(f"\n🤖 DBS Customer Service: {response['answer']}")
                print(f"\n⏱️ Response time: {response_time:.2f} seconds | 📊 Confidence: {response['confidence']:.2f}")

                if response['source_documents'] and response['confidence'] > 0.5:
                    print(f"\n📚 Reference sources:")
                    for i, source in enumerate(response['source_documents'][:2], 1):
                        print(f"  {i}. {source['question']}")

                if response['confidence'] < 0.4:
                    print(
                        f"\n💡 Tip: For more accurate information, we recommend visiting the DBS official website or calling customer service.")

            except KeyboardInterrupt:
                print("\n\nThank you for using DBS Bank Smart Customer Service, goodbye!")
                break
            except Exception as e:
                print(f"\nSorry, an error occurred: {e}")
                print("Please ask again or contact human customer service.")


def main():
    """Main function"""
    print("=== DBS Bank Smart Customer Service System ===")
    print("This system will:")
    print("1. Check for existing QA data")
    print("2. Generate new QA pairs if needed")
    print("3. Start interactive chat")
    print("=" * 50)

    try:
        # Initialize chatbot (this will automatically handle QA data loading/generation)
        print("Initializing system...")
        chatbot = SmartDBSBankChatBot()

        if not chatbot.qa_pairs:
            print("❌ Unable to load or generate QA pair data")
            return

        print(f"✅ System initialization completed! Loaded {len(chatbot.qa_pairs)} QA pairs")

        # Start interactive chat
        print("\nEntering interactive chat mode...")
        interactive_chat = InteractiveChat(chatbot)
        interactive_chat.start_chat()

    except Exception as e:
        print(f"System startup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

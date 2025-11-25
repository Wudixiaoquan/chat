import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''


class EnhancedQASearcher:
    def __init__(self, qa_pairs: List[Dict[str, str]]):
        self.qa_pairs = qa_pairs
        self.bank_keywords = {
            # Credit card
            'credit card': 3.0, 'annual fee': 2.0, 'credit limit': 2.0, 'points': 1.8,
            'promotion': 1.8, 'application conditions': 2.5, 'application requirements': 2.5,
            'eligibility': 2.0, 'income requirement': 2.0,
            'age requirement': 2.0, 'application process': 2.0, 'apply': 2.0, 'activate': 1.8,

            # Account
            'deposit account': 2.5, 'savings account': 2.5, 'open account': 2.0, 'account': 1.8,
            'current account': 2.0,

            # Loan
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
        scores = []
        query_lower = query.lower()

        for i, qa in enumerate(self.qa_pairs):
            score = self.calculate_enhanced_similarity(query_lower, qa)
            if score > 0.5:
                scores.append((i, score, qa))
        scores.sort(key=lambda x: x[1], reverse=True)

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
        score = 0.0
        question = qa['question'].lower()
        answer = qa['answer'].lower()
        content = question + " " + answer


        if query in question:
            score += 5.0


        if query in answer:
            score += 3.0


        for keyword, weight in self.bank_keywords.items():
            keyword_lower = keyword.lower()
            if keyword_lower in query and keyword_lower in content:
                score += weight * 2
            elif keyword_lower in content:
                score += weight * 0.5

        if self.is_application_question(query) and self.contains_application_info(content):
            score += 3.0

        if self.is_fee_question(query) and self.contains_fee_info(content):
            score += 2.5

        if self.is_requirement_question(query) and self.contains_requirement_info(content):
            score += 3.0

        if len(qa['answer']) > 100:
            score += 1.0

        if any(marker in qa['answer'] for marker in ['condition', 'requirement', 'need', 'must', 'step', 'process']):
            score += 1.5

        return score

    def is_application_question(self, query: str) -> bool:
        application_words = ['apply', 'application', 'how to', 'process', 'condition', 'requirement', 'need']
        return any(word in query for word in application_words)

    def is_fee_question(self, query: str) -> bool:
        fee_words = ['fee', 'annual fee', 'service charge', 'interest rate', 'charge']
        return any(word in query for word in fee_words)

    def is_requirement_question(self, query: str) -> bool:
        requirement_words = ['condition', 'requirement', 'eligibility', 'what do I need', 'what is required']
        return any(word in query for word in requirement_words)

    def contains_application_info(self, content: str) -> bool:
        application_indicators = ['apply', 'application', 'step', 'process', 'submit', 'material', 'document']
        return any(indicator in content for indicator in application_indicators)

    def contains_fee_info(self, content: str) -> bool:
        fee_indicators = ['fee', 'annual fee', 'free', 'charge', 'interest rate', '%']
        return any(indicator in content for indicator in fee_indicators)

    def contains_requirement_info(self, content: str) -> bool:
        requirement_indicators = ['condition', 'requirement', 'must', 'need', 'age', 'income', 'document']
        return any(indicator in content for indicator in requirement_indicators)


class SmartDBSBankChatBot:

    def __init__(self, qa_data_path: str = "D:/chatchat/real_dbs_qa\dbs_qa_real_20251118_161406.json"):
        self.qa_data_path = qa_data_path
        self.qa_pairs = self.load_qa_data()

        if not self.qa_pairs:
            raise ValueError("Unable to load QA pair data")

        self.searcher = EnhancedQASearcher(self.qa_pairs)
        self.llm = self.setup_llm()

        logger.info(f"✅ Chatbot initialization completed, loaded {len(self.qa_pairs)} QA pairs")

    def load_qa_data(self) -> List[Dict[str, str]]:
        try:
            with open(self.qa_data_path, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            logger.info(f"Successfully loaded {len(qa_pairs)} QA pairs")
            return qa_pairs
        except Exception as e:
            logger.error(f"Failed to load QA pair data: {e}")
            return []

    def setup_llm(self) -> ChatOpenAI:

        return ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=1000,
            timeout=30,
            max_retries=1,
        )

    def chat(self, question: str) -> Dict[str, Any]:
        try:
            logger.info(f"User question: {question}")

            search_results = self.searcher.search(question, top_k=5)

            if not search_results:
                return self.get_fallback_response(question)

            context = self.build_detailed_context(search_results, question)


            prompt = self.build_enhanced_prompt(question, context, search_results)

            response = self.llm.invoke(prompt)
            answer = response.content

            answer = self.post_process_answer(answer, question)

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

        if any(phrase in answer for phrase in ['cannot answer', 'no information', 'insufficient information', 'please contact customer service']):
            confidence -= 0.3

        return min(max(confidence, 0.0), 1.0)

    def get_fallback_response(self, question: str) -> Dict[str, Any]:
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
        return {
            "answer": "Sorry, the system is temporarily unable to process your request. Please try again later or contact DBS official customer service for assistance.",
            "source_documents": [],
            "confidence": 0.0
        }


class InteractiveChat:


    def __init__(self, chatbot: SmartDBSBankChatBot):
        self.chatbot = chatbot
        self.chat_history = []

    def start_chat(self):
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
                    print(f"\n💡 Tip: For more accurate information, we recommend visiting the DBS official website or calling customer service.")

            except KeyboardInterrupt:
                print("\n\nThank you for using DBS Bank Smart Customer Service, goodbye!")
                break
            except Exception as e:
                print(f"\nSorry, an error occurred: {e}")
                print("Please ask again or contact human customer service.")


def main():
    print("=== DBS Bank Smart Customer Service System ===")

    try:
        print("Initializing system...")
        chatbot = SmartDBSBankChatBot()

        if not chatbot.qa_pairs:
            print("❌ Unable to load QA pair data, please check data file path")
            return

        print(f"✅ System initialization completed! Loaded {len(chatbot.qa_pairs)} QA pairs")

        print("\nEntering interactive chat mode...")
        interactive_chat = InteractiveChat(chatbot)
        interactive_chat.start_chat()

    except Exception as e:
        print(f"System startup failed: {e}")


if __name__ == "__main__":
    main()

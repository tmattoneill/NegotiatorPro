import re
import logging
from typing import Dict, List, Tuple
import tiktoken

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Intelligent text preprocessing for negotiation content to optimize token usage.
    Removes irrelevant content while preserving negotiation-critical information.
    """
    
    def __init__(self):
        # Email signature patterns
        self.signature_patterns = [
            # Phone numbers
            r'(?:Phone|Tel|Mobile|Cell)?\s*:?\s*[\+]?[\d\s\(\)\-\.]{10,}',
            # Email addresses
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # Job titles at end of lines
            r'\n[A-Z][a-z\s]+(?:Manager|Director|Executive|President|VP|CEO|CFO|CTO|Officer|Lead|Head|Specialist|Analyst|Coordinator|Representative|Consultant)\s*$',
            # Company addresses
            r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Place|Pl)\s*,?\s*[A-Za-z\s]*\d{5}',
        ]
        
        # Email footer patterns
        self.footer_patterns = [
            # Confidentiality notices
            r'(?:CONFIDENTIAL|PRIVILEGED|PRIVATE)(?:\s+AND\s+CONFIDENTIAL)?[^\n]*(?:\n[^\n]*){0,10}',
            r'This e?-?mail.*(?:confidential|privileged|proprietary).*?(?:\.|$)',
            r'The information.*(?:confidential|privileged|proprietary).*?(?:\.|$)',
            r'If you are not the intended recipient.*?(?:\.|$)',
            r'Any disclosure.*(?:prohibited|unauthorized).*?(?:\.|$)',
            
            # Environmental notices
            r'Please consider.*environment.*print.*',
            r'Think before you print.*',
            r'Save.*tree.*print.*',
            
            # Auto-generated notices
            r'This email was sent by.*',
            r'Sent from my.*',
            r'Get Outlook for.*',
            r'Virus-free.*(?:checked|scanned).*',
            
            # Legal disclaimers
            r'DISCLAIMER:.*?(?:\n\n|\Z)',
            r'This communication.*(?:disclaimer|notice).*?(?:\n\n|\Z)',
            r'The sender.*(?:responsible|liable).*?(?:\n\n|\Z)',
        ]
        
        # Forwarding header patterns
        self.forwarding_patterns = [
            r'^\s*From:.*?\n',
            r'^\s*Sent:.*?\n',
            r'^\s*To:.*?\n',
            r'^\s*Subject:.*?\n',
            r'^\s*Date:.*?\n',
            r'^\s*Cc:.*?\n',
            r'^\s*Bcc:.*?\n',
            r'-----Original Message-----.*?\n',
            r'_____+.*?\n',
        ]
        
        # Negotiation-aware stop words (words we can safely remove)
        self.safe_stopwords = {
            'a', 'an', 'the', 'is', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'up', 'about', 'into', 'through',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Words to NEVER remove (negotiation-critical)
        self.preserve_words = {
            # Emotional/negotiation words
            'urgent', 'important', 'critical', 'worried', 'concerned', 'disappointed', 'frustrated',
            'excited', 'pleased', 'satisfied', 'unhappy', 'angry', 'upset', 'concerned',
            # Action words
            'agree', 'disagree', 'accept', 'reject', 'approve', 'deny', 'confirm', 'cancel',
            'commit', 'promise', 'guarantee', 'deliver', 'provide', 'offer', 'propose',
            # Time/urgency words
            'deadline', 'urgent', 'asap', 'immediately', 'soon', 'delay', 'postpone', 'reschedule',
            # Money/value words
            'price', 'cost', 'budget', 'expensive', 'cheap', 'value', 'worth', 'pay', 'payment',
            'discount', 'premium', 'fee', 'charge', 'bill', 'invoice', 'quote', 'estimate',
            # Negotiation terms
            'terms', 'conditions', 'contract', 'agreement', 'deal', 'negotiation', 'discussion',
            'proposal', 'offer', 'counteroffer', 'compromise', 'concession', 'leverage'
        }
        
        # Initialize token encoder for counting
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        except:
            self.encoder = None
            logger.warning("Could not load tiktoken encoder for token counting")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def remove_email_signatures(self, text: str) -> str:
        """Remove email signatures and contact information"""
        for pattern in self.signature_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        return text
    
    def remove_email_footers(self, text: str) -> str:
        """Remove email footers, disclaimers, and boilerplate"""
        for pattern in self.footer_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        return text
    
    def remove_forwarding_headers(self, text: str) -> str:
        """Remove email forwarding headers"""
        for pattern in self.forwarding_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return text
    
    def clean_formatting(self, text: str) -> str:
        """Clean up formatting and whitespace issues"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
        
        # Remove repeated punctuation (but keep emphasis)
        text = re.sub(r'[!]{3,}', '!!', text)
        text = re.sub(r'[?]{3,}', '??', text)
        text = re.sub(r'[.]{4,}', '...', text)
        
        # Remove excessive dashes
        text = re.sub(r'-{4,}', '---', text)
        text = re.sub(r'_{4,}', '___', text)
        
        return text.strip()
    
    def intelligent_stopword_removal(self, text: str) -> str:
        """Remove stop words intelligently, preserving negotiation context"""
        words = text.split()
        filtered_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?";:')
            
            # Always preserve certain types of words
            if (
                word_lower in self.preserve_words or  # Negotiation-critical words
                word.isdigit() or  # Numbers
                '$' in word or '€' in word or '£' in word or  # Currency
                re.match(r'\d+[%]', word) or  # Percentages
                word.isupper() and len(word) > 1 or  # Acronyms
                '@' in word or  # Email addresses
                word[0].isupper()  # Proper nouns (names, companies)
            ):
                filtered_words.append(word)
                continue
            
            # Remove safe stop words only if they're not adding important context
            if word_lower in self.safe_stopwords:
                # Check context - keep stop words near important elements
                prev_word = words[i-1].lower().strip('.,!?";:') if i > 0 else ''
                next_word = words[i+1].lower().strip('.,!?";:') if i < len(words)-1 else ''
                
                # Keep stop words near numbers, currency, or important words
                if (
                    any(char.isdigit() for char in prev_word) or
                    any(char.isdigit() for char in next_word) or
                    prev_word in self.preserve_words or
                    next_word in self.preserve_words or
                    '$' in prev_word or '$' in next_word
                ):
                    filtered_words.append(word)
                # Otherwise, skip the stop word
            else:
                # Keep all other words
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def remove_redundant_phrases(self, text: str) -> str:
        """Remove common redundant phrases that add no value"""
        redundant_phrases = [
            r'(?:please )?let me know if you have any questions',
            r'feel free to (?:reach out|contact me|call me)',
            r'don\'t hesitate to (?:reach out|contact me|call me)',
            r'i hope this (?:email )?finds you well',
            r'i hope you\'re doing well',
            r'thank you for your (?:time|patience|understanding)',
            r'looking forward to (?:hearing from you|your (?:response|reply))',
            r'best regards?',
            r'kind regards?',
            r'sincerely yours?',
            r'yours? (?:truly|faithfully)',
            r'have a (?:great|good|nice|wonderful) (?:day|week|weekend)',
            r'thanks? in advance',
        ]
        
        for phrase in redundant_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess(self, text: str) -> Dict[str, any]:
        """
        Main preprocessing function that applies all optimizations.
        Returns both the processed text and statistics.
        """
        original_text = text
        original_tokens = self.count_tokens(original_text)
        
        # Apply preprocessing steps
        logger.info("Starting text preprocessing...")
        
        # Step 1: Remove email signatures and contact info
        text = self.remove_email_signatures(text)
        
        # Step 2: Remove email footers and disclaimers
        text = self.remove_email_footers(text)
        
        # Step 3: Remove forwarding headers
        text = self.remove_forwarding_headers(text)
        
        # Step 4: Clean formatting
        text = self.clean_formatting(text)
        
        # Step 5: Remove redundant phrases
        text = self.remove_redundant_phrases(text)
        
        # Step 6: Intelligent stop word removal (be conservative)
        text = self.intelligent_stopword_removal(text)
        
        # Final cleanup
        text = self.clean_formatting(text)  # One more cleanup pass
        
        # Calculate statistics
        processed_tokens = self.count_tokens(text)
        tokens_saved = original_tokens - processed_tokens
        reduction_percentage = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
        
        # Calculate estimated cost savings (rough estimate: $0.03 per 1K tokens for GPT-4)
        cost_per_1k_tokens = 0.03
        estimated_savings = (tokens_saved / 1000) * cost_per_1k_tokens
        
        result = {
            'original_text': original_text,
            'processed_text': text,
            'original_tokens': original_tokens,
            'processed_tokens': processed_tokens,
            'tokens_saved': tokens_saved,
            'reduction_percentage': reduction_percentage,
            'estimated_cost_savings': estimated_savings,
            'character_reduction': len(original_text) - len(text)
        }
        
        logger.info(f"Preprocessing complete: {tokens_saved} tokens saved ({reduction_percentage:.1f}% reduction)")
        
        return result
    
    def preview_changes(self, text: str) -> Dict[str, List[str]]:
        """
        Preview what would be removed without actually removing it.
        Returns categorized lists of items that would be removed.
        """
        preview = {
            'signatures': [],
            'footers': [],
            'headers': [],
            'redundant_phrases': [],
            'stop_words': []
        }
        
        # Find signatures
        for pattern in self.signature_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            preview['signatures'].extend(matches)
        
        # Find footers
        for pattern in self.footer_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            preview['footers'].extend(matches)
        
        # Find headers
        for pattern in self.forwarding_patterns:
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            preview['headers'].extend(matches)
        
        return preview
import threading
from collections import OrderedDict
import pickle
import torch

class AdvancedLangEmbeddingBuffer:
    def __init__(self, language_encoder, goal_instruction_buffer_size=10000):
        self.language_encoder = language_encoder
        self.goal_instruction_buffer_size = goal_instruction_buffer_size
        self.goal_instruction_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

    def get_or_encode_batch(self, texts):
        # print(texts)
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            with self.buffer_lock:
                uncached_texts = [text for text in texts if text not in self.goal_instruction_buffer]
            
            if uncached_texts:
                # Directly use the language encoder on the uncached texts
                encoded_batch = self.language_encoder(uncached_texts)
                
                for text, embedding in zip(uncached_texts, encoded_batch):
                    self.add_to_buffer(text, embedding)
            
            with self.buffer_lock:
                encoded_texts = [self.goal_instruction_buffer[text] for text in texts]
            
            return torch.stack(encoded_texts)

        except Exception as e:
            print(f"Error encoding texts: {e}")
            # If all else fails, return a dummy tensor
            # Assuming the output dimension of the language encoder is known
            return torch.zeros((len(texts), self.language_encoder.output_dim))

    def add_to_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.goal_instruction_buffer) >= self.goal_instruction_buffer_size:
                self.goal_instruction_buffer.popitem(last=False)
            self.goal_instruction_buffer[key] = value

    def get_goal_instruction_embedding(self, goal_instruction):
        return self.get_or_encode_batch([goal_instruction])

    def get_goal_instruction_embeddings(self, goal_instructions):
        return self.get_or_encode_batch(goal_instructions)

    def clear_buffer(self):
        with self.buffer_lock:
            self.goal_instruction_buffer.clear()

    def get_buffer_size(self):
        with self.buffer_lock:
            return len(self.goal_instruction_buffer)

    def preload_common_strings(self, goal_instruction_list):
        self.get_or_encode_batch(goal_instruction_list)

    def save_buffer(self, filepath):
        with self.buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.goal_instruction_buffer, f)

    def load_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.buffer_lock:
            self.goal_instruction_buffer = OrderedDict(list(loaded_buffer.items())[-self.goal_instruction_buffer_size:])



class VLMEmbeddingBuffer:
    """
    Buffer for caching text embeddings with proper padding
    """
    def __init__(
        self,
        tokenizer,
        language_model,
        buffer_size=10000,
        max_length=77,
        device=None,
        pad_token_id=None
    ):
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.device = device if device is not None else next(language_model.parameters()).device
        self.buffer = OrderedDict()
        self.buffer_lock = threading.Lock()
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id or 0

    def pad_sequence(self, sequences, max_len=None):
        """Pad a list of variable length Tensors to same length"""
        if max_len is None:
            max_len = max(seq.size(0) for seq in sequences)
        
        padded_seqs = []
        attention_masks = []
        
        for seq in sequences:
            pad_length = max_len - seq.size(0)
            if pad_length > 0:
                # Pad embeddings
                if seq.dim() == 2:  # For embeddings
                    padding = torch.zeros(pad_length, seq.size(1), device=self.device)
                    padded_seq = torch.cat([seq, padding], dim=0)
                else:  # For input_ids and attention_mask
                    padding = torch.zeros(pad_length, device=self.device).fill_(self.pad_token_id)
                    padded_seq = torch.cat([seq, padding], dim=0)
                
                # Create attention mask
                attention_mask = torch.cat([
                    torch.ones(seq.size(0), device=self.device),
                    torch.zeros(pad_length, device=self.device)
                ], dim=0)
            else:
                padded_seq = seq
                attention_mask = torch.ones(seq.size(0), device=self.device)
                
            padded_seqs.append(padded_seq)
            attention_masks.append(attention_mask)
            
        return padded_seqs, attention_masks

    def _compute_embeddings(self, texts):
        """Compute embeddings with padding"""
        # Tokenize texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            text_outputs = self.language_model.get_input_embeddings()(tokenized['input_ids'])

        return {
            'embeddings': text_outputs,
            'attention_mask': tokenized['attention_mask'].to(self.device),
            'input_ids': tokenized['input_ids'].to(self.device)
        }

    def get_or_encode_batch(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            with self.buffer_lock:
                uncached_texts = [text for text in texts if text not in self.buffer]
            
            if uncached_texts:
                batch_outputs = self._compute_embeddings(uncached_texts)
                
                # Ensure device consistency when storing
                for idx, text in enumerate(uncached_texts):
                    self.add_to_buffer(text, {
                        'embeddings': batch_outputs['embeddings'][idx, :batch_outputs['attention_mask'][idx].sum()].to(self.device),
                        'attention_mask': batch_outputs['attention_mask'][idx, :batch_outputs['attention_mask'][idx].sum()].to(self.device),
                        'input_ids': batch_outputs['input_ids'][idx, :batch_outputs['attention_mask'][idx].sum()].to(self.device)
                    })
            
            # Retrieve and ensure all cached outputs are on correct device
            with self.buffer_lock:
                cached_outputs = [
                    {k: v.to(self.device) for k, v in self.buffer[text].items()}
                    for text in texts
                ]
            
            # Get lists ensuring device consistency
            embeddings_list = [o['embeddings'].to(self.device) for o in cached_outputs]
            input_ids_list = [o['input_ids'].to(self.device) for o in cached_outputs]
            
            # Pad sequences
            padded_embeddings, attention_masks = self.pad_sequence(embeddings_list)
            padded_input_ids, _ = self.pad_sequence(input_ids_list)
            
            # Stack ensuring device consistency
            return {
                'embeddings': torch.stack(padded_embeddings).to(self.device),
                'attention_mask': torch.stack(attention_masks).to(self.device),
                'input_ids': torch.stack(padded_input_ids).to(self.device)
            }
        except Exception as e:
            print(f"Error processing texts: {e}")
            print(f"Text samples: {texts[:2]}")
            batch_size = len(texts)
            return {
                'embeddings': torch.zeros((batch_size, self.max_length, self.language_model.config.hidden_size), device=self.device),
                'attention_mask': torch.zeros((batch_size, self.max_length), device=self.device),
                'input_ids': torch.zeros((batch_size, self.max_length), device=self.device, dtype=torch.long)
            }
            
    def add_to_buffer(self, key, value):
        """Add item to buffer with proper device placement"""
        with self.buffer_lock:
            if len(self.buffer) >= self.buffer_size:
                self.buffer.popitem(last=False)
            # Ensure all items are moved to the correct device
            self.buffer[key] = {k: v.to(self.device) for k, v in value.items()}

    def save_buffer(self, filepath):
        """Save buffer to disk"""
        with self.buffer_lock:
            cpu_buffer = OrderedDict()
            for k, v in self.buffer.items():
                cpu_buffer[k] = {tk: tv.cpu() for tk, tv in v.items()}
            with open(filepath, 'wb') as f:
                pickle.dump(cpu_buffer, f)

    def load_buffer(self, filepath):
        """Load buffer from disk"""
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.buffer_lock:
            self.buffer = OrderedDict()
            for k, v in list(loaded_buffer.items())[-self.buffer_size:]:
                self.buffer[k] = {tk: tv.to(self.device) for tk, tv in v.items()}
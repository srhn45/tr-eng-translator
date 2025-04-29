import torch
import sentencepiece as spm
from transformer_copy import Transformer
import torch.nn.functional as F

class InferBottle:
    def __init__(self, model_path="../model/best_model.pth", device="cuda" if torch.cuda.is_available() else "cpu", vocab_size=50000,
                 en_tokenizer_path="../data/en_spm.model", tr_tokenizer_path="../data/tr_spm.model", beam_width=5, max_len=500, sampling_method="all"):

        sp_en = spm.SentencePieceProcessor()
        sp_en.load(en_tokenizer_path)
        self.en_tokenizer = sp_en

        sp_tr = spm.SentencePieceProcessor()
        sp_tr.load(tr_tokenizer_path)
        self.tr_tokenizer = sp_tr

        self.vocab_size = vocab_size
        self.model = Transformer(vocab_size=self.vocab_size, embed_dim=512, ff_dim=2048, num_heads=8, n_encoders=4, n_decoders=4)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.sampling_method = sampling_method
        self.beam_width = beam_width
        self.max_len = max_len

        self.BOS_ID = sp_tr.bos_id()
        self.EOS_ID = sp_tr.eos_id()
        self.PAD_ID = sp_en.pad_id() if sp_en.pad_id() != -1 else 0

        assert sampling_method in ["greedy", "top_p", "beam_search", "all"], "Invalid sampling method. Choose one of 'greedy', 'top_p', 'beam_search', or 'all'."

    def translate(self, text):
        input_ids = self.en_tokenizer.encode(text, out_type=int)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        src_key_padding_mask = (input_ids == self.PAD_ID)

        def decode_sequence(decoded_ids):
            return self.tr_tokenizer.decode(decoded_ids)

        if self.sampling_method in ["greedy", "all"]:
            decoded_ids = self._greedy_decode(input_ids, src_key_padding_mask)
            greedy_output = decode_sequence(decoded_ids)

        if self.sampling_method in ["top_p", "all"]:
            decoded_ids = self._top_p_decode(input_ids, src_key_padding_mask)
            top_p_output = decode_sequence(decoded_ids)

        if self.sampling_method in ["beam_search", "all"]:
            decoded_ids = self.beam_search(input_ids, src_key_padding_mask)
            beam_output = decode_sequence(decoded_ids)

        if self.sampling_method == "greedy":
            return {
                "text": text,
                "greedy": greedy_output,
            }
        elif self.sampling_method == "top_p":
            return {
                "text": text,
                "top_p": top_p_output,
            }
        elif self.sampling_method == "beam_search":
            return {
                "text": text,
                "beam_search": beam_output
            }
        elif self.sampling_method == "all":
            return {
                "text": text,
                "greedy": greedy_output,
                "top_p": top_p_output,
                "beam_search": beam_output
            }

    def _greedy_decode(self, input_ids, src_key_padding_mask):
        decoded_ids = [self.BOS_ID]
        for _ in range(self.max_len):
            tgt_input = torch.tensor(decoded_ids).unsqueeze(0).to(self.device)
            tgt_key_padding_mask = (tgt_input == 0)

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    tgt_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )[0, -1, :]

            next_token = self.greedy_sampling(logits) # simply taking the argmax over the logits
            if next_token == self.EOS_ID:
                break
            decoded_ids.append(next_token)
        return decoded_ids[1:]

    def _top_p_decode(self, input_ids, src_key_padding_mask):
        decoded_ids = [self.BOS_ID]
        for _ in range(self.max_len):
            tgt_input = torch.tensor(decoded_ids).unsqueeze(0).to(self.device)
            tgt_key_padding_mask = (tgt_input == 0)

            with torch.no_grad():
                logits = self.model(
                    input_ids,
                    tgt_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )[0, -1, :]

            for token in set(decoded_ids):  # to discourage repetition, can be disabled for more advanced models
                logits[token] /= 1.2

            next_token = self.top_p_sample(logits, p=0.9, temperature=0.8) # random sampling from the top-p cumulative probability mass (0.9)
            if next_token == self.EOS_ID:
                break
            decoded_ids.append(next_token)
        return decoded_ids[1:]

    def beam_search(self, input_ids, src_key_padding_mask):
        beams = [([self.BOS_ID], 0.0, False)]
        for _ in range(self.max_len):
            candidates = []

            for seq, score, has_ended in beams:
                if has_ended:
                    candidates.append((seq, score, True))
                    continue

                tgt_input = torch.tensor(seq).unsqueeze(0).to(self.device)
                tgt_key_padding_mask = (tgt_input == 0)

                with torch.no_grad():
                    logits = self.model(
                        input_ids,
                        tgt_input,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )[0, -1, :]

                probs = F.log_softmax(logits, dim=-1) # log softmax for numerical stability
                topk_probs, topk_indices = torch.topk(probs, self.beam_width)

                for log_prob, token_id in zip(topk_probs.tolist(), topk_indices.tolist()):
                    new_seq = seq + [token_id]
                    new_score = score + log_prob
                    ended = token_id == self.EOS_ID
                    candidates.append((new_seq, new_score, ended))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_width] # keep the top-beam_width sequences

            if all(ended for _, _, ended in beams): # break if all sequences have ended
                break

        best_seq = beams[0][0]
        if best_seq[0] == self.BOS_ID:
            best_seq = best_seq[1:]
        if best_seq and best_seq[-1] == self.EOS_ID:
            best_seq = best_seq[:-1]
        return best_seq

    @staticmethod
    def greedy_sampling(logits):
        return torch.argmax(logits, dim=-1).item()

    @staticmethod
    def top_p_sample(logits, p=0.9, temperature=1.0):
        logits = logits / temperature # temperature scaling to control randomness/exploration
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone() # shift the mask to the right to include the first token where cumulative_probs >= p
        sorted_mask[..., 0] = False # to always keep the first token

        sorted_probs[sorted_mask] = 0
        sorted_probs /= sorted_probs.sum() # normalizing the probabilities

        next_token = sorted_indices[torch.multinomial(sorted_probs, 1)] # randomly sampling from the distribution
        return next_token.item()

def main():
    print("🔥 InferBottle Translator 🔥\n")
    
    while True:
        method = input("\nChoose method [greedy/top_p/beam/all] (write 'exit' to exit!): ").lower()
        if method not in ['greedy', 'top_p', 'beam', 'all']:
            print("❌ Invalid method! Must be one of 'greedy', 'top_p', 'beam' or 'all'.")
            continue

        infer = InferBottle(model_path="../model/best_model.pth", sampling_method=method)
            
        text = input("Enter English text (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
            
        print("\n🔄 Translating...")
        outputs = infer.translate(text)

        print("\n📝 Results:")
        for method, text in outputs.items():
            print(f"{method}: {text}")
            
    print("\n Bye.")

if __name__ == "__main__":
    main()
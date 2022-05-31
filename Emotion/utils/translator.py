import tqdm
from typing import Literal, List
from omegaconf import DictConfig

from transformers import MarianTokenizer, AutoModelForSeq2SeqLM


class RusEngTranslator:
    def __init__(
        self,
        mt_config: DictConfig, 
        kind: Literal['ru_en', 'en_ru'] = 'ru_en',
        device: str = 'cuda:0'
        ):

        self._device = device
        model_name = mt_config['name'][kind]
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _switch_device(self):
        self._old_device = next(iter(self.translator.parameters())).device
        if self._old_device != self._device:
            self.translator.to(self._device)

    def translate_text(self, text: str) -> str:
        """ Translates one text """

        self._switch_device()

        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self._device)
        outputs = self.translator.generate(input_ids)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.translator.to(self._device)
        return decoded

    def translate_texts(
        self, 
        texts: List[str], 
        verbose: bool = False, 
        desc: str = ''
        ) -> List[str]:
        """ Translates list of texts """

        self._switch_device()

        if verbose:
            texts = tqdm.tqdm(texts, desc=desc)

        decoded_texts = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self._device)
            outputs = self.translator.generate(input_ids)

            sents = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                sents.append(decoded)
            sents = " ".join(sents)
            decoded_texts.append(sents)

        self.translator.to(self._device)
        return decoded_texts

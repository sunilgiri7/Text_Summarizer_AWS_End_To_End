from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
        pipe = pipeline("summarization", model=self.config.model_name_or_path, tokenizer=tokenizer)
        # return summarizer(text, max_length=self.config.max_length, min_length=self.config.min_length, num_beams=self.config.num_beams, early_stopping=True)
        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel summary")
        print(output)
        return output
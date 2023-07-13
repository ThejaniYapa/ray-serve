from ray import serve
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'EleutherAI/pythia-70m'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    async def __call__(self, request):
        text = await request.json()
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs)
        prediction = self.tokenizer.decode(outputs[0])
        return prediction

client = serve.start()
client.create_backend("model_backend", Model)
client.create_endpoint("model_endpoint", backend="model_backend", route="/predict")

print("Model is served at: http://localhost:8000/predict")

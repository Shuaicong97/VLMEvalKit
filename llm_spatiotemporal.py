# First: export GOOGLE_API_KEY="Your API_KEY"

from vlmeval.config import supported_VLM
model = supported_VLM['Idefics3-8B-Llama3']()

# 1. Gemini
# model = supported_VLM['GeminiPro2-5']()
# 2. GPT-4V
#model = supported_VLM['gpt-5.1-2025-11-13']()
# 3. Claude
#model = supported_VLM['Claude4_Sonnet']()



# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
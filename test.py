import requests
import json

# First API call with reasoning
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "openai/gpt-5.2-codex",
    "messages": [
        {
          "role": "user",
          "content": "How many r's are in the word 'strawberry'?"
        }
      ],
    "reasoning": {"effort": "xhigh"},
    "temperature": 0.0,
  })
)

# Extract the assistant message with reasoning_details
response = response.json()
response = response['choices'][0]['message']

# Preserve the assistant message with reasoning_details
messages = [
  {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
  {
    "role": "assistant",
    "content": response.get('content'),
    "reasoning_details": response.get('reasoning_details')  # Pass back unmodified
  },
  {"role": "user", "content": "Are you sure? Think carefully."}
]

print(messages)

# Second API call - model continues reasoning from where it left off
response2 = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "openai/gpt-5.2-codex",
    "messages": messages,  # Includes preserved reasoning_details
    "reasoning": {"effort": "xhigh"}
  })
)

response2 = response2.json()

print(response2)